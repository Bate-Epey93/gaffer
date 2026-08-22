"""Client for the official FPL API.

Everything goes through :class:`Cache`. The API is Fastly-fronted with
``max-age=300``, so polling faster than the configured TTL buys nothing and
risks a block; always send a browser User-Agent (bare clients get 403s).

The one performance-critical call is :meth:`FPLClient.all_element_summaries`,
which needs one request per player (587 in 2026/27). It runs on a small thread
pool with a per-request stagger, caches each response individually, and treats
an individual failure as a logged skip rather than a fatal error — a single
flaky player must not kill a projection run.
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

import requests

from gaffer.core.config import FPL_API_BASE, USER_AGENT, Config
from gaffer.data.cache import Cache

log = logging.getLogger(__name__)

# Per-player history only changes when matches are played, so it does not need
# the 5-minute live TTL. Without this a warm run would refetch 587 files.
ELEMENT_SUMMARY_TTL = 6 * 3600
# Finished gameweeks are immutable once bonus is applied; unfinished ones are not.
LIVE_TTL = 300

RETRY_STATUS = (429, 500, 502, 503, 504)


class FPLError(RuntimeError):
    """Any non-recoverable failure talking to the FPL API."""


class FPLNotFound(FPLError):
    """A 404. Expected pre-season for entry picks, which do not exist yet."""


class FPLClient:
    def __init__(
        self,
        config: Config,
        cache: Cache,
        max_retries: int = 4,
        timeout: float = 30.0,
        max_workers: int = 8,
        stagger_seconds: float = 0.03,
    ) -> None:
        self.config = config
        self.cache = cache
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = min(max_workers, 8)  # be a polite citizen
        self.stagger_seconds = stagger_seconds
        self.base = FPL_API_BASE
        self._local = threading.local()
        # Populated by all_element_summaries so callers can report data gaps.
        self.failed_element_summaries: List[int] = []

    # -- plumbing -----------------------------------------------------------
    @property
    def _session(self) -> requests.Session:
        """One Session per thread: Session is not documented as thread-safe."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "en-GB,en;q=0.9",
                }
            )
            self._local.session = session
        return session

    def _fetch(self, url: str) -> Any:
        last_error = None
        for attempt in range(self.max_retries):
            if attempt:
                # Deterministic exponential backoff; no jitter so reruns match.
                time.sleep(0.8 * (2 ** (attempt - 1)))
            try:
                resp = self._session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_error = exc
                log.warning("GET %s failed (%s), attempt %d", url, exc, attempt + 1)
                continue
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise FPLError("GET %s returned non-JSON: %s" % (url, exc))
            if resp.status_code == 404:
                raise FPLNotFound("GET %s -> 404" % url)
            if resp.status_code in RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 30.0))
                    except ValueError:
                        pass
                last_error = FPLError("GET %s -> HTTP %d" % (url, resp.status_code))
                log.warning("GET %s -> %d, attempt %d", url, resp.status_code, attempt + 1)
                continue
            raise FPLError("GET %s -> HTTP %d" % (url, resp.status_code))
        raise FPLError("GET %s failed after %d attempts: %s" % (url, self.max_retries, last_error))

    def _get(
        self,
        path: str,
        key: str,
        ttl: Optional[int] = None,
        force: bool = False,
    ) -> Any:
        ttl = self.config.cache_ttl_seconds if ttl is None else ttl
        if not force:
            cached = self.cache.get(key, ttl=ttl)
            if cached is not None:
                return cached
        data = self._fetch(self.base.rstrip("/") + path)
        self.cache.set(key, data)
        return data

    # -- endpoints ----------------------------------------------------------
    def bootstrap(self, force: bool = False) -> Dict[str, Any]:
        return self._get("/bootstrap-static/", "bootstrap", force=force)

    def fixtures(self, force: bool = False) -> List[Dict[str, Any]]:
        return self._get("/fixtures/", "fixtures", force=force)

    def element_summary(self, player_id: int, force: bool = False) -> Dict[str, Any]:
        return self._get(
            "/element-summary/%d/" % player_id,
            "element_summary_%d" % player_id,
            ttl=ELEMENT_SUMMARY_TTL,
            force=force,
        )

    def live(self, gw: int, force: bool = False) -> Dict[str, Any]:
        return self._get("/event/%d/live/" % gw, "live_%d" % gw, ttl=LIVE_TTL, force=force)

    def entry(self, entry_id: int) -> Dict[str, Any]:
        return self._get("/entry/%d/" % entry_id, "entry_%d" % entry_id)

    def h2h_matches(self, league_id: int, entry_id: int) -> Dict[str, Any]:
        """A head-to-head league's fixture list for one entry, all 38 weeks."""
        return self._get(
            "/leagues-h2h-matches/league/%d/?page=1&entry=%d" % (league_id, entry_id),
            "h2h_%d_%d" % (league_id, entry_id))

    def entry_history(self, entry_id: int) -> Dict[str, Any]:
        return self._get("/entry/%d/history/" % entry_id, "entry_%d_history" % entry_id)

    def entry_picks(self, entry_id: int, gw: int) -> Dict[str, Any]:
        # 404s before the GW1 deadline; the caller must handle FPLNotFound.
        return self._get(
            "/entry/%d/event/%d/picks/" % (entry_id, gw),
            "entry_%d_picks_%d" % (entry_id, gw),
        )

    def event_status(self) -> Dict[str, Any]:
        return self._get("/event-status/", "event_status")

    def set_piece_notes(self) -> Dict[str, Any]:
        return self._get("/team/set-piece-notes/", "set_piece_notes")

    # -- bulk ---------------------------------------------------------------
    def all_element_summaries(
        self,
        player_ids: Sequence[int],
        force: bool = False,
        progress: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        """Fetch every player's detail page. Failures are logged, never raised.

        Returns ``{player_id: summary}`` for everything that succeeded; the ids
        that failed are left in ``self.failed_element_summaries``.
        """
        self.failed_element_summaries = []
        results: Dict[int, Dict[str, Any]] = {}
        pending: List[int] = []

        for pid in player_ids:
            if force:
                pending.append(pid)
                continue
            cached = self.cache.get("element_summary_%d" % pid, ttl=ELEMENT_SUMMARY_TTL)
            if cached is None:
                pending.append(pid)
            else:
                results[pid] = cached

        if not pending:
            return results

        total = len(pending)
        if progress:
            sys.stderr.write(
                "element-summary: %d cached, fetching %d with %d workers\n"
                % (len(results), total, self.max_workers)
            )
            sys.stderr.flush()

        done = 0
        step = max(1, total // 10)
        started = time.time()

        def work(pid: int) -> Any:
            # Stagger inside the worker so the pool does not fire 8 requests in
            # the same millisecond on every wave.
            if self.stagger_seconds:
                time.sleep(self.stagger_seconds)
            return self.element_summary(pid, force=force)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(work, pid): pid for pid in pending}
            for fut in as_completed(futures):
                pid = futures[fut]
                done += 1
                try:
                    results[pid] = fut.result()
                except Exception as exc:  # a bad player must not kill the run
                    self.failed_element_summaries.append(pid)
                    log.warning("element-summary %d failed: %s", pid, exc)
                if progress and (done % step == 0 or done == total):
                    sys.stderr.write(
                        "element-summary: %d/%d (%.0f%%) %.1fs\n"
                        % (done, total, 100.0 * done / total, time.time() - started)
                    )
                    sys.stderr.flush()

        if self.failed_element_summaries:
            log.warning(
                "element-summary: %d of %d failed: %s",
                len(self.failed_element_summaries),
                total,
                sorted(self.failed_element_summaries)[:20],
            )
        return results

    # -- convenience --------------------------------------------------------
    def player_ids(self, force: bool = False) -> List[int]:
        return [int(e["id"]) for e in self.bootstrap(force=force)["elements"]]


def default_client(config: Optional[Config] = None) -> FPLClient:
    config = config or Config.load()
    return FPLClient(config, Cache(default_ttl=config.cache_ttl_seconds))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = Config.load()
    client = default_client(cfg)

    t0 = time.time()
    boot = client.bootstrap()
    print("bootstrap        : %d elements, %d teams, %d events (%.2fs)"
          % (len(boot["elements"]), len(boot["teams"]), len(boot["events"]), time.time() - t0))

    t0 = time.time()
    fx = client.fixtures()
    print("fixtures         : %d (%.2fs)" % (len(fx), time.time() - t0))

    status = client.event_status()
    print("event-status     : %s" % (status.get("status") or status))

    notes = client.set_piece_notes()
    print("set-piece-notes  : %d teams" % len(notes.get("teams", [])))

    one = client.element_summary(1)
    print("element-summary 1: history=%d history_past=%d fixtures=%d"
          % (len(one["history"]), len(one["history_past"]), len(one["fixtures"])))

    ids = client.player_ids()
    t0 = time.time()
    summaries = client.all_element_summaries(ids)
    print("all summaries    : %d/%d ok, %d failed (%.1fs)"
          % (len(summaries), len(ids), len(client.failed_element_summaries), time.time() - t0))

    t0 = time.time()
    summaries = client.all_element_summaries(ids)
    print("warm summaries   : %d ok (%.2fs)" % (len(summaries), time.time() - t0))

    try:
        client.live(1)
        print("live gw1         : ok")
    except FPLError as exc:
        print("live gw1         : %s" % exc)
