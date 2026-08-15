"""Disk cache for every remote fetch.

One JSON file per key under ``data/cache/{key}.json`` holding
``{"fetched_at": iso8601, "data": ...}``. Two TTL regimes are in play:

* live FPL endpoints use ``config.cache_ttl_seconds`` (300s) because Fastly
  caches them for exactly that long — polling faster gains nothing;
* historical files (football-data.co.uk, the vaastav archive) and
  ``element-summary`` only change after matches are played, so they take
  ``TTL_HISTORICAL``. Without that split a warm ``load_game_state`` would
  re-issue 587 player requests every five minutes.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any, List, Optional

from gaffer.core.config import CACHE_DIR

log = logging.getLogger(__name__)

DEFAULT_TTL = 300
TTL_HISTORICAL = 24 * 3600
# A negative TTL means "never expires"; 0 means "always stale".
TTL_FOREVER = -1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_KEY_LEN = 120


def safe_key(key: str) -> str:
    """Filesystem-safe filename stem for an arbitrary cache key."""
    cleaned = _UNSAFE.sub("_", key).strip("._-")
    if not cleaned:
        cleaned = "key"
    if len(cleaned) > _MAX_KEY_LEN:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        cleaned = cleaned[: _MAX_KEY_LEN - 13] + "_" + digest
    return cleaned


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(text: str) -> Optional[datetime]:
    if not text:
        return None
    raw = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class Cache:
    def __init__(self, cache_dir: Optional[str] = None, default_ttl: int = DEFAULT_TTL) -> None:
        self.dir = cache_dir or CACHE_DIR
        self.default_ttl = default_ttl
        os.makedirs(self.dir, exist_ok=True)

    # -- paths --------------------------------------------------------------
    def path_for(self, key: str) -> str:
        return os.path.join(self.dir, safe_key(key) + ".json")

    def has(self, key: str) -> bool:
        return os.path.exists(self.path_for(key))

    def age_seconds(self, key: str) -> Optional[float]:
        """Seconds since this key was written, or None if it is not cached."""
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (ValueError, OSError):
            return None
        fetched = _parse_iso(payload.get("fetched_at", ""))
        if fetched is None:
            # Fall back to the file mtime if the stamp is unreadable.
            return max(0.0, _utcnow().timestamp() - os.path.getmtime(path))
        return max(0.0, (_utcnow() - fetched).total_seconds())

    def is_fresh(self, key: str, ttl: Optional[int] = None) -> bool:
        ttl = self.default_ttl if ttl is None else ttl
        age = self.age_seconds(key)
        if age is None:
            return False
        if ttl < 0:
            return True
        return age < ttl

    # -- read / write -------------------------------------------------------
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Cached value, or None when absent, unreadable or older than ttl."""
        ttl = self.default_ttl if ttl is None else ttl
        path = self.path_for(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (ValueError, OSError) as exc:
            log.warning("cache: unreadable entry %s (%s); ignoring", key, exc)
            return None
        if not isinstance(payload, dict) or "data" not in payload:
            log.warning("cache: malformed entry %s; ignoring", key)
            return None
        if ttl >= 0:
            fetched = _parse_iso(payload.get("fetched_at", ""))
            if fetched is None:
                return None
            if (_utcnow() - fetched).total_seconds() >= ttl:
                return None
        return payload["data"]

    def set(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        payload = {"fetched_at": _utcnow().isoformat(), "data": value}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write to a sibling temp file and rename: a killed process must never
        # leave a half-written JSON file that later reads as corrupt.
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def clear(self, pattern: str = "*") -> int:
        """Delete cached keys matching a glob. Returns the number removed."""
        removed = 0
        for name in os.listdir(self.dir):
            if not name.endswith(".json"):
                continue
            key = name[: -len(".json")]
            if fnmatch.fnmatch(key, pattern) or fnmatch.fnmatch(key, safe_key(pattern)):
                os.unlink(os.path.join(self.dir, name))
                removed += 1
        return removed

    def keys(self, pattern: str = "*") -> List[str]:
        out = []
        for name in sorted(os.listdir(self.dir)):
            if not name.endswith(".json"):
                continue
            key = name[: -len(".json")]
            if fnmatch.fnmatch(key, pattern):
                out.append(key)
        return out


if __name__ == "__main__":
    import shutil
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tmpdir = tempfile.mkdtemp(prefix="gaffer-cache-smoke-")
    try:
        c = Cache(tmpdir, default_ttl=2)
        c.set("bootstrap", {"elements": [1, 2, 3]})
        print("round trip        :", c.get("bootstrap"))
        print("key on disk       :", os.path.basename(c.path_for("element-summary/12/")))
        c.set("element-summary/12/", {"history": []})
        print("unsafe key value  :", c.get("element-summary/12/"))
        print("age (s)           : %.3f" % c.age_seconds("bootstrap"))
        print("fresh @ttl=2      :", c.is_fresh("bootstrap"))
        time.sleep(2.1)
        print("fresh after 2.1s  :", c.is_fresh("bootstrap"), "-> get:", c.get("bootstrap"))
        print("get ttl=-1        :", c.get("bootstrap", ttl=TTL_FOREVER) is not None)
        long_key = "x" * 400
        c.set(long_key, 1)
        print("long key len      :", len(os.path.basename(c.path_for(long_key))))
        print("keys              :", c.keys())
        print("cleared           :", c.clear("*"), "-> remaining", c.keys())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
