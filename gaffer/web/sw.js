/* gaffer service worker — offline shell, honest data.
 *
 * Served from "/" (never from a subdirectory) so its scope covers the whole
 * origin, and served with `Cache-Control: no-cache` so the browser always
 * revalidates this file. See `/sw.js` in gaffer/api/server.py.
 *
 * The two rules that shape everything below:
 *
 * 1. The **shell** (index, CSS, JS, icons) is immutable per build. The server
 *    stamps every shell URL with the file's mtime (`app.js?v=1755…`) and
 *    injects the stamped list into SHELL, plus a BUILD hash into the cache
 *    name. So a stamped URL can be served cache-first with no revalidation at
 *    all, and editing any shell file changes this file's bytes -> the browser
 *    sees a new worker -> a new cache is filled and the old one deleted.
 *
 * 2. **API responses are never served from cache while the network works.**
 *    A projection is a decision input with a deadline attached; showing a
 *    silently stale one is worse than showing nothing. So /api is network
 *    first, the cache is a last resort for a phone that cannot reach the Mac,
 *    and every response that comes out of the cache carries
 *    `X-Gaffer-Cached-At`, which app.js turns into a loud banner.
 */

const CACHE_VERSION = "v1";                  // bump by hand to invalidate everything
const BUILD = "__GAFFER_BUILD__";            // replaced at serve time with a shell hash

/* Replaced at serve time with the mtime-stamped shell URLs. The literal below
   is the fallback for anyone opening this file straight off disk, and doubles
   as the documentation of what the shell actually is. */
const SHELL_INJECTED = "__GAFFER_SHELL__";
const SHELL = Array.isArray(SHELL_INJECTED) ? SHELL_INJECTED : [
  "/",
  "/styles.css",
  "/sample.js",
  "/ui.js",
  "/views.js",
  "/app.js",
  "/manifest.webmanifest",
  "/icons/icon-180.png",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-512-maskable.png"
];

const SHELL_CACHE = "gaffer-shell-" + CACHE_VERSION + "-" + BUILD;
/* Deliberately *not* keyed on BUILD: the last-known-good projections should
   survive a code edit, they are the only thing standing between the user and a
   blank screen on a train. */
const API_CACHE = "gaffer-api-" + CACHE_VERSION;

/* Paths whose answers must never be replayed. /health is how the app decides
   the backend is alive, /refresh and /optimize mutate or cost minutes, and the
   docs are noise. */
const NEVER_CACHE = [/^\/api\/health/, /^\/api\/refresh/, /^\/api\/docs/, /^\/api\/openapi/];

// ------------------------------------------------------------------ install --

self.addEventListener("install", function (event) {
  event.waitUntil(
    caches.open(SHELL_CACHE).then(function (cache) {
      // Individually, and tolerantly: one 404 must not fail the whole install
      // and leave the app with no worker at all. Whatever misses here is
      // fetched and cached on first use instead.
      return Promise.all(SHELL.map(function (url) {
        return cache.add(new Request(url, { cache: "reload" })).catch(function (err) {
          console.warn("[sw] precache miss", url, err && err.message);
        });
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

// ----------------------------------------------------------------- activate --

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (names) {
      return Promise.all(names.map(function (name) {
        if (name.indexOf("gaffer-") !== 0) return null;        // not ours
        if (name === SHELL_CACHE || name === API_CACHE) return null;
        return caches.delete(name);
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

// -------------------------------------------------------------------- fetch --

function isNeverCached(path) {
  for (var i = 0; i < NEVER_CACHE.length; i++) {
    if (NEVER_CACHE[i].test(path)) return true;
  }
  return false;
}

/* Store a copy stamped with the time it was stored. The stamp is added only to
   the *cached* copy, so a response that reaches the page carrying
   X-Gaffer-Cached-At is, by construction, one the network could not supply. */
function putStamped(cacheName, request, response) {
  var copy = response.clone();
  return copy.blob().then(function (body) {
    var headers = new Headers(copy.headers);
    headers.set("X-Gaffer-Cached-At", new Date().toISOString());
    return caches.open(cacheName).then(function (cache) {
      return cache.put(request, new Response(body, {
        status: copy.status, statusText: copy.statusText, headers: headers
      }));
    });
  }).catch(function () { /* quota, opaque body, whatever: caching is optional */ });
}

function networkFirst(event, cacheName) {
  var request = event.request;
  return fetch(request).then(function (response) {
    if (response && response.ok) {
      event.waitUntil(putStamped(cacheName, request, response));
    }
    return response;
  }).catch(function (err) {
    // The page aborted it (its own timeout budget expired) — it does not want
    // an answer any more, and a stale one would be a lie about a live call.
    if (err && err.name === "AbortError") throw err;
    return caches.open(cacheName)
      .then(function (cache) { return cache.match(request, { ignoreVary: true }); })
      .then(function (hit) {
        if (hit) return hit;
        throw err;
      });
  });
}

self.addEventListener("fetch", function (event) {
  var request = event.request;
  if (request.method !== "GET") return;                     // POST /optimize etc: straight through

  var url;
  try { url = new URL(request.url); } catch (e) { return; }
  if (url.origin !== self.location.origin) return;          // we proxy nothing off-origin

  // The document itself: network first, so a phone in range always gets the
  // current build, and the cached copy only ever covers a dead backend.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).then(function (response) {
        if (response && response.ok) {
          event.waitUntil(putStamped(SHELL_CACHE, new Request("/"), response));
        }
        return response;
      }).catch(function () {
        return caches.open(SHELL_CACHE).then(function (cache) {
          return cache.match("/", { ignoreSearch: true, ignoreVary: true });
        }).then(function (hit) {
          return hit || new Response(
            "<!doctype html><meta charset=utf-8><title>gaffer — offline</title>" +
            "<body style=\"background:#07080a;color:#e9edf3;font:14px -apple-system,sans-serif;padding:2rem\">" +
            "<h1 style=\"color:#3ee0b8\">gaffer is offline</h1><p>The Mac is not reachable and " +
            "nothing is cached yet. Open this once while connected and it will work offline after that.</p>",
            { status: 503, headers: { "Content-Type": "text/html; charset=utf-8" } });
        });
      })
    );
    return;
  }

  if (url.pathname.indexOf("/api/") === 0) {
    if (isNeverCached(url.pathname)) return;                // straight to the network, no fallback
    event.respondWith(networkFirst(event, API_CACHE));
    return;
  }

  // Shell assets. The URL carries the build stamp, so a hit is by definition
  // the right bytes for this build and needs no revalidation.
  event.respondWith(
    caches.open(SHELL_CACHE).then(function (cache) {
      return cache.match(request).then(function (hit) {
        if (hit) return hit;
        return fetch(request).then(function (response) {
          if (response && response.ok && response.type === "basic") {
            event.waitUntil(putStamped(SHELL_CACHE, request, response));
          }
          return response;
        }).catch(function (err) {
          // Offline and this exact stamp was never cached (an asset changed
          // while the phone was away): fall back to any stamp we do hold.
          return cache.match(request, { ignoreSearch: true, ignoreVary: true })
            .then(function (stale) {
              if (stale) return stale;
              throw err;
            });
        });
      });
    })
  );
});
