"""Authentication for the HTTP API.

Locally this server is a private tool on 127.0.0.1 and a password would be pure
friction. Deployed, it is on the public internet and two of its routes are
genuinely dangerous to leave open:

  POST /api/refresh   fires 587 requests at the FPL API. Anyone who finds the
                      URL can loop it until Render's egress IP is rate-limited
                      or blocked, which breaks the data for the one person the
                      tool exists for.
  POST /api/optimize  runs a MILP solve that peaks near 200 MB. A handful of
                      concurrent calls will OOM a 512 MB instance.

So the rule here is: exposure demands a password. The middleware fails CLOSED —
if the server is reachable from anywhere but loopback and no password is set, it
refuses every request with a 503 that says exactly what to do, rather than
serving happily and quietly to the whole internet. Forgetting to set the
variable is a mistake you find immediately, not one you find in your FPL data.

Credentials are accepted three ways, all checked in constant time:

  Authorization: Basic <base64 user:pass>   the browser's native prompt, which
                                            iOS stores in the keychain, so an
                                            installed PWA asks once
  Authorization: Bearer <password>          scripts and curl
  X-Gaffer-Key: <password>                  fetch() from the dashboard

/api/health is always open: Render's health check cannot send credentials, and a
health probe that 401s reads as a dead service and rolls the deploy back. It
exposes no player data and no way to spend CPU.
"""
from __future__ import annotations

import base64
import binascii
import ipaddress
import os
import secrets
from typing import Optional, Set

from starlette.requests import Request
from starlette.responses import JSONResponse

# Open routes. Keep this list as short as it can possibly be.
PUBLIC_PATHS: Set[str] = {
    "/api/health",
}

DEFAULT_USERNAME = "gaffer"
REALM = "gaffer"


def configured_password() -> str:
    """The shared secret, or "" when none is set."""
    return (os.environ.get("GAFFER_PASSWORD") or "").strip()


def configured_username() -> str:
    return (os.environ.get("GAFFER_USERNAME") or "").strip() or DEFAULT_USERNAME


def auth_disabled() -> bool:
    """Explicit opt-out, for running exposed on a network you already trust.

    Deliberately awkward to set by accident, and it is the only way to serve a
    non-loopback address without a password.
    """
    return (os.environ.get("GAFFER_AUTH_DISABLED") or "").strip() == "i-know-what-im-doing"


def is_loopback(host: Optional[str]) -> bool:
    """True when the peer is this machine, so local use needs no password."""
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Unix socket, or a hostname we cannot parse: treat as remote. Failing
        # closed on an unknown peer is the whole point of this module.
        return False


def _matches(supplied: str, expected: str) -> bool:
    """Constant-time compare, so a wrong password leaks no timing signal."""
    return secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def _credentials_ok(request: Request, password: str) -> bool:
    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    scheme = scheme.lower()

    if scheme == "bearer" and value:
        return _matches(value.strip(), password)

    if scheme == "basic" and value:
        try:
            decoded = base64.b64decode(value.strip(), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return False
        user, _, supplied = decoded.partition(":")
        # Both halves are compared, and both in constant time, so neither the
        # username nor the password can be probed character by character.
        user_ok = _matches(user, configured_username())
        pass_ok = _matches(supplied, password)
        return user_ok and pass_ok

    key = request.headers.get("x-gaffer-key")
    if key:
        return _matches(key.strip(), password)

    return False


def _unauthorized() -> JSONResponse:
    # The WWW-Authenticate header is what makes Safari and Chrome show their own
    # login prompt and offer to save the result, which is the whole reason Basic
    # is supported at all.
    return JSONResponse(
        {"error": "unauthorized",
         "detail": "gaffer is password-protected. Supply HTTP Basic credentials, "
                   "an Authorization: Bearer token, or an X-Gaffer-Key header."},
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="%s", charset="UTF-8"' % REALM},
    )


def _misconfigured() -> JSONResponse:
    return JSONResponse(
        {"error": "server_not_configured",
         "detail": "gaffer is reachable from outside this machine but GAFFER_PASSWORD "
                   "is not set, so it is refusing to serve. Set GAFFER_PASSWORD in the "
                   "environment and restart. To serve an already-trusted network without "
                   "a password, set GAFFER_AUTH_DISABLED=i-know-what-im-doing."},
        status_code=503,
    )


async def auth_middleware(request: Request, call_next):
    """Gate every request that is not explicitly public."""
    path = request.url.path

    # CORS preflight carries no credentials by design; the CORS middleware
    # answers it and the real request that follows is still gated.
    if request.method == "OPTIONS" or path in PUBLIC_PATHS:
        return await call_next(request)

    if auth_disabled():
        return await call_next(request)

    password = configured_password()
    client = request.client.host if request.client else None

    if not password:
        # No password configured: local use is fine, exposure is not.
        if is_loopback(client):
            return await call_next(request)
        return _misconfigured()

    if _credentials_ok(request, password):
        return await call_next(request)
    return _unauthorized()


def startup_warning(host: str) -> Optional[str]:
    """A line for the CLI to print when `serve` is about to expose the server.

    Returns None when the configuration is safe.
    """
    if auth_disabled():
        return ("auth is DISABLED (GAFFER_AUTH_DISABLED is set). Every request is "
                "served without a password.")
    if configured_password():
        return None
    try:
        exposed = not ipaddress.ip_address(host).is_loopback
    except ValueError:
        exposed = True
    if exposed:
        return ("binding %s with no GAFFER_PASSWORD set: every request will be "
                "refused with 503 until you set one." % host)
    return None
