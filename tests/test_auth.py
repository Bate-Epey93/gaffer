"""The password gate.

These tests exist because the failure mode is silent and expensive: a server
that serves the whole internet without complaining looks exactly like a server
that is working. The important assertions here are the negative ones — that an
exposed server with no password refuses to serve, and that /api/health stays
open so Render's health check does not roll a good deploy back.
"""
from __future__ import annotations

import base64

import pytest

from gaffer.api import auth

pytest.importorskip("httpx", reason="starlette's TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402  (after importorskip)

from gaffer.api.server import create_app  # noqa: E402

PASSWORD = "correct-horse-battery-staple"
LOOPBACK = ("127.0.0.1", 5000)
REMOTE = ("203.0.113.7", 5000)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Never let the developer's own shell decide the outcome of a test."""
    monkeypatch.delenv("GAFFER_PASSWORD", raising=False)
    monkeypatch.delenv("GAFFER_USERNAME", raising=False)
    monkeypatch.delenv("GAFFER_AUTH_DISABLED", raising=False)


def basic(user: str, password: str) -> dict:
    token = base64.b64encode(("%s:%s" % (user, password)).encode()).decode()
    return {"Authorization": "Basic " + token}


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setenv("GAFFER_PASSWORD", PASSWORD)
    return TestClient(create_app(), client=REMOTE)


# --- the gate is closed ----------------------------------------------------

def test_no_credentials_is_rejected(secured):
    assert secured.get("/api/players").status_code == 401


def test_dashboard_itself_is_gated(secured):
    """The HTML is as private as the API; it is the whole tool."""
    assert secured.get("/").status_code == 401


def test_wrong_password_is_rejected(secured):
    assert secured.get("/api/players", headers=basic("gaffer", "nope")).status_code == 401


def test_wrong_username_is_rejected(secured):
    assert secured.get("/api/players", headers=basic("admin", PASSWORD)).status_code == 401


def test_malformed_basic_header_is_rejected_not_crashed(secured):
    for value in ("Basic ", "Basic !!!not-base64!!!", "Basic " + base64.b64encode(b"\xff\xfe").decode()):
        assert secured.get("/api/players", headers={"Authorization": value}).status_code == 401


def test_401_invites_the_browser_prompt(secured):
    """Without this header iOS never offers to save the password."""
    header = secured.get("/api/players").headers.get("www-authenticate", "")
    assert header.lower().startswith("basic")
    assert 'realm="gaffer"' in header


# --- the gate opens for the right key --------------------------------------

@pytest.mark.parametrize("headers", [
    basic("gaffer", PASSWORD),
    {"Authorization": "Bearer " + PASSWORD},
    {"X-Gaffer-Key": PASSWORD},
])
def test_every_supported_credential_form_is_accepted(secured, headers):
    assert secured.get("/api/players", headers=headers).status_code == 200


def test_username_is_configurable(monkeypatch):
    monkeypatch.setenv("GAFFER_PASSWORD", PASSWORD)
    monkeypatch.setenv("GAFFER_USERNAME", "hardpro")
    client = TestClient(create_app(), client=REMOTE)
    assert client.get("/api/players", headers=basic("hardpro", PASSWORD)).status_code == 200
    assert client.get("/api/players", headers=basic("gaffer", PASSWORD)).status_code == 401


# --- health stays open -----------------------------------------------------

def test_health_needs_no_credentials(secured):
    """Render's probe cannot authenticate; a 401 here rolls back a good deploy."""
    assert secured.get("/api/health").status_code == 200


def test_health_is_open_even_when_unconfigured():
    assert TestClient(create_app(), client=REMOTE).get("/api/health").status_code == 200


# --- failing closed --------------------------------------------------------

def test_exposed_without_a_password_refuses_to_serve():
    """The point of the whole module: no password + reachable = serve nothing."""
    response = TestClient(create_app(), client=REMOTE).get("/api/players")
    assert response.status_code == 503
    assert response.json()["error"] == "server_not_configured"
    assert "GAFFER_PASSWORD" in response.json()["detail"]


def test_loopback_without_a_password_still_works():
    """Local use must not need a password, or nobody will run it locally."""
    assert TestClient(create_app(), client=LOOPBACK).get("/api/players").status_code == 200


def test_opt_out_serves_without_a_password(monkeypatch):
    monkeypatch.setenv("GAFFER_AUTH_DISABLED", "i-know-what-im-doing")
    assert TestClient(create_app(), client=REMOTE).get("/api/players").status_code == 200


def test_opt_out_requires_the_exact_phrase(monkeypatch):
    """"1"/"true" must NOT disable auth: too easy to set by accident."""
    for value in ("1", "true", "yes", "on", ""):
        monkeypatch.setenv("GAFFER_AUTH_DISABLED", value)
        assert not auth.auth_disabled()
        assert TestClient(create_app(), client=REMOTE).get("/api/players").status_code == 503


# --- the startup warning the CLI prints ------------------------------------

def test_startup_warning_fires_on_a_public_bind():
    assert "GAFFER_PASSWORD" in (auth.startup_warning("0.0.0.0") or "")


def test_startup_warning_is_silent_on_loopback():
    assert auth.startup_warning("127.0.0.1") is None


def test_startup_warning_is_silent_when_configured(monkeypatch):
    monkeypatch.setenv("GAFFER_PASSWORD", PASSWORD)
    assert auth.startup_warning("0.0.0.0") is None


def test_unparseable_peer_is_treated_as_remote():
    """An unknown peer must fail closed, not open."""
    assert not auth.is_loopback("testclient")
    assert not auth.is_loopback(None)
    assert auth.is_loopback("127.0.0.1")
    assert auth.is_loopback("::1")
