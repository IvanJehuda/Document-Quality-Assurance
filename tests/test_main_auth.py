"""Auth is a cookie-session gate over the data APIs (see main.py). The app shell
(/) is served openly so the frontend can render it blurred behind a login card
and reveal it in place after /api/login succeeds.

A local .env may define APP_USERNAME/APP_PASSWORD, but conftest's autouse fixture
clears them, so each test starts with auth disabled and turns it on explicitly
via monkeypatch. A fresh TestClient per test keeps each cookie jar isolated."""

from fastapi.testclient import TestClient

import main

USER = "user@example.com"
PASS = "s3cret*#"


def _client() -> TestClient:
    return TestClient(main.app)


def test_shell_and_apis_open_when_credentials_unset(monkeypatch):
    monkeypatch.delenv("APP_USERNAME", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)

    client = _client()
    assert client.get("/").status_code == 200
    status = client.get("/api/auth-status").json()
    assert status == {"auth_required": False, "authenticated": True}


def test_shell_stays_open_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    # The shell must load (blurred) without a session; only data APIs are gated.
    assert _client().get("/").status_code == 200


def test_auth_status_reports_locked_when_enabled_and_no_session(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    status = _client().get("/api/auth-status").json()
    assert status == {"auth_required": True, "authenticated": False}


def test_data_api_returns_401_without_session(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    assert _client().get("/api/tables").status_code == 401


def test_login_wrong_password_is_rejected(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    client = _client()
    resp = client.post("/api/login", data={"username": USER, "password": "nope"})
    assert resp.status_code == 401
    assert resp.json()["ok"] is False
    assert main.SESSION_COOKIE not in client.cookies
    # Still locked out afterwards.
    assert client.get("/api/tables").status_code == 401


def test_login_success_sets_session_and_unlocks_apis(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    client = _client()
    resp = client.post("/api/login", data={"username": USER, "password": PASS})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert main.SESSION_COOKIE in resp.cookies

    # Cookie now in the jar: auth-status flips and gated APIs pass the middleware.
    assert client.get("/api/auth-status").json()["authenticated"] is True
    assert client.get("/api/tables").status_code != 401


def test_logout_clears_session(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    client = _client()
    client.post("/api/login", data={"username": USER, "password": PASS})
    assert client.get("/api/tables").status_code != 401

    client.post("/api/logout")
    assert client.get("/api/tables").status_code == 401


def test_health_stays_open_when_auth_enabled(monkeypatch):
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    assert _client().get("/health").status_code == 200


def test_session_cookie_is_samesite_none_secure_over_https(monkeypatch):
    """Over HTTPS (e.g. Hugging Face's cross-site iframe) the cookie must be
    SameSite=None; Secure, otherwise it is withheld on later /api/* calls and
    every request after login 401s."""
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    resp = _client().post(
        "/api/login",
        data={"username": USER, "password": PASS},
        headers={"x-forwarded-proto": "https"},
    )
    set_cookie = resp.headers["set-cookie"].lower()
    assert "samesite=none" in set_cookie
    assert "secure" in set_cookie


def test_session_cookie_is_lax_over_plain_http(monkeypatch):
    """Local dev over plain http can't use Secure, so fall back to Lax."""
    monkeypatch.setenv("APP_USERNAME", USER)
    monkeypatch.setenv("APP_PASSWORD", PASS)

    resp = _client().post("/api/login", data={"username": USER, "password": PASS})
    set_cookie = resp.headers["set-cookie"].lower()
    assert "samesite=lax" in set_cookie
    assert "secure" not in set_cookie
