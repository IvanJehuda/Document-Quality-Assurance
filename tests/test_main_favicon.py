from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_favicon_is_served_as_svg():
    resp = client.get("/favicon.svg")

    assert resp.status_code == 200
    assert "image/svg+xml" in resp.headers["content-type"]
    assert b"<svg" in resp.content


def test_index_links_the_favicon():
    resp = client.get("/")

    assert resp.status_code == 200
    assert 'rel="icon"' in resp.text
    assert "/favicon.svg" in resp.text
