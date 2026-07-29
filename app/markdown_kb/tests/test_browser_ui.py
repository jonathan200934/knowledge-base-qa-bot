from fastapi.testclient import TestClient

from app.main import app


def test_root_serves_browser_ui():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Knowledge Base Q&A" in response.text
    assert "question-input" in response.text


def test_static_assets_are_served():
    client = TestClient(app)

    js_response = client.get("/static/app.js")
    css_response = client.get("/static/styles.css")

    assert js_response.status_code == 200
    assert 'fetch("/chat"' in js_response.text
    assert css_response.status_code == 200
    assert ".result-panel" in css_response.text
