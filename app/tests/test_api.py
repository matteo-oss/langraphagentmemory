"""
tests/test_api.py — API Endpoint Tests

Tests for all three endpoints of the text analysis agent:
  - GET  /health
  - POST /analyze
  - POST /analyze/stream

Uses FastAPI's built-in TestClient (based on httpx) which runs the app
in a synchronous context — no real server needed.

Run with:
    pytest -xvs tests/test_api.py
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.main import api

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

client = TestClient(api)

SAMPLE_NEWS_TEXT = (
    "OpenAI announced the release of GPT-4o, a new multimodal model "
    "capable of processing text, audio, and images in real time. "
    "The model was unveiled at a live event in San Francisco."
)

SAMPLE_BLOG_TEXT = (
    "Today I want to share my experience learning Python. "
    "It has been a fun journey and I highly recommend it to beginners."
)

# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


def test_health_check_status_code():
    """Health check should return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_body():
    """Health check should return status 'ok'."""
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "message" in body


# ---------------------------------------------------------------------------
# POST /analyze — Synchronous
# ---------------------------------------------------------------------------


def test_analyze_returns_200():
    """Analyze endpoint should return HTTP 200 for valid input."""
    response = client.post("/analyze", json={"text": SAMPLE_NEWS_TEXT})
    assert response.status_code == 200


def test_analyze_response_has_required_fields():
    """Analyze response must contain classification, sentiment, and summary."""
    response = client.post("/analyze", json={"text": SAMPLE_NEWS_TEXT})
    body = response.json()
    assert "classification" in body
    assert "sentiment" in body
    assert "summary" in body


def test_analyze_news_has_entities():
    """News text should trigger entity extraction — entities field should be populated."""
    response = client.post("/analyze", json={"text": SAMPLE_NEWS_TEXT})
    body = response.json()
    # News/Research category → entities should be present
    assert body.get("entities") is not None
    assert len(body["entities"]) > 0


def test_analyze_blog_may_skip_entities():
    """
    Blog text should not trigger entity extraction.
    The entities field should be None or absent.
    """
    response = client.post("/analyze", json={"text": SAMPLE_BLOG_TEXT})
    body = response.json()
    # Blog category → entities should be None
    assert body.get("entities") is None


def test_analyze_empty_text_returns_422():
    """
    Empty text should fail Pydantic validation.
    FastAPI returns HTTP 422 Unprocessable Entity for invalid input.
    """
    response = client.post("/analyze", json={"text": ""})
    assert response.status_code == 422


def test_analyze_missing_text_returns_422():
    """Missing 'text' field should fail Pydantic validation."""
    response = client.post("/analyze", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /analyze/stream — Streaming SSE
# ---------------------------------------------------------------------------


def test_stream_returns_200():
    """Stream endpoint should return HTTP 200."""
    with client.stream("POST", "/analyze/stream", json={"text": SAMPLE_NEWS_TEXT}) as response:
        assert response.status_code == 200


def test_stream_content_type_is_sse():
    """Stream endpoint should respond with text/event-stream content type."""
    with client.stream("POST", "/analyze/stream", json={"text": SAMPLE_NEWS_TEXT}) as response:
        assert "text/event-stream" in response.headers["content-type"]


def test_stream_yields_valid_chunks():
    """
    Each line from the stream should follow the SSE format:
        data: {"node": "...", "result": "..."}

    All chunks must be valid JSON with 'node' and 'result' fields.
    """
    chunks = []

    with client.stream("POST", "/analyze/stream", json={"text": SAMPLE_NEWS_TEXT}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                chunks.append(payload)

    assert len(chunks) > 0, "Expected at least one SSE chunk"

    for chunk in chunks:
        assert "node" in chunk, f"Missing 'node' in chunk: {chunk}"
        assert "result" in chunk, f"Missing 'result' in chunk: {chunk}"
        assert len(chunk["result"]) > 0, f"Empty result in chunk: {chunk}"


def test_stream_covers_expected_nodes():
    """
    The stream should yield chunks for classification, sentiment, and summary.
    For news text, entities should also be present.
    """
    received_nodes = set()

    with client.stream("POST", "/analyze/stream", json={"text": SAMPLE_NEWS_TEXT}) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
                received_nodes.add(payload["node"])

    assert "classification" in received_nodes
    assert "sentiment" in received_nodes
    assert "summary" in received_nodes


def test_stream_empty_text_returns_422():
    """Empty text should fail validation even on the stream endpoint."""
    response = client.post("/analyze/stream", json={"text": ""})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Authentication (only runs if API_KEY is set in environment)
# ---------------------------------------------------------------------------


def test_auth_missing_key_returns_401(monkeypatch):
    """If API_KEY is set in env, missing header should return 401."""
    monkeypatch.setenv("API_KEY", "secret-test-key")

    # Re-import to pick up the new env var
    import importlib
    import app.api.main as main_module
    importlib.reload(main_module)
    auth_client = TestClient(main_module.api)

    response = auth_client.post("/analyze", json={"text": SAMPLE_NEWS_TEXT})
    assert response.status_code == 401


def test_auth_wrong_key_returns_403(monkeypatch):
    """If API_KEY is set in env, wrong key should return 403."""
    monkeypatch.setenv("API_KEY", "secret-test-key")

    import importlib
    import app.api.main as main_module
    importlib.reload(main_module)
    auth_client = TestClient(main_module.api)

    response = auth_client.post(
        "/analyze",
        json={"text": SAMPLE_NEWS_TEXT},
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 403


def test_auth_correct_key_returns_200(monkeypatch):
    """Correct API key should allow the request through."""
    monkeypatch.setenv("API_KEY", "secret-test-key")

    import importlib
    import app.api.main as main_module
    importlib.reload(main_module)
    auth_client = TestClient(main_module.api)

    response = auth_client.post(
        "/analyze",
        json={"text": SAMPLE_NEWS_TEXT},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert response.status_code == 200