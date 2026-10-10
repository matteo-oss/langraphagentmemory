"""
main.py — FastAPI Application Entry Point

This module defines the FastAPI application for the text analysis agent.
It exposes three endpoints:
  - GET  /health          → Health check for monitoring / load balancers
  - POST /analyze         → Synchronous full analysis (returns when complete)
  - POST /analyze/stream  → Streaming analysis via SSE (streams node by node)

Authentication is handled via an optional API key mechanism (see verify_api_key).
If the API_KEY environment variable is not set, auth is skipped (useful for local dev).
"""

import json
import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.graph import agent
from app.api.models import AnalyzeRequest, AnalyzeResponse, HealthResponse, StreamChunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("API_KEY")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

api = FastAPI(
    title="Text Analysis Agent",
    description=(
        "A production-ready LangGraph agent that classifies text, "
        "extracts entities, analyzes sentiment, and generates summaries. "
        "Supports both synchronous and streaming (SSE) responses."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def verify_api_key(x_api_key: str = Header(default=None)) -> bool:
    """
    Dependency that validates the X-API-Key request header.

    Behavior:
        - If API_KEY env var is not set → skip auth entirely (dev mode).
        - If API_KEY is set but header is missing → raise 401 Unauthorized.
        - If API_KEY is set but header value is wrong → raise 403 Forbidden.
        - If both match → return True and allow the request through.

    Args:
        x_api_key: Value of the X-API-Key request header (injected by FastAPI).

    Returns:
        True if the request is authorized.

    Raises:
        HTTPException 401: Missing API key header.
        HTTPException 403: Invalid API key value.
    """
    if not API_KEY:
        # No key configured → open access (useful for local development)
        return True

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@api.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Used by load balancers, container orchestration (e.g., Kubernetes),
    and monitoring tools to verify that the service is alive and ready.

    Returns:
        HealthResponse with status "ok".
    """
    return HealthResponse(status="ok", message="Service is running")


@api.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    authorized: bool = Depends(verify_api_key),
) -> AnalyzeResponse:
    """
    Synchronous text analysis endpoint.

    Runs the full LangGraph pipeline and waits for all nodes to complete
    before returning the result. Use this when you need the full analysis
    at once and don't care about intermediate progress.

    Pipeline executed:
        classification → [entity_extraction + sentiment] → summarization

    Args:
        request: The analysis request. Must contain a non-empty 'text' field.
        authorized: Injected by verify_api_key; ensures the request is authenticated.

    Returns:
        AnalyzeResponse containing:
            - classification: Category of the text (e.g., "News", "Blog").
            - entities: Named entities (only for News / Research categories).
            - sentiment: Detected sentiment (e.g., "Positive", "Negative").
            - summary: A short summary of the input text.

    Example request body:
        {
            "text": "OpenAI launched GPT-4o, a new multimodal model..."
        }
    """
    result = agent.invoke({"text": request.text})

    return AnalyzeResponse(
        classification=result["classification"],
        entities=result.get("entities"),
        sentiment=result["sentiment"],
        summary=result["summary"],
    )


@api.post("/analyze/stream")
async def analyze_stream(
    request: AnalyzeRequest,
    authorized: bool = Depends(verify_api_key),
) -> StreamingResponse:
    """
    Streaming text analysis endpoint using Server-Sent Events (SSE).

    Runs the same LangGraph pipeline as /analyze, but streams each node's
    result to the client as soon as it completes — without waiting for the
    full pipeline to finish.

    This is useful for:
        - Showing real-time progress in a UI.
        - Reducing perceived latency for the end user.
        - Allowing clients to react to partial results immediately.

    SSE format (one chunk per completed node):
        data: {"node": "classification", "result": "News"}\\n\\n
        data: {"node": "sentiment",      "result": "Positive"}\\n\\n
        data: {"node": "entities",       "result": "OpenAI, GPT-4o"}\\n\\n
        data: {"node": "summary",        "result": "OpenAI announced..."}\\n\\n

    Args:
        request: The analysis request. Must contain a non-empty 'text' field.
        authorized: Injected by verify_api_key; ensures the request is authenticated.

    Returns:
        A StreamingResponse with media type 'text/event-stream'.
        Each event is a JSON-encoded StreamChunk with 'node' and 'result' fields.
    """

    async def event_generator():
        """
        Async generator that listens to LangGraph node events and yields
        SSE-formatted strings as each node completes.

        node_map translates internal LangGraph node names into the clean
        field names we want to expose in the API response.
        """
        node_map = {
            "classification_node": "classification",
            "entity_extraction": "entities",
            "sentiment": "sentiment",
            "summarization": "summary",
        }

        async for event in agent.astream_events(
            {"text": request.text}, version="v2"
        ):
            # We only care about node completion events, not intermediate ones
            if event["event"] != "on_chain_end":
                continue

            node_name = event.get("name")
            if node_name not in node_map:
                continue

            output = event.get("data", {}).get("output", {})
            field = node_map[node_name]
            result = output.get(field)

            if not result:
                continue

            chunk = StreamChunk(node=field, result=result)
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )