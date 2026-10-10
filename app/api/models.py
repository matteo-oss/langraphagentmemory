from pydantic import BaseModel, Field
from typing import Optional


class AnalyzeRequest(BaseModel):
    """
    Request model for the text analysis endpoint.

    Attributes:
        text: The raw text to analyze. Must be non-empty.
        include_entities: Whether to force entity extraction
                          regardless of classification. Defaults to False.
    """
    text: str = Field(..., min_length=1, description="The text to analyze")
    include_entities: Optional[bool] = Field(
        default=False,
        description="Force entity extraction regardless of classification"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "OpenAI has announced GPT-4 in San Francisco.",
                "include_entities": False
            }
        }
    }


class AnalyzeResponse(BaseModel):
    """
    Response model for the text analysis endpoint.

    Attributes:
        classification: The category of the text (News, Blog, Research, Other).
        entities: Extracted entities, if applicable.
        sentiment: Emotional tone (Positive, Negative, Neutral).
        summary: One-sentence summary of the text.
    """
    classification: str
    entities: Optional[str] = None
    sentiment: str
    summary: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "classification": "News",
                "entities": "OpenAI, GPT-4, San Francisco",
                "sentiment": "Positive",
                "summary": "OpenAI announced GPT-4 in San Francisco."
            }
        }
    }


class StreamChunk(BaseModel):
    """
    A single chunk in a streaming response.

    Each chunk represents the completion of one node in the graph.

    Attributes:
        node: The name of the node that just completed.
        result: The output produced by that node.
    """
    node: str
    result: str


class HealthResponse(BaseModel):
    """
    Response model for the health check endpoint.

    Attributes:
        status: Current API status.
        message: Human-readable status message.
    """
    status: str
    message: str