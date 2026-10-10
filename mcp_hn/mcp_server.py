"""
mcp_server.py — MCP Server for HackerNews

This server exposes three tools that fetch content from HackerNews.
It runs as a standalone process and communicates via stdio using the MCP protocol.

Tools:
  - get_top_stories(limit)       → Fetch trending stories right now
  - search_stories(query, limit) → Search stories by keyword
  - get_story_content(story_id)  → Fetch full content of a specific story

Run directly to test:
    python mcp_server.py

Or connect via Claude Desktop / MCP client.
"""

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Server initialization
# ---------------------------------------------------------------------------

mcp = FastMCP("hackernews-server")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HN_BASE_URL = "https://hacker-news.firebaseio.com/v0"
HN_ALGOLIA_URL = "http://hn.algolia.com/api/v1"

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_top_stories(limit: int = 10) -> str:
    """
    Fetch the top trending stories from HackerNews right now.

    Use this when the user wants to know what's popular or trending on HackerNews.

    Args:
        limit: Number of stories to fetch (default: 10, max recommended: 30).

    Returns:
        A formatted string with story titles, URLs, scores, and comment counts.
    """
    # Step 1: Get list of top story IDs
    response = httpx.get(f"{HN_BASE_URL}/topstories.json")
    story_ids = response.json()[:limit]

    # Step 2: Fetch details for each story
    stories = []
    for story_id in story_ids:
        story_response = httpx.get(f"{HN_BASE_URL}/item/{story_id}.json")
        story = story_response.json()

        if story and story.get("type") == "story":
            stories.append({
                "id": story.get("id"),
                "title": story.get("title", "No title"),
                "url": story.get("url", "No URL"),
                "score": story.get("score", 0),
                "comments": story.get("descendants", 0),
            })

    # Step 3: Format the output as readable text
    if not stories:
        return "No stories found."

    result = f"Top {len(stories)} stories on HackerNews:\n\n"
    for i, story in enumerate(stories, 1):
        result += f"{i}. {story['title']}\n"
        result += f"   ID: {story['id']}\n"
        result += f"   URL: {story['url']}\n"
        result += f"   Score: {story['score']} points | Comments: {story['comments']}\n\n"

    return result


@mcp.tool()
def search_stories(query: str, limit: int = 10) -> str:
    """
    Search HackerNews stories by keyword or topic.

    Use this when the user asks about a specific topic (e.g. 'machine learning',
    'startups', 'OpenAI'). Returns the most relevant stories for the query.

    Args:
        query: Search term or topic to look for.
        limit: Number of results to return (default: 10).

    Returns:
        A formatted string with matching story titles, URLs, and scores.
    """
    # Use Algolia's HackerNews search API
    response = httpx.get(
        f"{HN_ALGOLIA_URL}/search",
        params={
            "query": query,
            "tags": "story",
            "hitsPerPage": limit,
        }
    )
    data = response.json()
    hits = data.get("hits", [])

    if not hits:
        return f"No stories found for query: '{query}'"

    result = f"Search results for '{query}' ({len(hits)} stories):\n\n"
    for i, hit in enumerate(hits, 1):
        result += f"{i}. {hit.get('title', 'No title')}\n"
        result += f"   ID: {hit.get('objectID', 'N/A')}\n"
        result += f"   URL: {hit.get('url', 'No URL')}\n"
        result += f"   Score: {hit.get('points', 0)} points\n\n"

    return result


@mcp.tool()
def get_story_content(story_id: int) -> str:
    """
    Fetch the full content and details of a specific HackerNews story by its ID.

    Use this when the user wants to analyze or read a specific story.
    The returned text can be passed directly to the text analysis agent.

    Args:
        story_id: The HackerNews story ID (visible in get_top_stories output).

    Returns:
        A formatted string with the story title, URL, score, and full text if available.
        If the story has no text body (e.g. it's a link), returns the title and URL only.
    """
    response = httpx.get(f"{HN_BASE_URL}/item/{story_id}.json")
    story = response.json()

    if not story:
        return f"Story with ID {story_id} not found."

    title = story.get("title", "No title")
    url = story.get("url", "No URL")
    score = story.get("score", 0)
    comments = story.get("descendants", 0)
    text = story.get("text", "")  # Only available for Ask HN / Show HN posts

    result = f"Story ID: {story_id}\n"
    result += f"Title: {title}\n"
    result += f"URL: {url}\n"
    result += f"Score: {score} points | Comments: {comments}\n\n"

    if text:
        # HN stores text as HTML — strip basic tags for cleaner output
        import re
        clean_text = re.sub(r"<[^>]+>", " ", text).strip()
        result += f"Content:\n{clean_text}\n"
    else:
        result += "Content: This story links to an external URL (no text body on HackerNews).\n"
        result += f"To analyze the content, fetch the article directly from: {url}\n"

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")