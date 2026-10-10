"""
mcp_client.py — MCP Client + Text Analysis Integration

This module is the "brain" of the operation. It:
  1. Starts the MCP server as a subprocess
  2. Discovers available tools from the server
  3. Sends user queries to an LLM with the tool list
  4. Executes tools when the LLM decides to use them
  5. Pipes story content directly into our text analysis agent

Usage:
    python mcp_hn/mcp_client.py

Make sure mcp_server.py is in the same directory before running.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Path setup — allow importing from the parent project (text analysis agent)
# ---------------------------------------------------------------------------

# Add project root to path so we can import from app/
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.agent.graph import agent  # Our text analysis LangGraph agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

# ---------------------------------------------------------------------------
# Step 1: Tool Discovery
# ---------------------------------------------------------------------------


async def discover_tools(session: ClientSession) -> list[dict]:
    """
    Ask the MCP server what tools it has available.

    This is the "Discovery Phase" — before doing anything, we ask the server
    to introduce itself. The returned tool list is later sent to the LLM so
    it knows what capabilities it has.

    Args:
        session: An active MCP client session.

    Returns:
        A list of dicts, each containing tool name, description, and schema.
    """
    tools_result = await session.list_tools()
    tool_info = []

    for tool_type, tool_list in tools_result:
        if tool_type == "tools":
            for tool in tool_list:
                tool_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema,
                })

    return tool_info


# ---------------------------------------------------------------------------
# Step 2: Tool Execution
# ---------------------------------------------------------------------------


async def execute_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    Execute a specific tool on the MCP server and return its result.

    This is the "Execution Phase" — once the LLM decides which tool to use,
    we call it here and get back the raw result.

    Args:
        session: An active MCP client session.
        tool_name: The name of the tool to call (e.g. "get_top_stories").
        arguments: A dict of arguments to pass to the tool.

    Returns:
        The tool's output as a string.
    """
    print(f"\n⚙️  Executing tool: '{tool_name}' with args: {arguments}")
    result = await session.call_tool(tool_name, arguments)

    # MCP returns a list of content blocks — we join them into one string
    output = ""
    for block in result.content:
        if hasattr(block, "text"):
            output += block.text

    return output


# ---------------------------------------------------------------------------
# Step 3: Text Analysis Integration
# ---------------------------------------------------------------------------


def analyze_with_agent(text: str, user_id: str = "default_user") -> dict:
    """
    Pass text through our LangGraph text analysis agent.

    This is what connects MCP to our existing project — instead of just
    returning raw HackerNews content, we run it through the full pipeline:
    classification → entity extraction → sentiment → summary.

    Args:
        text: The story content to analyze.

    Returns:
        A dict with classification, entities, sentiment, and summary.
    """
    print("\n🔬 Sending content to text analysis agent...")
    result = agent.invoke({
        "text": text,
        "user_id": user_id,
    })

    return {
        "classification": result.get("classification"),
        "entities": result.get("entities"),
        "sentiment": result.get("sentiment"),
        "summary": result.get("summary"),
    }


# ---------------------------------------------------------------------------
# Step 4: LLM Decision Making
# ---------------------------------------------------------------------------


async def query_with_tools(
    session: ClientSession,
    tool_info: list[dict],
    user_query: str,
) -> str:
    """
    Send a user query to the LLM along with available tools.

    The LLM reads the tool descriptions and decides:
    - If the query needs a tool → respond with JSON specifying which tool to use
    - If the query can be answered directly → respond in plain text

    After tool execution, the result is optionally passed to our text analysis
    agent if the content is suitable for analysis.

    Args:
        session: An active MCP client session.
        tool_info: List of available tools (from discover_tools).
        user_query: The user's question or request.

    Returns:
        A final natural language response.
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Build a description of all available tools for the LLM
    tool_descriptions = "\n\n".join([
        f"Tool: {t['name']}\nDescription: {t['description']}\nSchema: {json.dumps(t['schema'], indent=2)}"
        for t in tool_info
    ])

    system_prompt = f"""You are an AI assistant with access to HackerNews tools.

Available tools:
{tool_descriptions}

When you need to use a tool, respond ONLY with a JSON object in this exact format:
{{
    "tool": "tool_name",
    "arguments": {{
        "param1": "value1"
    }}
}}

Do not add any text before or after the JSON when using a tool.
For questions that don't need tools, respond normally in plain text.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query),
    ]

    print("\n🧠 LLM is deciding which tool to use...")
    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Check if the LLM wants to use a tool
    try:
        tool_request = json.loads(response_text)

        if "tool" in tool_request and "arguments" in tool_request:
            tool_name = tool_request["tool"]
            arguments = tool_request["arguments"]

            # Execute the tool
            tool_result = await execute_tool(session, tool_name, arguments)

            # If the tool fetched story content → run it through our agent
            if tool_name == "get_story_content":
                analysis = analyze_with_agent(tool_result, user_id="default_user")

                final_response = f"""
📰 Story fetched successfully. Here is the full analysis:

🏷️  Classification : {analysis['classification']}
😊 Sentiment      : {analysis['sentiment']}
🔍 Entities       : {analysis['entities'] or 'N/A'}
📝 Summary        : {analysis['summary']}
"""
                return final_response

            # For other tools, send the result back to LLM for interpretation
            followup_messages = messages + [
                response,
                HumanMessage(content=f"Tool result:\n{tool_result}\n\nPlease summarize this for the user."),
            ]

            followup_response = llm.invoke(followup_messages)
            return followup_response.content

    except (json.JSONDecodeError, KeyError):
        # LLM chose to answer directly without a tool
        pass

    return response_text


# ---------------------------------------------------------------------------
# Step 5: Interactive Chat Session
# ---------------------------------------------------------------------------


async def chat_session():
    """
    Run an interactive chat session with the MCP-powered HackerNews agent.

    The session:
      1. Starts the MCP server as a subprocess
      2. Discovers available tools
      3. Enters a loop where the user can ask questions
      4. Automatically uses tools and optionally runs text analysis

    Type 'exit' or 'quit' to end the session.
    """
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
    )

    print("🚀 Starting MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover tools once at the start
            tool_info = await discover_tools(session)

            print(f"\n✅ Connected to MCP server. {len(tool_info)} tools available:")
            for tool in tool_info:
                print(f"   • {tool['name']} — {tool['description'][:60]}...")

            print("\n" + "=" * 50)
            print("💬 HackerNews Analysis Agent")
            print("=" * 50)
            print("Examples:")
            print("  'What are the top stories right now?'")
            print("  'Search for stories about machine learning'")
            print("  'Analyze story 12345'")
            print("Type 'exit' to quit.\n")

            # Chat loop
            while True:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break

                response = await query_with_tools(session, tool_info, user_input)
                print(f"\nAgent: {response}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(chat_session())