"""
Agent Core - LLM Agent Initialization Module

This module provides the core functionality for initializing a Gemini based
LLM agent with MCP Neo4j tools for querying the IYP knowledge graph.

Components:
    - MCP Neo4j client for Cypher query execution
    - Google Gemini LLM integration via LangChain
    - System prompt construction with schema injection

Usage:
    agent, system_prompt = await initialize_agent(api_key="your_key")
"""

import os
import json
import functools
from typing import Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from .prompts import get_prompt
except ImportError:
    from prompts import get_prompt


# =============================================================================
# TOOL RESULT TRUNCATION
# =============================================================================

# Maximum items included in the content the LLM sees.
# Full results are always preserved in ToolMessage.artifact for export/comparison.
MAX_TOOL_RESULT_ITEMS = 50


def _truncate_json_content(content: str) -> str:
    """
    Truncate a JSON-array string if it exceeds MAX_TOOL_RESULT_ITEMS.

    Returns the original string unchanged when it is not a large JSON list.
    """
    if not isinstance(content, str):
        return content
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    if isinstance(parsed, list) and len(parsed) > MAX_TOOL_RESULT_ITEMS:
        total = len(parsed)
        truncated = parsed[:MAX_TOOL_RESULT_ITEMS]
        truncated.append({
            "__note": f"Showing {MAX_TOOL_RESULT_ITEMS} of {total} total results. "
                      f"Summarize accordingly and tell the user the total count."
        })
        return json.dumps(truncated)
    return content


def wrap_tool_with_truncation(tool):
    """
    Wraps the tool's coroutine to truncate large JSON-array results in the content
    field (shown to the LLM), while keeping the full results in the artifact field.
    """
    original_coro = tool.coroutine
    if original_coro is None:
        return tool  # nothing to wrap

    @functools.wraps(original_coro)
    async def _truncated(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        # MCP tools use response_format="content_and_artifact" → (content, artifact)
        if isinstance(result, tuple) and len(result) == 2:
            content, _original_artifact = result
            truncated_content = _truncate_json_content(content)
            # artifact = always the FULL content (for export / comparison)
            return (truncated_content, content)
        return result

    tool.coroutine = _truncated
    return tool


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

def build_system_prompt(schema: str) -> str:
    """
    Combine the prompt template with the schema.
    
    Args:
        schema: The database schema loaded from system-prompt file
    
    Returns:
        Complete system prompt with schema inserted.
    """
    template = get_prompt()
    return template.format(schema=schema)


async def initialize_agent(
    model_name: str = "gemini-2.5-flash",
    api_key: Optional[str] = None,
    system_prompt_path: str = "scripts/system-prompt",
    temperature: float = 0.0
) -> Tuple[Optional[object], Optional[str]]:
    """
    Initialize the LLM agent with MCP Neo4j tools.
    
    Args:
        model_name: Gemini model identifier (default: gemini-2.5-flash)
        api_key: Google API key. If None, uses GOOGLE_GENAI_API_KEY env var
        system_prompt_path: Path to the Neo4j schema file
        temperature: LLM temperature (0.0 = deterministic, higher = more creative)
    
    Returns:
        Tuple of (agent, system_prompt) on success, (None, None) on failure
    
    Example:
        >>> agent, prompt = await initialize_agent(api_key="your_key")
        >>> if agent:
        ...     # Use agent for queries
    """
    # Initialize MCP Client for Neo4j
    # Read timeout prevents crashes on very long queries (like 66.1 which takes 27-30 min average)
    # Gemini 2.5 Flash has 1M TPM so large responses are fine
    read_timeout = os.getenv("NEO4J_READ_TIMEOUT", "120")
    
    client = MultiServerMCPClient({
        "neo4j": {
            "command": "uvx",
            "args": [
                "--with", "fastmcp<3.0.0",
                "mcp-neo4j-cypher@0.5.2",
                "--db-url", os.getenv("NEO4J_URI", "bolt://iyp-bolt.ihr.live:7687"),
                "--username", os.getenv("NEO4J_USERNAME", "neo4j"),
                "--password", os.getenv("NEO4J_PASSWORD", ""),
                "--database", os.getenv("NEO4J_DATABASE", "neo4j"),
                "--read-timeout", read_timeout,
            ],
            "transport": "stdio"
        }
    })

    try:
        tools = await client.get_tools()
        # Keep only the read tool to force Cypher usage
        tools = [tool for tool in tools if tool.name == 'read_neo4j_cypher']
        # Truncate large results: LLM sees summary, full data goes to artifact
        tools = [wrap_tool_with_truncation(t) for t in tools]
    except Exception as e:
        print(f"Failed to initialize tools: {e}")
        return None, None

    # Load schema
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            schema = f.read()
    except FileNotFoundError:
        print(f"Warning: {system_prompt_path} not found. Using empty schema.")
        schema = ""

    # Build complete system prompt
    system_prompt = build_system_prompt(schema)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

    agent = create_agent(llm, tools)
    return agent, system_prompt