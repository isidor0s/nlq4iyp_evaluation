"""
Gemini Agent CLI similar to https://nlq4iyp.streamlit.app/

An interactive command line interface for querying the IYP Neo4j knowledge graph
using natural language. The agent translates questions to Cypher queries.

Features:
    - Real-time streaming of agent responses
    - Query capture and display
    - Automatic retry with LIMIT on timeout
    - Session statistics and query export
   
Commands:
    exit     - Quit the application
    /queries - Show all executed queries this session
    /stats   - Show session statistics
    /clear   - Clear conversation history

Usage:
    python gemini_agent.py

Requirements:
    - GOOGLE_GENAI_API_KEY environment variable
    - Neo4j connection details in environment variables (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
"""

import os
import asyncio
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Support running both as standalone script and as package import
try:
    from .agent_core import initialize_agent
    from .agent_utils import (
        extract_text_content,
        is_neo4j_timeout_error,
        classify_query_step,
        safe_json_dumps,
        count_items_in_json,
        TruncationConfig,
        truncate_output
    )
except ImportError:
    # Running as standalone script (python scripts/agent/gemini_agent.py)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from agent_core import initialize_agent
    from agent_utils import (
        extract_text_content,
        is_neo4j_timeout_error,
        classify_query_step,
        safe_json_dumps,
        count_items_in_json,
        TruncationConfig,
        truncate_output
    )

load_dotenv()

# =============================================================================
# ANSI COLORS
# =============================================================================

class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

@dataclass
class DisplayConfig:
    """Configuration for output display."""
    max_lines: int = 20
    max_chars: int = 2000
    show_query_details: bool = True


def display_tool_result(result_data, config: DisplayConfig = None):
    """
    Display tool output with truncation for large results.
    Uses shared utilities from agent_utils.
    """
    if not result_data:
        return
    
    config = config or DisplayConfig()
    c = Colors
    
    json_output = safe_json_dumps(result_data)
    truncation_config = TruncationConfig(max_lines=config.max_lines, max_chars=config.max_chars)
    truncated_text, was_truncated = truncate_output(json_output, truncation_config)
    
    item_count = count_items_in_json(result_data)
    
    if was_truncated:
        item_msg = f" (Total items: {item_count})" if item_count else ""
        print(f"\n{c.CYAN}--- Result Preview ---{c.RESET}")
        print(truncated_text)
        print(f"\n{c.YELLOW}... [Truncated. Showing {config.max_lines} lines{item_msg}] ...{c.RESET}\n")
    else:
        print(f"\n{c.CYAN}--- Database Result ---{c.RESET}")
        print(json_output)
        print()


def display_error_with_hint(error_msg: str, last_query: Optional[str] = None):
    """Display error with helpful hints for common issues."""
    c = Colors
    
    print(f"\n{c.RED}Error: {error_msg}{c.RESET}")
    
    # Provide helpful hints based on error type
    if is_neo4j_timeout_error(error_msg):
        print(f"\n{c.YELLOW}⏱️  Query timed out - too much data or complex query.{c.RESET}")
        if last_query:
            print(f"\n{c.CYAN}Captured query:{c.RESET}")
            print(f"  {last_query}")
            print(f"\n{c.YELLOW}You can run this query directly on:{c.RESET}")
            print(f"  {c.CYAN}https://iyp.iijlab.net{c.RESET} (web interface)")
            print(f"  Or: bolt://iyp-bolt.ihr.live:7687 (Neo4j Browser)")
        print(f"\n{c.YELLOW}💡 Tip: Add 'LIMIT 100' to reduce data size.{c.RESET}")
        print()


# =============================================================================
# STREAMING WITH QUERY CAPTURE
# =============================================================================

@dataclass
class StreamState:
    """Tracks state during agent streaming."""
    captured_queries: list = field(default_factory=list)
    tool_outputs: list = field(default_factory=list)
    final_answer: str = ""
    last_query: Optional[str] = None
    error: Optional[str] = None


async def stream_agent_interactive(
    agent,
    messages: list,
    display_config: DisplayConfig = None
) -> StreamState:
    """
    Stream agent responses with real time display and query capture.
    Returns StreamState with all captured queries.
    """
    state = StreamState()
    config = display_config or DisplayConfig()
    c = Colors
    
    try:
        async for event in agent.astream({"messages": messages}, stream_mode="values"):
            if "messages" not in event:
                continue
            
            last_msg = event["messages"][-1]
            
            #HANDLE TOOL CALLS (Query Execution)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool in last_msg.tool_calls:
                    if tool['name'] == 'read_neo4j_cypher':
                        query = tool['args'].get('cypher') or tool['args'].get('query', '')
                        
                        if query:
                            # Capture for potential export
                            if query not in state.captured_queries:
                                state.captured_queries.append(query)
                            state.last_query = query
                            
                            # Display with step classification
                            if config.show_query_details:
                                step_name = classify_query_step(query)
                                print(f"{c.BLUE}  [{step_name}]{c.RESET}: {query}")
            
            #HANDLE TOOL OUTPUTS (Results/Errors)
            if last_msg.type == "tool":
                content = last_msg.content
                state.tool_outputs.append(content)
                
                # Check for errors in tool response
                if isinstance(content, str) and "error" in content.lower():
                    state.error = content
                    display_error_with_hint(content, state.last_query)
                else:
                    display_tool_result(content, config)
            
            #HANDLE FINAL ANSWER
            if last_msg.type == "ai" and not getattr(last_msg, 'tool_calls', None):
                state.final_answer = extract_text_content(last_msg.content)
    
    except Exception as e:
        state.error = str(e)
        display_error_with_hint(str(e), state.last_query)
    
    return state


def check_for_timeout_error(state: StreamState) -> bool:
    """Check if the stream state contains a timeout error."""
    if state.error and is_neo4j_timeout_error(state.error):
        return True
    for output in state.tool_outputs:
        if isinstance(output, str) and is_neo4j_timeout_error(output):
            return True
    return False


def create_limit_retry_message(failing_query: Optional[str], limit: int = 100) -> str:
    """Create a follow-up message instructing retry with LIMIT."""
    if failing_query:
        return f"""The previous query timed out. Please add LIMIT {limit} and retry:

Original query:
{failing_query}

Retry with: {failing_query} LIMIT {limit}"""
    return f"The query timed out. Please add 'LIMIT {limit}' at the end and retry."


def create_rethink_retry_message(failing_query: Optional[str], original_prompt: str) -> str:
    """Create a follow-up message when a LIMIT retry returned 0 results.
    
    Instead of accepting empty results, this asks the agent to reason about
    WHY the query failed and generate a completely new approach.
    """
    query_context = f"\n\nThe failing query was:\n```cypher\n{failing_query}\n```" if failing_query else ""
    return f"""The previous query returned 0 results. The query logic is likely flawed - adding LIMIT did not help.{query_context}

Please do NOT just re-run the same query. Instead:
1. Re-read the original question: \"{original_prompt}\"
2. Analyze WHY the query returned nothing (wrong relationships? wrong filters? wrong traversal path?)
3. Generate a completely NEW query with a different approach
4. Execute the new query

Remember to check the schema for correct node labels, relationship types and directions."""


def check_for_empty_results(state: StreamState) -> bool:
    """Check if the agent completed but all tool outputs returned empty/no results."""
    if not state.tool_outputs:
        return True
    for output in state.tool_outputs:
        if isinstance(output, str):
            # Check if output has actual data
            stripped = output.strip()
            if stripped and stripped not in ('[]', '"[]"', '') and 'error' not in stripped.lower():
                return False
        elif output:  # non-string, non-empty
            return False
    return True


# =============================================================================
# CHAT SESSION
# =============================================================================

@dataclass 
class ChatSession:
    """Manages a chat session with history and query tracking."""
    agent: any
    system_prompt: str
    history: list = field(default_factory=list)
    all_queries: list = field(default_factory=list)  #across conversation
    answer_times: list = field(default_factory=list)  # Track response times
    display_config: DisplayConfig = field(default_factory=DisplayConfig)
    max_timeout_retries: int = 3  # timeout -> LIMIT -> rethink
    
    def __post_init__(self):
        self.history = [{"role": "system", "content": self.system_prompt}]
    
    async def send_message(self, user_input: str) -> str:
        """Send a message and get response, with automatic timeout retry and rethink."""
        c = Colors
        
        self.history.append({"role": "user", "content": user_input})
        used_limit = False
        
        for attempt in range(self.max_timeout_retries):
            if attempt == 0:
                print(f"\n{c.YELLOW}Agent is thinking...{c.RESET}")
            elif not used_limit:
                print(f"\n{c.YELLOW}⏱️ Retrying with LIMIT (attempt {attempt + 1})...{c.RESET}")
            else:
                print(f"\n{c.YELLOW}🔄 Agent is rethinking with a new approach (attempt {attempt + 1})...{c.RESET}")
            
            # Stream with query capture
            state = await stream_agent_interactive(
                self.agent, 
                self.history,
                self.display_config
            )
            
            # Track queries from this turn
            for q in state.captured_queries:
                if q not in self.all_queries:
                    self.all_queries.append(q)
            
            # Check for timeout - if so, retry with LIMIT or rethink
            if check_for_timeout_error(state) and attempt < self.max_timeout_retries - 1:
                # If we already tried LIMIT and it still timed out, ask agent to rethink
                if used_limit:
                    print(f"\n{c.YELLOW}⚠️ LIMIT retry also timed out. Asking agent to rethink...{c.RESET}")
                    self.history.append({"role": "assistant", "content": "The query timed out even with LIMIT."})
                    rethink_msg = create_rethink_retry_message(state.last_query, user_input)
                    self.history.append({"role": "user", "content": rethink_msg})
                    continue  # Rethink
                
                # First timeout: try adding LIMIT
                print(f"\n{c.YELLOW}⏱️ Query timed out. Automatically retrying with LIMIT...{c.RESET}")
                self.history.append({"role": "assistant", "content": "The query timed out due to too much data."})
                retry_msg = create_limit_retry_message(state.last_query)
                self.history.append({"role": "user", "content": retry_msg})
                used_limit = True
                continue  # Retry
            
            # Check for empty results after LIMIT retry - ask agent to rethink
            if used_limit and check_for_empty_results(state) and attempt < self.max_timeout_retries - 1:
                print(f"\n{c.YELLOW}⚠️ LIMIT retry returned 0 results. Asking agent to rethink...{c.RESET}")
                self.history.append({"role": "assistant", "content": state.final_answer or "The query returned no results."})
                rethink_msg = create_rethink_retry_message(state.last_query, user_input)
                self.history.append({"role": "user", "content": rethink_msg})
                continue  # Rethink
            
            # Success or final attempt - break out
            break
        
        # Add final answer to history
        if state.final_answer:
            self.history.append({"role": "assistant", "content": state.final_answer})
        
        return state.final_answer
    
    def export_queries(self) -> list:
        """Export all queries from the session (for debugging/analysis)."""
        return self.all_queries.copy()
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "total_messages": len(self.history),
            "total_queries": len(self.all_queries),
            "user_messages": sum(1 for m in self.history if m.get("role") == "user")
        }


# =============================================================================
# MAIN CLI
# =============================================================================

def print_banner():
    """Print welcome banner."""
    c = Colors
    print(f"{c.BOLD}{c.GREEN}" + "="*60)
    print("NEO4J IYP GRAPH AGENT v3")
    print("="*60 + f"{c.RESET}")
    print(f"Commands:")
    print(f"  {c.CYAN}exit{c.RESET}     - Quit the agent")
    print(f"  {c.CYAN}/queries{c.RESET} - Show all executed queries this session")
    print(f"  {c.CYAN}/stats{c.RESET}   - Show session statistics")
    print(f"  {c.CYAN}/clear{c.RESET}   - Clear conversation history")
    print()


async def run_cli():
    """Main CLI entry point."""
    c = Colors
    
    # Check API key
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print(f"{c.RED}Error: GOOGLE_GENAI_API_KEY not found in environment.{c.RESET}")
        return
    
    # Initialize agent
    print("Initializing Agent...")
    agent, system_prompt = await initialize_agent(api_key=api_key)
    
    if not agent:
        print(f"{c.RED}Failed to initialize agent. Check Neo4j connection.{c.RESET}")
        return
    
    # Create session
    session = ChatSession(agent=agent, system_prompt=system_prompt)
    
    print_banner()
    
    while True:
        try:
            user_input = input(f"{c.BOLD}User:{c.RESET} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            elif user_input == "/queries":
                queries = session.export_queries()
                if queries:
                    print(f"\n{c.CYAN}Executed queries this session ({len(queries)}):{c.RESET}")
                    for i, q in enumerate(queries, 1):
                        print(f"  {i}. {q[:100]}{'...' if len(q) > 100 else ''}")
                else:
                    print(f"{c.YELLOW}No queries executed yet.{c.RESET}")
                print()
                continue
            
            elif user_input == "/stats":
                stats = session.get_stats()
                print(f"\n{c.CYAN}Session Statistics:{c.RESET}")
                print(f"  Messages: {stats['user_messages']} user, {stats['total_messages']} total")
                print(f"  Queries executed: {stats['total_queries']}")
                if session.answer_times:
                    avg_time = sum(session.answer_times) / len(session.answer_times)
                    total_time = sum(session.answer_times)
                    print(f"  Average answer time: {avg_time:.1f}s")
                    print(f"  Total query time: {total_time:.1f}s")
                print()
                continue
            
            elif user_input == "/clear":
                session.history = [{"role": "system", "content": session.system_prompt}]
                session.all_queries = []
                print(f"{c.GREEN}Conversation cleared.{c.RESET}\n")
                continue
            
            # Send message
            start_time = time.time()
            response = await session.send_message(user_input)
            elapsed = time.time() - start_time
            session.answer_times.append(elapsed)
            
            # Display response
            print(f"\n{c.GREEN}{c.BOLD}Agent:{c.RESET} {response}")
            print(f"\n{c.CYAN}Answered in {elapsed:.1f}s{c.RESET}\n")
            print(f"{c.CYAN}" + "-"*60 + f"{c.RESET}")
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        except Exception as e:
            # Catch-all for unexpected errors
            print(f"\n{c.RED}Unexpected error: {e}{c.RESET}")
            print(f"{c.YELLOW}You can continue chatting or type 'exit' to quit.{c.RESET}\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_cli())
