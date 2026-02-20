"""
Batch Evaluation Script

Processes text-to-Cypher tasks with streaming agent execution, capturing generated queries 
and results in real-time using a FIFO queue.

Features:
- Streaming execution with real-time query/result capture via FIFO matching
- retry: Auto-adds LIMIT on Neo4j timeouts; prompts agent to rethink if retry fails
- Detailed recording: Queries, results, answers, execution time, errors (CSV + JSON output)
- Configurable: Timeouts, retries, rate limits, output paths via EvalConfig
"""

import asyncio
import csv
import os
import argparse
import time
import random
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Add eval dir to path for loocv import (avoid 'eval' package name conflict with builtin)
from pathlib import Path
_eval_dir = Path(__file__).parent
_scripts_dir = _eval_dir.parent
sys.path.insert(0, str(_scripts_dir))
sys.path.insert(0, str(_eval_dir))

from agent.agent_core import initialize_agent
from agent.agent_utils import (
    TaskStatus,
    extract_text_content,
    classify_error
)
from agent.prompts import get_active_prompt_name
from loocv import build_prompt_excluding_task

load_dotenv()

# Handle large CSV fields
csv.field_size_limit(10 * 1024 * 1024)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for batch evaluation."""
    # Timeouts
    agent_timeout_seconds: int = 300
    neo4j_timeout_seconds: int = 60
    
    # Retry settings
    max_retries: int = 4
    recursion_limit: int = 50
    
    # Rate limiting
    rate_limit_base_wait: int = 45
    rate_limit_max_wait: int = 600
    
    # Query modification
    default_limit: int = 100
    
    # Batch processing
    batch_size: int = 5
    concurrency: int = 1
    
    # Output
    results_folder: str = "query_results/generated"


# =============================================================================
# QUERY RESULT CAPTURE
# =============================================================================

@dataclass
class QueryExecution:
    """A single query execution with its result."""
    query: str
    success: bool
    results: list = field(default_factory=list)
    error: Optional[str] = None
    execution_time_ms: float = 0


def parse_tool_result(content: Any) -> tuple[bool, list, Optional[str]]:
    """
    Parse a Neo4j tool result to extract success, results, and error.
    
    Tool results can be:
    - List of records (success)
    - String with error message (failure)
    - Dict with error info (failure)
    """
    if content is None:
        return False, [], "No result"
    
    # String result - could be error or empty result message
    if isinstance(content, str):
        # Try parsing as JSON FIRST (MCP often returns results as JSON string)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return True, parsed, None
            if isinstance(parsed, dict):
                # Check if this is a result object with "type", "text", etc.
                if "type" in parsed and "text" in parsed:
                    # This is a LangChain message object - parse the text field
                    try:
                        inner = json.loads(parsed["text"])
                        if isinstance(inner, list):
                            return True, inner, None
                        if isinstance(inner, dict):
                            return True, [inner], None
                    except Exception:
                        pass
                    return True, [], None
                # Regular error dict
                if "error" in parsed:
                    return False, [], str(parsed.get("error", "Unknown error"))
                return True, [parsed], None
        except json.JSONDecodeError:
            pass
        
        # Only check for error keywords if JSON parsing failed
        content_lower = content.lower()
        if "error" in content_lower or "timeout" in content_lower:
            return False, [], content[:500]
        if "no results" in content_lower or "empty" in content_lower:
            return True, [], None
        return True, [], None
    
    # List result - success
    if isinstance(content, list):
        return True, content, None
    
    # Dict result - could be error or single record
    if isinstance(content, dict):
        # Check if this is a result object with "type", "text", etc.
        if "type" in content and "text" in content:
            # This is a LangChain message object parse the text field
            try:
                parsed = json.loads(content["text"])
                if isinstance(parsed, list):
                    return True, parsed, None
                if isinstance(parsed, dict):
                    return True, [parsed], None
            except:
                pass
            return True, [], None
        # Check for error
        if "error" in content:
            return False, [], str(content.get("error", "Unknown error"))
        return True, [content], None
    
    return True, [], None


def convert_neo4j_objects(results: list) -> list:
    """Convert Neo4j node/relationship objects to plain dicts."""
    converted = []
    for record in results:
        if isinstance(record, dict):
            new_record = {}
            for key, value in record.items():
                if hasattr(value, '__dict__'):
                    try:
                        new_record[key] = dict(value)
                    except Exception:
                        new_record[key] = str(value)
                else:
                    new_record[key] = value
            converted.append(new_record)
        else:
            converted.append(record)
    return converted


# =============================================================================
# STREAMING WITH RESULT CAPTURE
# =============================================================================

@dataclass
class StreamResult:
    """Result from streaming the agent with query results captured."""
    queries: list = field(default_factory=list)          # Query strings
    executions: list = field(default_factory=list)       # Query Execution objects
    answer: str = ""
    error: Optional[Exception] = None
    error_status: Optional[TaskStatus] = None
    completed: bool = False


async def stream_agent_with_capture(agent, messages: list, config: EvalConfig) -> StreamResult:
    """
    Stream the agent and capture BOTH queries AND their results.
    
    KEY FEATURE: capture the tool output (Neo4j results) as they stream, 
    in order to have the exact data the agent saw for each query.
    
    Uses a FIFO queue for pending queries and message ID tracking to handle
    cases where the agent runs multiple queries (streaming can repeat messages).
    """
    result = StreamResult()
    pending_queries = []  # Queue of (query, start_time) - FIFO for matching results
    seen_msg_ids = set()  # Prevent duplicate processing from streaming
    
    try:
        async for event in agent.astream(
            {"messages": messages},
            config={"recursion_limit": config.recursion_limit},
            stream_mode="values"
        ):
            if "messages" not in event:
                continue
            
            for msg in event["messages"]:
                # Skip already-processed messages (streaming repeats them)
                msg_id = getattr(msg, 'id', None) or id(msg)
                if msg_id in seen_msg_ids:
                    continue
                seen_msg_ids.add(msg_id)
                
                # CAPTURE TOOL CALLS (queries) - This happens BEFORE execution
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool in msg.tool_calls:
                        if tool['name'] == 'read_neo4j_cypher':
                            query = tool['args'].get('cypher') or tool['args'].get('query', '')
                            if query:
                                pending_queries.append((query, time.time()))
                                if query not in result.queries:
                                    result.queries.append(query)
                
                # CAPTURE TOOL OUTPUTS (query results)
                if msg.type == "tool":
                    # Full results live in artifact (content may be truncated for the LLM)
                    content = getattr(msg, 'artifact', None) or msg.content
                    
                    # Parse the tool result
                    success, records, error = parse_tool_result(content)
                    records = convert_neo4j_objects(records)
                    
                    # Match result with oldest pending query (FIFO)
                    if pending_queries:
                        query, start_time = pending_queries.pop(0)
                        elapsed = (time.time() - start_time) * 1000
                        execution = QueryExecution(
                            query=query,
                            success=success,
                            results=records,
                            error=error,
                            execution_time_ms=elapsed
                        )
                        result.executions.append(execution)
                    
                    # Check for error that might indicate Neo4j timeout
                    if isinstance(content, str) and "timeout" in content.lower():
                        # Don't raise - we've captured what we can
                        pass
                
                # CAPTURE FINAL ANSWER
                if msg.type == "ai" and not getattr(msg, 'tool_calls', None):
                    result.answer = extract_text_content(msg.content)
        
        result.completed = True
        
    except asyncio.TimeoutError:
        result.error = asyncio.TimeoutError("Agent timeout")
        result.error_status = TaskStatus.TIMEOUT
        
    except Exception as e:
        result.error = e
        status, _ = classify_error(str(e))
        result.error_status = status
    
    return result


# =============================================================================
# TASK PROCESSING
# =============================================================================

@dataclass
class TaskResult:
    """Result from processing a task, including query executions."""
    task_id: str
    difficulty: str
    prompt: str
    canonical_solution: str
    generated_queries: list
    executions: list  # List of QueryExecution
    answer: str
    execution_time: float
    status: TaskStatus
    error_detail: str = ""
    used_limit_fallback: bool = False
    prompt_variant: str = ""  # Track which prompt was used (e.g., "INVESTIGATOR_V2", "PYTHIA_PROMPT")


def create_limit_retry_message(failing_query: Optional[str], limit: int) -> dict:
    """Create a follow-up message for LIMIT retry."""
    if failing_query:
        return {
            "role": "user",
            "content": f"""The previous query timed out. Please add LIMIT {limit} and retry:

```cypher
{failing_query}
LIMIT {limit}
```

Execute the modified query now."""
        }
    return {
        "role": "user",
        "content": f"""The query timed out. Please add 'LIMIT {limit}' at the end and retry."""
    }


def create_rethink_retry_message(failing_query: Optional[str], original_prompt: str) -> dict:
    """Create a follow-up message when a LIMIT retry returned 0 results.
    
    Instead of accepting empty results, this asks the agent to reason about
    WHY the query failed and generate a completely new approach.
    """
    query_context = f"\n\nThe failing query was:\n```cypher\n{failing_query}\n```" if failing_query else ""
    return {
        "role": "user",
        "content": f"""The previous query returned 0 results. The query logic is likely flawed, adding LIMIT did not help.{query_context}

Please do NOT just re-run the same query. Instead:
1. Re-read the original question: \"{original_prompt}\"
2. Analyze WHY the query returned nothing (wrong relationships? wrong filters? wrong traversal path?)
3. Generate a completely NEW query with a different approach
4. Execute the new query

Remember to check the schema for correct node labels, relationship types and directions."""
    }


async def process_task(
    task: dict,
    agent,
    system_prompt: str,
    task_num: int,
    total_tasks: int,
    config: EvalConfig,
    prompt_variant: str = "UNKNOWN"
) -> TaskResult:
    """Process a single task with query result capture."""
    task_id = task["id"]
    difficulty = task["difficulty"]
    prompt = task["prompt"]
    canonical = task["canonical_solution"]
    
    start_time = time.time()
    print(f"\n[{task_num}/{total_tasks}] Task {task_id} ({difficulty})")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Track across retry attempts
    all_queries = []
    all_executions = []
    final_answer = ""
    final_status = TaskStatus.FAILED
    error_detail = ""
    used_limit = False
    rate_limit_wait = 0
    
    for attempt in range(config.max_retries):
        attempt_start = time.time()
        
        try:
            stream_result = await asyncio.wait_for(
                stream_agent_with_capture(agent, messages, config),
                timeout=config.agent_timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"  [!] Agent timeout on attempt {attempt + 1}/{config.max_retries}")
            final_status = TaskStatus.TIMEOUT
            error_detail = f"Agent timeout after {time.time() - attempt_start:.1f}s"
            continue
        
        # Accumulate queries and executions
        for q in stream_result.queries:
            if q not in all_queries:
                all_queries.append(q)
        all_executions.extend(stream_result.executions)
        
        # Success
        if stream_result.completed and not stream_result.error:
            final_answer = stream_result.answer
            final_status = TaskStatus.SUCCESS
            
            elapsed = time.time() - attempt_start
            print(f"  [OK] Success! {len(stream_result.queries)} queries in {elapsed:.1f}s")
            
            # Check for hallucination
            if not stream_result.queries:
                phrases = ["i don't have", "cannot access", "unable to query"]
                if any(p in final_answer.lower() for p in phrases):
                    final_status = TaskStatus.HALLUCINATION
                    error_detail = "Agent answered without executing queries"
                    print(f"  [!] Hallucination detected")
            
            # Check for empty results after a LIMIT retry - agent should rethink
            has_any_results = any(e.success and e.results for e in stream_result.executions)
            if used_limit and not has_any_results and attempt < config.max_retries - 1:
                failing = stream_result.queries[-1] if stream_result.queries else None
                messages.append(create_rethink_retry_message(failing, prompt))
                final_status = TaskStatus.FAILED  # Reset - not actually successful
                print(f"  [!] LIMIT retry returned 0 results. Asking agent to rethink...")
                continue
            
            break
        
        # Handle errors
        if stream_result.error:
            error_msg = str(stream_result.error)
            error_status = stream_result.error_status or TaskStatus.ERROR
            
            print(f"  [!] {error_status.value} on attempt {attempt + 1}/{config.max_retries}")
            
            # NEO4J TIMEOUT - Retry with LIMIT or rethink
            if error_status == TaskStatus.NEO4J_TIMEOUT:
                if attempt < config.max_retries - 1:
                    failing = stream_result.queries[-1] if stream_result.queries else None
                    
                    # If we already tried LIMIT and it still timed out, ask agent to rethink
                    if used_limit:
                        print(f"  [!] LIMIT retry also timed out. Asking agent to rethink...")
                        messages.append(create_rethink_retry_message(failing, prompt))
                        continue
                    
                    # First timeout: try adding LIMIT
                    messages.append(create_limit_retry_message(failing, config.default_limit))
                    used_limit = True
                    print(f"  [>] Retrying with LIMIT {config.default_limit}...")
                    continue
                else:
                    final_status = TaskStatus.NEO4J_TIMEOUT
                    error_detail = "Neo4j timeout even with LIMIT"
                    break
            
            # RATE LIMIT - Wait and retry
            elif error_status == TaskStatus.RATE_LIMIT:
                wait = min(
                    config.rate_limit_base_wait * (2 ** attempt) + random.randint(5, 15),
                    config.rate_limit_max_wait
                )
                rate_limit_wait += wait
                
                if rate_limit_wait > config.rate_limit_max_wait:
                    print(f"  [X] Exceeded max rate limit wait")
                    final_status = TaskStatus.RATE_LIMIT
                    error_detail = f"Rate limited after {rate_limit_wait}s"
                    break
                
                print(f"  [~] Rate limited. Waiting {wait}s...")
                await asyncio.sleep(wait)
                continue
            
            # Other errors - don't retry
            else:
                final_status = error_status
                _, error_detail = classify_error(error_msg)
                break
    
    # Build result
    execution_time = time.time() - start_time
    
    if final_status == TaskStatus.SUCCESS and used_limit:
        final_answer = f"[LIMIT {config.default_limit}] {final_answer}"
    elif final_status != TaskStatus.SUCCESS:
        if all_queries:
            final_answer = f"[{final_status.value.upper()}] Partial. {len(all_queries)} queries. {error_detail}"
        else:
            final_answer = f"[{final_status.value.upper()}] Failed. {error_detail}"
    
    if all_queries:
        print(f"  [#] Final: {len(all_queries)} queries, {len(all_executions)} results captured")
    
    return TaskResult(
        task_id=task_id,
        difficulty=difficulty,
        prompt=prompt,
        canonical_solution=canonical,
        generated_queries=all_queries,
        executions=all_executions,
        answer=final_answer.replace('\n', ' ').strip(),
        execution_time=execution_time,
        status=final_status,
        error_detail=error_detail,
        used_limit_fallback=used_limit,
        prompt_variant=prompt_variant
    )


# =============================================================================
# FILE I/O
# =============================================================================

def load_tasks(csv_path: str) -> list[dict]:
    """Load tasks from CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)
    
    tasks = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append({
                "id": row.get("Task ID", row.get("id", "")),
                "difficulty": row.get("Difficulty Level", row.get("difficulty", "")),
                "prompt": row.get("Prompt", row.get("prompt", "")),
                "canonical_solution": row.get("Canonical Solution", row.get("canonical_solution", ""))
            })
    return tasks


def save_csv_results(results: list[TaskResult], output_path: str):
    """Save results to CSV (append mode)."""
    if not results:
        return
    
    fieldnames = [
        "id", "difficulty", "prompt", "canonical_solution",
        "generated_cypher_query", "agent_answer",
        "execution_time_s", "status", "prompt_variant"
    ]
    
    file_exists = os.path.isfile(output_path)
    mode = 'a' if file_exists else 'w'
    
    with open(output_path, mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for result in results:
            writer.writerow({
                "id": result.task_id,
                "difficulty": result.difficulty,
                "prompt": result.prompt,
                "canonical_solution": result.canonical_solution,
                "generated_cypher_query": json.dumps(result.generated_queries),
                "agent_answer": result.answer,
                "execution_time_s": f"{result.execution_time:.1f}",
                "status": result.status.value,
                "prompt_variant": result.prompt_variant
            })


def save_json_results(results: list[TaskResult], output_folder: str):
    """
    Save query results as JSON files for comparison.
    
    Format matches compare_results.py expectations:
    {
        "task_id": "1.1",
        "prompt": "...",
        "difficulty": "...",
        "query": "MATCH...",
        "success": true,
        "results": [...],
        "count": 42,
        "error": null
    }
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for result in results:
        # Priority: 1) Successful execution WITH results, 2) Last successful, 3) Last execution
        final_exec = None
        
        # First, look for successful execution with actual results
        for exec in reversed(result.executions):
            if exec.success and exec.results:
                final_exec = exec
                break
        
        # If none with results, fall back to any successful execution
        if not final_exec:
            for exec in reversed(result.executions):
                if exec.success:
                    final_exec = exec
                    break
        
        # Last resort: use last execution
        if not final_exec and result.executions:
            final_exec = result.executions[-1]
        
        # Build JSON data
        if final_exec:
            data = {
                "task_id": result.task_id,
                "difficulty": result.difficulty,
                "prompt": result.prompt,
                "query": final_exec.query,
                "success": final_exec.success,
                "results": final_exec.results,
                "count": len(final_exec.results),
                "error": final_exec.error,
                "execution_time_ms": final_exec.execution_time_ms,
                "prompt_variant": result.prompt_variant
            }
        else:
            # No executions - use first query if available
            data = {
                "task_id": result.task_id,
                "difficulty": result.difficulty,
                "prompt": result.prompt,
                "query": result.generated_queries[0] if result.generated_queries else "",
                "success": False,
                "results": [],
                "count": 0,
                "error": result.error_detail or "No execution",
                "execution_time_ms": 0,
                "prompt_variant": result.prompt_variant
            }
        
        # Save file
        safe_id = result.task_id.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(output_folder, f"task_{safe_id}.json")
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    return len(results)


def save_summary(results: list[TaskResult], output_folder: str, temperature: float = 0.0):
    """Save execution summary."""
    successful = sum(1 for r in results if r.status == TaskStatus.SUCCESS)
    with_results = sum(1 for r in results if r.executions and any(e.success for e in r.executions))
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "temperature": temperature,
        "total_tasks": len(results),
        "successful_tasks": successful,
        "tasks_with_results": with_results,
        "by_status": {},
        "by_difficulty": {}
    }
    
    for r in results:
        status = r.status.value
        summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        
        diff = r.difficulty.split()[0] if r.difficulty else "Unknown"
        summary["by_difficulty"][diff] = summary["by_difficulty"].get(diff, 0) + 1
    
    filepath = os.path.join(output_folder, "_summary.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def print_statistics(csv_path: str, json_folder: str, total_time: float):
    """Print final statistics."""
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        print("No results.")
        return
    
    # Count by status
    status_counts = {}
    queries_captured = 0
    for r in results:
        status = r.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
        
        queries = json.loads(r.get('generated_cypher_query', '[]'))
        if queries:
            queries_captured += 1
    
    # Count JSON files
    json_count = len([f for f in os.listdir(json_folder) if f.startswith("task_")])
    
    total = len(results)
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total tasks: {total}")
    print(f"Queries captured: {queries_captured}/{total} ({queries_captured/total*100:.1f}%)")
    print(f"JSON results saved: {json_count}")
    print(f"\nStatus breakdown:")
    
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        mark = "[OK]" if status == "success" else "[!]" if "timeout" in status else "[X]"
        print(f"  {mark} {status:20} {count:3} ({pct:5.1f}%)")
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON results: {json_folder}/")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Batch Text-to-Cypher Evaluation - With Result Capture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--csv", required=True, help="Input CSV with tasks")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV")
    parser.add_argument("--results-folder", default="query_results/generated", help="JSON results folder")
    parser.add_argument("--system-prompt", default="main_scripts/system-prompt", help="System prompt file")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel tasks")
    parser.add_argument("--timeout", type=int, default=300, help="Agent timeout (seconds)")
    parser.add_argument("--neo4j-timeout", type=int, default=60, help="Neo4j query timeout")
    parser.add_argument("--batch-size", type=int, default=5, help="Tasks per batch")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature (default: 0.3)")
    parser.add_argument("--retry-timeouts", action="store_true", 
                        help="Auto-retry timed-out tasks with extended timeout after batch completes")
    parser.add_argument("--retry-timeout-seconds", type=int, default=600,
                        help="Extended timeout for retry phase (default: 10 min)")
    parser.add_argument("--loocv", action="store_true",
                        help="Enable leave-one-out cross-validation (excludes matching examples from prompt)")
    args = parser.parse_args()
    
    config = EvalConfig(
        agent_timeout_seconds=args.timeout,
        neo4j_timeout_seconds=args.neo4j_timeout,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        results_folder=args.results_folder
    )
    
    # Clear previous outputs
    if os.path.exists(args.output):
        os.remove(args.output)
    if os.path.exists(args.results_folder):
        for f in os.listdir(args.results_folder):
            os.remove(os.path.join(args.results_folder, f))
    
    # Check API key
    api_key = os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        print("Error: GOOGLE_GENAI_API_KEY not found")
        return
    
    # Banner
    print("=" * 80)
    print("BATCH EVALUATOR - With Query Result Capture")
    print("=" * 80)
    print()
    print("Key feature: Captures query RESULTS during agent execution")
    print("  -> No need to re-run queries with query_executor.py")
    print("  -> JSON results saved directly for comparison")
    print()
    print("Configuration:")
    print(f"  Model temperature: {args.temperature}")
    print(f"  Agent timeout: {config.agent_timeout_seconds}s")
    print(f"  Neo4j timeout: {config.neo4j_timeout_seconds}s")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Results folder: {config.results_folder}")
    print(f"  LOOCV enabled: {args.loocv}")
    print()
    
    # Initialize agent
    agent, system_prompt = await initialize_agent(
        model_name="gemini-2.5-flash",
        api_key=api_key,
        system_prompt_path=args.system_prompt,
        temperature=args.temperature
    )
    
    if not agent:
        print("Failed to initialize agent.")
        return
    
    # Load tasks
    tasks = load_tasks(args.csv)
    print(f"Loaded {len(tasks)} tasks from {args.csv}\n")
    
    os.environ['NEO4J_QUERY_TIMEOUT'] = str(config.neo4j_timeout_seconds)
    
    # Process
    semaphore = asyncio.Semaphore(config.concurrency)
    start_time = time.time()
    all_results = []
    
    async def wrapper(task: dict, idx: int) -> TaskResult:
        """Wrapper with enhanced error handling to prevent data loss."""
        async with semaphore:
            try:
                # LOOCV: Build task-specific prompt excluding matching examples
                if args.loocv:
                    task_prompt = build_prompt_excluding_task(system_prompt, task['id'])
                else:
                    task_prompt = system_prompt
                
                prompt_name = get_active_prompt_name()
                return await process_task(task, agent, task_prompt, idx, len(tasks), config, prompt_variant=prompt_name)
            except Exception as e:
                error_msg = str(e)
                # Classify error for status
                status, detail = classify_error(error_msg)
                
                # Check for rate limit - may need global pause
                if 'rate' in error_msg.lower() or '429' in error_msg or 'quota' in error_msg.lower():
                    print(f"  [RATE LIMIT] Task {task['id']} hit rate limit")
                    status = TaskStatus.RATE_LIMIT
                
                # Return a partial result instead of crashing
                return TaskResult(
                    task_id=task.get('id', 'unknown'),
                    difficulty=task.get('difficulty', ''),
                    prompt=task.get('prompt', ''),
                    canonical_solution=task.get('canonical_solution', ''),
                    generated_queries=[],
                    executions=[],
                    answer=f"[CRASH] {detail}",
                    execution_time=0,
                    status=status,
                    error_detail=error_msg[:500],
                    used_limit_fallback=False,
                    prompt_variant=get_active_prompt_name()
                )
    
    total_batches = (len(tasks) + config.batch_size - 1) // config.batch_size
    
    for batch_idx in range(0, len(tasks), config.batch_size):
        batch = tasks[batch_idx : batch_idx + config.batch_size]
        batch_num = batch_idx // config.batch_size + 1
        
        print(f"\n{'='*80}")
        print(f"Batch {batch_num}/{total_batches}")
        print(f"{'='*80}")
        
        coros = [wrapper(task, batch_idx + i + 1) for i, task in enumerate(batch)]
        
        # CRITICAL: return_exceptions=True prevents entire batch loss on one failure
        batch_results_raw = await asyncio.gather(*coros, return_exceptions=True)
        
        # Filter valid results and handle any remaining exceptions
        batch_results = []
        for i, result in enumerate(batch_results_raw):
            if isinstance(result, Exception):
                # This shouldn't happen due to wrapper, but handle it anyway
                task = batch[i]
                print(f"  [CRITICAL] Unhandled exception for {task.get('id', '?')}: {result}")
                batch_results.append(TaskResult(
                    task_id=task.get('id', 'unknown'),
                    difficulty=task.get('difficulty', ''),
                    prompt=task.get('prompt', ''),
                    canonical_solution=task.get('canonical_solution', ''),
                    generated_queries=[],
                    executions=[],
                    answer=f"[EXCEPTION] {str(result)[:200]}",
                    execution_time=0,
                    status=TaskStatus.ERROR,
                    error_detail=str(result)[:500],
                    used_limit_fallback=False
                ))
            else:
                batch_results.append(result)
        
        # Save CSV and JSON IMMEDIATELY after each batch (protects against crashes)
        save_csv_results(batch_results, args.output)
        saved_count = save_json_results(batch_results, config.results_folder)
        all_results.extend(batch_results)
        
        # Check for rate limit hits - pause if needed
        rate_limited = sum(1 for r in batch_results if r.status == TaskStatus.RATE_LIMIT)
        if rate_limited > 0:
            wait_time = min(60 * rate_limited, 300)  # Up to 5 min pause
            print(f"\n  [!] {rate_limited} rate limits hit, pausing {wait_time}s...")
            await asyncio.sleep(wait_time)
        
        successes = sum(1 for r in batch_results if r.status == TaskStatus.SUCCESS)
        with_results = sum(1 for r in batch_results if r.executions)
        print(f"\nBatch {batch_num}: [OK] {successes}/{len(batch)} success, {with_results}/{len(batch)} have results")
        print(f"  Saved {saved_count} JSON files to {config.results_folder}/")
    
    # Summary
    save_summary(all_results, config.results_folder, args.temperature)
    total_time = time.time() - start_time
    print_statistics(args.output, config.results_folder, total_time)
    
    # Auto-retry timed-out tasks if enabled
    if args.retry_timeouts:
        await retry_timeout_tasks(all_results, config, args.retry_timeout_seconds)


async def retry_timeout_tasks(results: list, config: EvalConfig, retry_timeout: int):
    """
    Retry timed-out tasks using direct Neo4j connection with longer timeout.
    
    This runs the captured queries directly against Neo4j (bypassing the agent)
    with an extended timeout, then updates the JSON result files.
    """
    # Find tasks that timed out but have captured queries
    timeout_statuses = {TaskStatus.TIMEOUT, TaskStatus.NEO4J_TIMEOUT}
    timed_out = [r for r in results if r.status in timeout_statuses and r.generated_queries]
    
    if not timed_out:
        print("\n✓ No timed-out tasks to retry.")
        return
    
    print(f"\n{'='*80}")
    print(f"RETRYING {len(timed_out)} TIMED-OUT TASKS")
    print(f"{'='*80}")
    print(f"Using extended timeout: {retry_timeout}s")
    print()
    
    # Import Neo4j driver for direct execution
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("Error: neo4j package not installed. Run: pip install neo4j")
        print("Alternatively, run query_executor_v2.py manually on timed-out queries.")
        return
    
    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://iyp-bolt.ihr.live:7687")
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"✓ Connected to Neo4j at {uri}")
    except Exception as e:
        print(f"✗ Failed to connect to Neo4j: {e}")
        print("Run query_executor_v2.py manually on timed-out queries.")
        return
    
    retried = 0
    succeeded = 0
    
    for result in timed_out:
        task_id = result.task_id
        query = result.generated_queries[-1]  # Use last (most refined) query
        
        print(f"\n[{retried+1}/{len(timed_out)}] Retrying task {task_id}")
        print(f"  Query: {query[:80]}...")
        
        try:
            with driver.session() as session:
                neo4j_result = session.run(query, timeout=retry_timeout)
                records = []
                for record in neo4j_result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, '__dict__'):
                            try:
                                record_dict[key] = dict(value)
                            except:
                                record_dict[key] = str(value)
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                print(f"  ✓ Success! {len(records)} results")
                
                # Update JSON file
                safe_id = task_id.replace("/", "_").replace("\\", "_")
                filepath = os.path.join(config.results_folder, f"task_{safe_id}.json")
                
                data = {
                    "task_id": task_id,
                    "difficulty": result.difficulty,
                    "prompt": result.prompt,
                    "query": query,
                    "success": True,
                    "results": records,
                    "count": len(records),
                    "error": None,
                    "execution_time_ms": 0,  # Not tracked in retry
                    "retried": True
                }
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)
                
                succeeded += 1
                
        except Exception as e:
            error_msg = str(e)[:200]
            print(f"  ✗ Still failed: {error_msg}")
        
        retried += 1
    
    driver.close()
    
    print(f"\n{'='*80}")
    print(f"RETRY COMPLETE: {succeeded}/{len(timed_out)} recovered")
    print(f"{'='*80}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())

