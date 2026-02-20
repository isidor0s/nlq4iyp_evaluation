#!/usr/bin/env python3
"""Execute canonical or generated Cypher queries directly against Neo4j.

Reads tasks from an evaluation CSV, executes the selected Cypher column
(canonical or generated) against the Neo4j database, and saves per-task
JSON result files compatible with compare_results.py.

Usage examples:
    # Execute canonical solutions
    python scripts/eval/execute_canonical.py --canonical \\
        --csv data/eval_data/cypherEval/variation-B-filtered.csv \\
        --output data/eval_data/results/canonical

    # Execute generated queries from a results CSV
    python scripts/eval/execute_canonical.py --generated \\
        --csv data/eval_data/results/run_v2_2_.csv \\
        --output data/eval_data/results/generated_rerun

    # Run a single task
    python scripts/eval/execute_canonical.py --canonical --task 1.1
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Neo4j object conversion
# ---------------------------------------------------------------------------

def convert_neo4j_value(value):
    """Recursively convert a Neo4j value to a JSON-serializable type."""
    if value is None:
        return None

    # Neo4j Node
    if hasattr(value, 'labels') and hasattr(value, 'items'):
        return {k: convert_neo4j_value(v) for k, v in dict(value).items()}

    # Neo4j Relationship
    if hasattr(value, 'type') and hasattr(value, 'start_node'):
        return {k: convert_neo4j_value(v) for k, v in dict(value).items()}

    # Neo4j Path
    if hasattr(value, 'nodes') and hasattr(value, 'relationships'):
        return {
            '_nodes': [convert_neo4j_value(n) for n in value.nodes],
            '_relationships': [convert_neo4j_value(r) for r in value.relationships],
        }

    # list / tuple
    if isinstance(value, (list, tuple)):
        return [convert_neo4j_value(v) for v in value]

    # dict
    if isinstance(value, dict):
        return {k: convert_neo4j_value(v) for k, v in value.items()}

    # Primitive types that are already JSON-safe
    if isinstance(value, (str, int, float, bool)):
        return value

    # Fallback
    return str(value)


def convert_neo4j_records(records: list[dict]) -> list[dict]:
    """Convert a list of Neo4j record dicts to JSON-serializable dicts."""
    return [
        {k: convert_neo4j_value(v) for k, v in record.items()}
        for record in records
    ]

# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_tasks(csv_path: str, column: str) -> list[dict]:
    """Load tasks from CSV using the specified Cypher column.

    Args:
        csv_path: Path to the evaluation CSV.
        column: Either 'canonical' or 'generated'.

    Returns:
        List of task dicts with keys: id, difficulty, prompt, query.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        sys.exit(1)

    # Map column selection to possible CSV header names
    canonical_headers = ['Canonical Solution', 'canonical_solution']
    generated_headers = ['generated_cypher_query']

    tasks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get('Task ID', row.get('id', ''))
            difficulty = row.get('Difficulty Level', row.get('difficulty', ''))
            prompt = row.get('Prompt', row.get('prompt', ''))

            # Resolve the Cypher query string
            raw_query = ''
            if column == 'canonical':
                for h in canonical_headers:
                    if h in row and row[h]:
                        raw_query = row[h]
                        break
            else:  # generated
                for h in generated_headers:
                    if h in row and row[h]:
                        raw_query = row[h]
                        break

            # generated_cypher_query is stored as a JSON array of strings;
            # pick the last query (the final attempt) if multiple exist.
            query = raw_query
            if column == 'generated' and raw_query.startswith('['):
                try:
                    queries = json.loads(raw_query)
                    query = queries[-1] if queries else raw_query
                except json.JSONDecodeError:
                    query = raw_query

            tasks.append({
                'id': task_id,
                'difficulty': difficulty,
                'prompt': prompt,
                'query': query.strip() if query else '',
            })

    return tasks

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

def execute_query(driver, query: str, timeout_s: int) -> dict:
    """Run a single Cypher query and return results + metadata."""
    start = time.time()
    try:
        with driver.session(
            database=os.getenv('NEO4J_DATABASE', 'neo4j')
        ) as session:
            result = session.run(query, timeout=timeout_s)
            records = [dict(record) for record in result]
            records = convert_neo4j_records(records)
            elapsed_ms = (time.time() - start) * 1000
            return {
                'success': True,
                'results': records,
                'count': len(records),
                'error': None,
                'execution_time_ms': elapsed_ms,
            }
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            'success': False,
            'results': [],
            'count': 0,
            'error': str(e),
            'execution_time_ms': elapsed_ms,
        }

# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_result(output_dir: str, task: dict, exec_result: dict, column: str):
    """Save a single task result as a JSON file."""
    data = {
        'task_id': task['id'],
        'difficulty': task['difficulty'],
        'prompt': task['prompt'],
        'query': task['query'],
        'success': exec_result['success'],
        'results': exec_result['results'],
        'count': exec_result['count'],
        'error': exec_result['error'],
        'execution_time_ms': exec_result['execution_time_ms'],
        'prompt_variant': f'DIRECT_{column.upper()}',
    }
    filename = f"task_{task['id']}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Execute Cypher queries from an evaluation CSV against Neo4j.'
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--canonical', action='store_true',
        help='Execute the canonical solution column from the CSV.'
    )
    mode.add_argument(
        '--generated', action='store_true',
        help='Execute the generated query column from the CSV.'
    )
    parser.add_argument(
        '--csv', default='data/eval_data/cypherEval/variation-B-filtered.csv',
        help='Path to the evaluation CSV (default: variation-B-filtered.csv).'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output directory for JSON results (default: data/eval_data/results/<canonical|generated>).'
    )
    parser.add_argument(
        '--timeout', type=int, default=300,
        help='Per-query timeout in seconds (default: 300).'
    )
    parser.add_argument(
        '--task', default=None,
        help='Run only the task with this ID (e.g. --task 1.1).'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip tasks whose JSON output files already exist.'
    )
    args = parser.parse_args()

    column = 'canonical' if args.canonical else 'generated'

    # Default output directory
    if args.output is None:
        args.output = f'data/eval_data/results/{column}'

    # Load env
    load_dotenv()

    # Load tasks
    tasks = load_tasks(args.csv, column)
    if not tasks:
        print('No tasks loaded from CSV.')
        sys.exit(1)

    # Filter to single task if requested
    if args.task:
        tasks = [t for t in tasks if t['id'] == args.task]
        if not tasks:
            print(f"Task '{args.task}' not found in CSV.")
            sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Skip existing if requested
    if args.skip_existing:
        before = len(tasks)
        tasks = [
            t for t in tasks
            if not os.path.exists(
                os.path.join(args.output, f"task_{t['id']}.json")
            )
        ]
        skipped = before - len(tasks)
        if skipped:
            print(f"Skipping {skipped} tasks with existing results.")

    if not tasks:
        print('All tasks already have results. Nothing to do.')
        return

    # Connect to Neo4j
    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USERNAME', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', '')
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Verify connectivity
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {uri}")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    print(f"Column: {column}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {args.output}")
    print(f"Timeout: {args.timeout}s per query")
    print('-' * 60)

    success_count = 0
    fail_count = 0
    total_start = time.time()

    for i, task in enumerate(tasks, 1):
        task_id = task['id']
        query = task['query']

        if not query:
            print(f"[{i}/{len(tasks)}] Task {task_id}: SKIP (empty query)")
            fail_count += 1
            save_result(args.output, task, {
                'success': False,
                'results': [],
                'count': 0,
                'error': 'Empty query',
                'execution_time_ms': 0,
            }, column)
            continue

        print(f"[{i}/{len(tasks)}] Task {task_id}: ", end='', flush=True)
        result = execute_query(driver, query, args.timeout)
        save_result(args.output, task, result, column)

        if result['success']:
            success_count += 1
            print(f"OK  ({result['count']} records, "
                  f"{result['execution_time_ms']:.0f}ms)")
        else:
            fail_count += 1
            err_short = (result['error'] or '')[:80]
            print(f"FAIL  ({err_short})")

    total_time = time.time() - total_start
    driver.close()

    # Summary
    print('-' * 60)
    print(f"Done in {total_time:.1f}s")
    print(f"  Success: {success_count}/{len(tasks)}")
    print(f"  Failed:  {fail_count}/{len(tasks)}")

    # Save summary
    summary = {
        'column': column,
        'csv': args.csv,
        'total_tasks': len(tasks),
        'success': success_count,
        'failed': fail_count,
        'total_time_s': round(total_time, 2),
        'timeout_s': args.timeout,
    }
    summary_path = os.path.join(args.output, '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
