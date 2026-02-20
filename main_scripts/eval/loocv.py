"""
Leave-One-Out Cross-Validation (LOOCV) for Text-to-Cypher Evaluation

This module provides functions to build task-specific system prompts by excluding
examples that match the evaluation task. This prevents data leakage during evaluation.

Usage:
    from loocv import build_prompt_excluding_task
    
    modified_prompt = build_prompt_excluding_task(system_prompt, "42.1")
    # Returns prompt with Examples 83 and 84 removed (matching task 42)
"""

import re
from typing import Optional, Tuple


def get_example_pair_for_task(task_id: str) -> Tuple[int, int]:
    """
    Get the example numbers that correspond to a task ID.
    
    Task naming convention:
        - Task N.1 and N.2 both use Examples (N*2-1) and (N*2)
        - Example: Task 1.1 and 1.2 -> Examples 1, 2
        - Example: Task 42.1 and 42.2 -> Examples 83, 84
    
    Args:
        task_id: Task identifier like "1.1", "1.2", "42.1", etc.
    
    Returns:
        Tuple of (example_1, example_2) numbers to exclude
    """
    # Extract the task number (before the decimal)
    task_num = int(task_id.split('.')[0])
    
    # Calculate example numbers
    example_1 = task_num * 2 - 1
    example_2 = task_num * 2
    
    return (example_1, example_2)


def find_example_boundaries(prompt: str, example_num: int) -> Optional[Tuple[int, int]]:
    """
    Find the start and end positions of an example in the system prompt.
    
    Looks for pattern:
        Example N
        Prompt: ...
        Cypher query:
        ```cypher
        ...
        ```
    
    Args:
        prompt: The system prompt text
        example_num: The example number to find (1-indexed)
    
    Returns:
        Tuple of (start, end) character positions, or None if not found
    """
    # Pattern to match "Example N" header - handles both LF and CRLF
    pattern = rf'\r?\nExample\s+{example_num}\s*\r?\n'
    
    match = re.search(pattern, prompt, re.IGNORECASE)
    if not match:
        return None
    
    start = match.start()
    
    # Find ANY next example header (not just N+1, since examples might be renumbered)
    any_example_pattern = r'\r?\nExample\s+\d+\s*\r?\n'
    next_match = re.search(any_example_pattern, prompt[match.end():], re.IGNORECASE)
    
    if next_match:
        # End before the next example starts
        end = match.end() + next_match.start()
    else:
        # This is the last example - find the closing backticks after its code block
        # First, find the code block opening for this example
        code_start = prompt.find('```', match.end())
        if code_start != -1:
            # Find the closing backticks after the opening
            code_end = prompt.find('```', code_start + 3)
            if code_end != -1:
                end = code_end + 3
                # Skip trailing newlines
                while end < len(prompt) and prompt[end] in '\r\n':
                    end += 1
            else:
                end = len(prompt)
        else:
            end = len(prompt)
    
    return (start, end)


def build_prompt_excluding_task(prompt: str, task_id: str) -> str:
    """
    Build a modified system prompt with examples for the given task excluded.
    
    This implements Leave-One-Out Cross-Validation (LOOCV) by removing the
    examples that correspond to the evaluation task, preventing data leakage.
    
    Args:
        prompt: The original system prompt with all examples
        task_id: Task identifier like "42.1"
    
    Returns:
        Modified system prompt with matching examples removed
    """
    example_1, example_2 = get_example_pair_for_task(task_id)
    
    # Remove examples in reverse order (higher first) to preserve positions
    modified_prompt = prompt
    for example_num in sorted([example_1, example_2], reverse=True):
        boundaries = find_example_boundaries(modified_prompt, example_num)
        if boundaries:
            start, end = boundaries
            modified_prompt = modified_prompt[:start] + modified_prompt[end:]
    
    return modified_prompt


def count_examples(prompt: str) -> int:
    """Count the number of examples in a system prompt."""
    # Handle both LF and CRLF line endings
    pattern = r'\r?\nExample\s+\d+\s*\r?\n'
    matches = re.findall(pattern, prompt, re.IGNORECASE)
    return len(matches)


# For testing
if __name__ == "__main__":
    import sys
    
    # Load system prompt
    prompt_path = "scripts/system-prompt"
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: {prompt_path} not found")
        sys.exit(1)
    
    original_count = count_examples(prompt)
    print(f"Original prompt has {original_count} examples")
    
    # Test with task 1.1
    task_id = "2.2"
    modified = build_prompt_excluding_task(prompt, task_id)
    print(f"Modified prompt for task {task_id}:\n{modified[:20000]}...")  # Print first 500 chars
    new_count = count_examples(modified)
    
    print(f"After excluding task {task_id}: {new_count} examples")
    print(f"Removed: {original_count - new_count} examples")
