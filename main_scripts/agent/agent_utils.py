"""
Shared Utilities for NLQ4IYP Evaluation Framework

This module provides common utilities used across the evaluation pipeline:
- batch_evaluate.py (automated batch testing)
- gemini_agent.py (interactive CLI)
"""

import json
import re
from typing import Optional
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# STATUS ENUM
# =============================================================================

class TaskStatus(Enum):
    """Possible task completion statuses."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    NEO4J_TIMEOUT = "neo4j_timeout"
    RATE_LIMIT = "rate_limit_exceeded"
    TOKEN_LIMIT = "token_limit"
    RECURSION_LIMIT = "recursion_limit"
    SYNTAX_ERROR = "syntax_error"
    HALLUCINATION = "hallucination"
    ERROR = "error"
    FAILED = "failed"


# =============================================================================
# CONTENT EXTRACTION
# =============================================================================

def extract_text_content(content) -> str:
    """
    Extract text from various LLM response formats.
    
    LLM responses can be:
    - Plain string
    - List of strings
    - List of dicts with {"type": "text", "text": "..."}
    
    This normalizes all formats to a single string.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
        return "".join(text_parts)
    return str(content) if content else ""


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

def is_neo4j_timeout_error(error_msg: str) -> bool:
    """Check if an error message indicates Neo4j timeout."""
    error_lower = error_msg.lower().replace(" ", "")
    return "transactiontimedout" in error_lower or "timedout" in error_lower


def is_rate_limit_error(error_msg: str) -> bool:
    """Check if an error indicates API rate limiting."""
    error_lower = error_msg.lower()
    return any(x in error_lower for x in ["429", "quota", "resourceexhausted"])


def classify_error(error_msg: str) -> tuple[TaskStatus, str]:
    """
    Classify an exception into a status and detail message.
    
    Args:
        error_msg: The error message string
        
    Returns:
        Tuple of (TaskStatus, detail_string)
    """
    error_lower = error_msg.lower()
    error_normalized = error_lower.replace(" ", "")
    
    # Neo4j timeout
    if "transactiontimedout" in error_normalized or "timedout" in error_normalized:
        return TaskStatus.NEO4J_TIMEOUT, "Neo4j query timeout"
    
    # Rate limiting
    if any(x in error_lower for x in ["429", "quota", "resourceexhausted"]):
        return TaskStatus.RATE_LIMIT, "API rate limit hit"
    
    # Token limit
    if "token count exceeds" in error_lower:
        return TaskStatus.TOKEN_LIMIT, "Token limit exceeded"
    
    # Recursion
    if "recursion" in error_lower:
        return TaskStatus.RECURSION_LIMIT, "Recursion limit hit"
    
    # Syntax error
    if "syntax" in error_lower:
        return TaskStatus.SYNTAX_ERROR, "Cypher syntax error"
    
    # Generic error
    return TaskStatus.ERROR, error_msg[:100]


# =============================================================================
# QUERY UTILITIES
# =============================================================================

def classify_query_step(query: str) -> str:
    """
    Classify a Cypher query into a human-readable step name.
    Used for display in CLI and logging.
    """
    query_lower = query.lower()
    if "tolower" in query_lower or "contains" in query_lower:
        return "LOOKING UP VALUE"
    elif "labels(" in query_lower:
        return "CHECKING LABELS"
    elif "count(" in query_lower:
        return "AGGREGATING"
    elif "limit" in query_lower:
        return "FETCHING LIMITED"
    else:
        return "EXECUTING QUERY"


def extract_cypher_from_error(error_msg: str) -> Optional[str]:
    """
    Try to extract the failing Cypher query from a Neo4j error message.
    Useful for retry with LIMIT.
    """
    patterns = [
        r'MATCH\s+.*?(?:RETURN|$)',
        r'CALL\s+.*?(?:YIELD|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
    return None


def add_limit_to_query(query: str, limit: int = 100) -> str:
    """
    Add LIMIT clause to a Cypher query if not already present.
    """
    if "LIMIT" not in query.upper():
        return f"{query.rstrip().rstrip(';')} LIMIT {limit}"
    return query


# =============================================================================
# JSON UTILITIES
# =============================================================================

def safe_json_dumps(data, indent: int = 2) -> str:
    """Safely convert data to JSON string."""
    try:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return json.dumps(parsed, indent=indent, default=str)
            except json.JSONDecodeError:
                return data
        else:
            return json.dumps(data, indent=indent, default=str)
    except TypeError:
        return str(data)


def safe_json_loads(data: str) -> any:
    """Safely parse JSON string."""
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return data


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

@dataclass
class TruncationConfig:
    """Configuration for output truncation."""
    max_lines: int = 20
    max_chars: int = 2000


def truncate_output(text: str, config: TruncationConfig = None) -> tuple[str, bool]:
    """
    Truncate text output if too long.
    
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    config = config or TruncationConfig()
    lines = text.split('\n')
    
    if len(lines) > config.max_lines or len(text) > config.max_chars:
        truncated = '\n'.join(lines[:config.max_lines])
        return truncated, True
    
    return text, False


def count_items_in_json(data) -> Optional[int]:
    """Get item count if data is a list (JSON array)."""
    try:
        if isinstance(data, str):
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return len(parsed)
        elif isinstance(data, list):
            return len(data)
    except (json.JSONDecodeError, TypeError):
        pass
    return None
