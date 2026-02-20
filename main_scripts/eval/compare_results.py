"""
Text-to-Cypher Comparison Tool
Evaluates text-to-Cypher agents by comparing generated vs canonical query results.

Features:
    - Semantic equivalence detection (projections, LIMIT, groupings)
    - Multiple match types: exact, semantic, numeric, partial
    - Type-agnostic comparison (numeric strings match numbers)
    - Statistics by difficulty level and prompt type

Usage:
    python compare_results.py \
        --canonical data/canonical \
        --generated data/generated \
        --output comparison.json \
        --verbose
"""

import json
import argparse
import csv
import os
import glob
import ast
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Any


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    """Comparison thresholds and tolerances."""
    high_quality_threshold: float = 0.9  # F1 >= 0.9 is high quality
    acceptable_threshold: float = 0.5    # F1 >= 0.5 is acceptable
    numeric_tolerance: float = 0.02      # 2% numeric difference allowed


CONFIG = Config()


# =============================================================================
# ENUMS
# =============================================================================

class MatchType(Enum):
    """Classification of comparison results."""
    # Perfect matches
    EXACT = "exact_match"
    BOTH_EMPTY = "both_empty"
    
    # Semantic equivalence
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    SEMANTIC_PROJECTION = "semantic_equivalent_projection"
    SEMANTIC_GROUPED = "semantic_equivalent_grouped"
    SEMANTIC_LIMIT = "semantic_equivalent_limit"
    SEMANTIC_ANSWER = "semantic_equivalent_answer"
    SEMANTIC_NODE_PROPERTY = "semantic_equivalent_node_property"
    
    # Numeric matches
    NUMERIC_EXACT = "numeric_exact_match"
    NUMERIC_NEAR = "numeric_near_match"
    NUMERIC_MISMATCH = "numeric_mismatch"
    
    # Partial matches
    PARTIAL_MULTI_COLUMN = "partial_multi_column"
    HIGH_OVERLAP = "high_overlap"
    PARTIAL = "partial_match"
    LOW_OVERLAP = "low_overlap"
    NO_OVERLAP = "no_overlap"
    
    # Failures
    DIFFERENT_AGGREGATION = "different_aggregation_level"
    BOTH_FAILED = "both_failed"
    CANONICAL_EMPTY = "canonical_empty"
    CANONICAL_FAILED = "canonical_failed"
    GENERATED_EMPTY = "generated_empty"
    GENERATED_FAILED = "generated_failed"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ComparisonResult:
    """Result of comparing two query outputs."""
    execution_accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    jaccard_similarity: float = 0.0
    match_type: str = ""
    canonical_count: int = 0
    generated_count: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    details: str = ""
    is_correct: bool = False
    is_strict_match: bool = False
    percent_difference: Optional[float] = None
    query_similarity: Optional[float] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.percent_difference is None:
            del result['percent_difference']
        if self.query_similarity is None:
            del result['query_similarity']
        return result


@dataclass
class CategoryStats:
    """Aggregated statistics for a category."""
    total: int = 0
    correct: int = 0         # EX: execution accuracy (exact + semantic equivalent)
    high: int = 0
    acceptable: int = 0
    poor: int = 0
    strict_match: int = 0    # Structural exact match on results
    sum_f1: float = 0.0
    sum_precision: float = 0.0
    sum_recall: float = 0.0
    sum_query_similarity: float = 0.0
    query_similarity_count: int = 0

    def rate(self, tier: str) -> float:
        return getattr(self, tier) / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            'total': self.total,
            'EX': self.correct,
            'EX_rate': self.rate('correct'),
            'strict_match': self.strict_match,
            'strict_match_rate': self.rate('strict_match'),
            'high': self.high,
            'acceptable': self.acceptable,
            'poor': self.poor,
            'avg_f1': self.sum_f1 / self.total if self.total else 0.0,
            'avg_precision': self.sum_precision / self.total if self.total else 0.0,
            'avg_recall': self.sum_recall / self.total if self.total else 0.0,
            'avg_query_similarity': self.sum_query_similarity / self.query_similarity_count if self.query_similarity_count else 0.0
        }


# =============================================================================
# VALUE EXTRACTION
# =============================================================================

def flatten_dict(d: dict) -> list:
    """Recursively extract leaf values from a nested dict."""
    values = []
    for v in d.values():
        if v is None:
            continue
        values.extend(flatten_dict(v) if isinstance(v, dict) else [v])
    return values


def try_parse_number(s: str) -> Any:
    """Convert a string to int or float if it looks numeric."""
    s = s.strip()
    if not s:
        return s
    try:
        return round(float(s), 10) if ('.' in s or 'e' in s.lower()) else int(s)
    except ValueError:
        return s


def normalize(val: Any) -> Any:
    """Normalize a value for comparison."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        return round(val, 10)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        return try_parse_number(val.strip())
    if isinstance(val, dict):
        leaves = [normalize(v) for v in flatten_dict(val) if v is not None]
        return tuple(sorted(leaves, key=lambda x: (type(x).__name__, str(x)))) if leaves else None
    if isinstance(val, (list, tuple)):
        return tuple(normalize(v) for v in val if v is not None) or None
    return str(val)


def remove_null_records(results: list) -> list:
    """Filter out records that are entirely NULL."""
    if not results:
        return []
    cleaned = [
        r for r in results
        if not (isinstance(r, dict) and all(v is None for v in r.values())) and r is not None
    ]
    return cleaned or results


def parse_neo4j_repr(value: str) -> Optional[list]:
    """Extract property dicts from Neo4j Path/Node repr strings.

    Parses strings like:
        <Path start=<Node ... properties={'name': 'x'}> end=<Node ... properties={'asn': 1}> size=4>
    Returns list of property dicts, or None if not a repr string.
    """
    if not isinstance(value, str) or 'properties=' not in value:
        return None
    props = []
    for m in re.finditer(r"properties=(\{[^}]*\})", value):
        try:
            d = ast.literal_eval(m.group(1))
            if isinstance(d, dict) and d:
                props.append(d)
        except (ValueError, SyntaxError):
            pass
    return props or None


# Neo4j internal metadata keys that carry no semantic value for comparison
_NEO4J_METADATA_KEYS = frozenset({'_labels', '_element_id', '_start_node_element_id', '_end_node_element_id', '_type'})


def _strip_metadata(val):
    """Recursively remove Neo4j internal metadata keys from dicts."""
    if isinstance(val, dict):
        return {k: _strip_metadata(v) for k, v in val.items() if k not in _NEO4J_METADATA_KEYS}
    if isinstance(val, list):
        return [_strip_metadata(v) for v in val]
    return val


def preprocess_results(results: list) -> list:
    """Normalize result records for comparison.

    - Converts Neo4j Python repr strings (Path/Node) to structured dicts.
    - Strips internal Neo4j metadata (_labels, _element_id, etc.) that has
      no semantic meaning and would corrupt value-level comparisons.
    """
    if not results:
        return results
    processed = []
    for record in results:
        if not isinstance(record, dict):
            processed.append(record)
            continue
        new_rec = {}
        for key, val in record.items():
            parsed = parse_neo4j_repr(val) if isinstance(val, str) else None
            new_rec[key] = (parsed if len(parsed) > 1 else parsed[0]) if parsed else val
        processed.append(_strip_metadata(new_rec))
    return processed


def extract_values(results: list) -> frozenset:
    """Extract all values from results as a comparable set."""
    if not results:
        return frozenset()
    
    tuples = []
    for record in results:
        if isinstance(record, dict):
            leaves = flatten_dict(record)
            if leaves:
                normalized = [normalize(v) for v in leaves if v is not None]
                if normalized:
                    tuples.append(tuple(sorted(normalized, key=lambda x: (type(x).__name__, str(x)))))
        elif (val := normalize(record)) is not None:
            tuples.append(val)
    return frozenset(tuples)


def extract_primitives(results: list) -> set:
    """Recursively extract all primitive values from nested structures."""
    values = set()
    
    def extract(val):
        if val is None:
            return
        if isinstance(val, dict):
            for v in val.values():
                extract(v)
        elif isinstance(val, (list, tuple)):
            for v in val:
                extract(v)
        elif isinstance(val, (str, int, float)) and not isinstance(val, bool):
            if (n := normalize(val)) is not None:
                values.add(n)
    
    for record in results:
        extract(record)
    return values


def _is_id_key(key: str) -> bool:
    """Check if key refers to an identifier field (exact base-name match)."""
    _ID_KEYS = frozenset({'asn', 'name', 'prefix', 'ip', 'country_code', 'countrycode', 'id', 'url'})
    return key.lower().rsplit('.', 1)[-1] in _ID_KEYS


def extract_identifiers(results: list) -> set:
    """Extract identifier values (asn, name, prefix, ip, etc.)."""
    values = set()
    for record in results:
        if not isinstance(record, dict):
            continue
        for key, val in record.items():
            if isinstance(val, dict):
                for nested_key, nested_val in val.items():
                    if _is_id_key(nested_key) and nested_val is not None:
                        values.add(normalize(nested_val))
            elif _is_id_key(key) and val is not None:
                values.add(normalize(val))
    return values


def extract_numeric_columns(results: list) -> dict:
    """Extract numeric values grouped by column name."""
    data = defaultdict(list)
    for record in results:
        if isinstance(record, dict):
            for key, val in record.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    data[key].append(val)
    return dict(data)


def extract_property(results: list, prop: str = None) -> set:
    """Extract specific property values from nested structures."""
    values = set()
    for record in results:
        if not isinstance(record, dict):
            continue
        for key, val in record.items():
            if isinstance(val, dict):
                if prop and prop in val and (n := normalize(val[prop])) is not None:
                    values.add(n)
                elif not prop:
                    values.update(n for v in val.values() if (n := normalize(v)) is not None)
            elif val is not None and (prop is None or key == prop):
                values.add(normalize(val))
    return values


# =============================================================================
# PATH RESULTS
# =============================================================================

# Identifier keys used to fingerprint path nodes
_PATH_ID_KEYS = ('asn', 'name', 'prefix', 'ip')


def _extract_path_nodes(val) -> list:
    """Extract node property dicts from either Neo4j path serialization:
    - list format : [node_dict, "REL_TYPE", node_dict, ...]
    - dict format : {"_nodes": [...], "_relationships": [...]}
    """
    if isinstance(val, list) and len(val) >= 3:
        nodes = [v for v in val if isinstance(v, dict)]
        if nodes and any(isinstance(v, str) for v in val):
            return nodes
    if isinstance(val, dict) and '_nodes' in val:
        return val['_nodes']
    return []


def _path_signature(nodes: list) -> tuple:
    """Fingerprint a path as a sorted tuple of its node identifier values."""
    ids = [
        normalize(node[key])
        for node in nodes if isinstance(node, dict)
        for key in _PATH_ID_KEYS
        if node.get(key) is not None
    ]
    return tuple(sorted(ids, key=lambda x: (type(x).__name__, str(x))))


def _path_signatures(results: list) -> frozenset:
    """Return the set of unique path signatures across all result records."""
    sigs = set()
    for record in results:
        if isinstance(record, dict):
            for val in record.values():
                nodes = _extract_path_nodes(val)
                if nodes and (sig := _path_signature(nodes)):
                    sigs.add(sig)
    return frozenset(sigs)


def _has_path_values(results: list) -> bool:
    """Return True if any result record contains a path-formatted value."""
    return any(
        _extract_path_nodes(v)
        for record in results if isinstance(record, dict)
        for v in record.values()
    )


def compare_path_results(canonical: list, generated: list) -> Optional[ComparisonResult]:
    """Compare path results across the two supported Neo4j path serialization formats.

    Nodes are fingerprinted by identifier properties (asn, name, prefix, ip); relationship
    metadata and raw row counts are ignored.  Returns None when only one side has path
    values, allowing flat-column comparators to handle the fallback.
    """
    if not _has_path_values(canonical) or not _has_path_values(generated):
        return None

    can_sigs = _path_signatures(canonical)
    gen_sigs = _path_signatures(generated)

    if not can_sigs or not gen_sigs:
        return None

    tp = len(can_sigs & gen_sigs)
    fp = len(gen_sigs - can_sigs)
    fn = len(can_sigs - gen_sigs)
    prec = tp / len(gen_sigs) if gen_sigs else 0.0
    rec  = tp / len(can_sigs) if can_sigs else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    jacc = tp / len(can_sigs | gen_sigs) if (can_sigs | gen_sigs) else 0.0

    # Semantically equivalent: same unique paths (possibly different raw counts / filters)
    if fn == 0 and fp == 0:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_EQUIVALENT.value,
            len(canonical), len(generated), tp, 0, 0,
            f"Path signatures match ({tp} unique paths)", True, False
        )

    # Generated is a proper superset — canonical fully covered
    if fn == 0:
        return ComparisonResult(
            1.0, prec, 1.0, f1, jacc, MatchType.SEMANTIC_EQUIVALENT.value,
            len(canonical), len(generated), tp, fp, 0,
            f"Path superset: all {tp} canonical paths present (+{fp} extra)", True, False
        )

    details = f"Path TP:{tp} FP:{fp} FN:{fn} | P:{prec:.3f} R:{rec:.3f} F1:{f1:.3f}"
    if f1 >= CONFIG.high_quality_threshold:
        mt = MatchType.HIGH_OVERLAP.value
    elif f1 >= CONFIG.acceptable_threshold:
        mt = MatchType.PARTIAL.value
    elif f1 > 0:
        mt = MatchType.LOW_OVERLAP.value
    else:
        mt = MatchType.NO_OVERLAP.value

    return ComparisonResult(0.0, prec, rec, f1, jacc, mt, len(canonical), len(generated), tp, fp, fn, details)


# =============================================================================
# QUERY PARSING
# =============================================================================

@dataclass
class QueryFeatures:
    """Semantic features extracted from a Cypher query."""
    has_distinct: bool = False
    has_count: bool = False
    has_aggregation: bool = False
    has_limit: bool = False
    return_items: list = field(default_factory=list)
    returns_properties: bool = False
    returns_nodes: bool = False


def parse_query(query: str) -> QueryFeatures:
    """Extract semantic features from a Cypher query."""
    if not query:
        return QueryFeatures()
    
    upper = query.upper()
    match = re.search(r'RETURN\s+(.+?)(?:ORDER\s+BY|LIMIT|$)', query, re.IGNORECASE | re.DOTALL)
    return_clause = match.group(1).strip() if match else ""
    return_items = [item.strip() for item in return_clause.split(',') if item.strip()]
    
    aggregations = ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(', 'COLLECT(']
    has_agg = any(agg in upper for agg in aggregations)
    returns_props = any('.' in item and 'COUNT' not in item.upper() for item in return_items)
    
    return QueryFeatures(
        has_distinct='DISTINCT' in upper,
        has_count='COUNT(' in upper,
        has_aggregation=has_agg,
        has_limit='LIMIT' in upper,
        return_items=return_items,
        returns_properties=returns_props,
        returns_nodes=not returns_props and not has_agg
    )


def compute_query_similarity(query1: str, query2: str) -> float:
    """Token-based Jaccard similarity of two Cypher queries."""
    if not query1 or not query2:
        return 0.0

    def tokenize(q: str) -> Counter:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*|[0-9]+(?:\.[0-9]+)?|[<>=!]+|'[^']*'", q)
        return Counter(t.upper() if not t.startswith("'") else t.lower() for t in tokens)

    t1 = tokenize(query1)
    t2 = tokenize(query2)

    intersection = sum((t1 & t2).values())
    union = sum((t1 | t2).values())
    return round(intersection / union, 4) if union else 0.0


# =============================================================================
# METRICS
# =============================================================================

def precision_recall_f1(canonical: frozenset, generated: frozenset) -> tuple:
    """Calculate precision, recall, and F1 score."""
    if not generated and not canonical:
        return 1.0, 1.0, 1.0
    if not generated or not canonical:
        return 0.0, 0.0, 0.0
    
    intersection = canonical & generated
    prec = len(intersection) / len(generated)
    rec = len(intersection) / len(canonical)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def jaccard(a: frozenset, b: frozenset) -> float:
    """Calculate Jaccard similarity."""
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# =============================================================================
# SPECIALIZED COMPARATORS
# =============================================================================

def compare_single_numeric(canonical: list, generated: list) -> Optional[ComparisonResult]:
    """Compare single numeric results."""
    if len(canonical) != 1 or len(generated) != 1:
        return None
    
    can_vals = extract_values(canonical)
    gen_vals = extract_values(generated)
    if len(can_vals) != 1 or len(gen_vals) != 1:
        return None
    
    can_val = list(can_vals)[0]
    gen_val = list(gen_vals)[0]
    
    # Unwrap single-element tuples
    if isinstance(can_val, tuple) and len(can_val) == 1:
        can_val = can_val[0]
    if isinstance(gen_val, tuple) and len(gen_val) == 1:
        gen_val = gen_val[0]
    
    if not (isinstance(can_val, (int, float)) and isinstance(gen_val, (int, float))):
        return None
    
    can_f, gen_f = float(can_val), float(gen_val)
    
    # Both zero
    if can_f == 0 and gen_f == 0:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.NUMERIC_EXACT.value,
            1, 1, 1, 0, 0, "Both values are 0", True, True, 0.0
        )
    
    # Canonical is zero (avoid division by zero)
    if can_f == 0:
        return ComparisonResult(
            match_type=MatchType.NUMERIC_MISMATCH.value,
            canonical_count=1, generated_count=1,
            false_positives=1, false_negatives=1,
            details=f"Canonical is 0, generated is {gen_f}",
            percent_difference=100.0
        )
    
    diff = abs(can_f - gen_f) / max(abs(can_f), abs(gen_f)) * 100
    
    if can_f == gen_f:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.NUMERIC_EXACT.value,
            1, 1, 1, 0, 0, f"Exact match: {can_f}", True, True, 0.0
        )
    elif diff < CONFIG.numeric_tolerance * 100:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.NUMERIC_NEAR.value,
            1, 1, 1, 0, 0, f"Values close: {can_f} vs {gen_f} ({diff:.2f}%)", True, False, diff
        )
    
    return ComparisonResult(
        match_type=MatchType.NUMERIC_MISMATCH.value,
        canonical_count=1, generated_count=1,
        false_positives=1, false_negatives=1,
        details=f"Values differ: {can_f} vs {gen_f} ({diff:.2f}%)",
        percent_difference=diff
    )


def compare_multi_column(canonical: list, generated: list) -> Optional[ComparisonResult]:
    """Compare single-record results with multiple columns."""
    if len(canonical) != 1 or len(generated) != 1:
        return None
    
    can_rec, gen_rec = canonical[0], generated[0]
    if not (isinstance(can_rec, dict) and isinstance(gen_rec, dict)):
        return None
    if len(can_rec) < 2 or len(gen_rec) < 2:
        return None
    
    can_nums = {k: v for k, v in can_rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    gen_nums = {k: v for k, v in gen_rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if len(can_nums) != len(gen_nums):
        return None
    
    can_sorted = sorted(can_nums.items(), key=lambda x: x[1])
    gen_sorted = sorted(gen_nums.items(), key=lambda x: x[1])
    matches, mismatches = [], []
    
    for (ck, cv), (gk, gv) in zip(can_sorted, gen_sorted):
        if cv == gv:
            matches.append(f"{ck}={cv}")
        elif cv > 0 and abs(cv - gv) / cv < 0.1:
            matches.append(f"{ck}≈{cv}")
        else:
            mismatches.append(f"{ck}:{cv} vs {gk}:{gv}")
    
    if matches and not mismatches:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.EXACT.value, 1, 1, len(matches), 0, 0,
            f"All columns match: {', '.join(matches)}", True, True
        )
    if matches and mismatches:
        # If identifier columns match, the core answer is correct despite metric differences
        def _id_vals(rec):
            return {normalize(v) for k, v in rec.items()
                    if _is_id_key(k) and v is not None}
        can_id_vals = _id_vals(can_rec)
        gen_id_vals = _id_vals(gen_rec)
        if can_id_vals and can_id_vals == gen_id_vals:
            return ComparisonResult(
                1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_ANSWER.value, 1, 1,
                len(matches) + len(mismatches), 0, 0,
                f"Identifiers match: {', '.join(matches)}; metrics differ: {', '.join(mismatches)}",
                True, False
            )
        return ComparisonResult(
            0.0, 0.5, 0.5, 0.5, 0.5, MatchType.PARTIAL_MULTI_COLUMN.value, 1, 1,
            len(matches), len(mismatches), len(mismatches),
            f"Partial: {', '.join(matches)} but {', '.join(mismatches)}"
        )
    return None


def compare_single_row_answer(canonical: list, generated: list) -> Optional[ComparisonResult]:
    """Compare single-row results focusing on identifier values."""
    if len(canonical) != 1 or len(generated) != 1:
        return None
    if not (isinstance(canonical[0], dict) and isinstance(generated[0], dict)):
        return None
    
    can_ids = extract_identifiers(canonical)
    gen_ids = extract_identifiers(generated)
    
    if can_ids and gen_ids and can_ids == gen_ids:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_ANSWER.value, 1, 1,
            len(can_ids), 0, 0, f"Same answer values: {can_ids}", True, False
        )
    
    if can_ids and gen_ids and gen_ids.issubset(can_ids) and gen_ids:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_ANSWER.value, 1, 1,
            len(gen_ids), 0, 0, f"Generated returns subset: {gen_ids}", True, False
        )
    
    can_all = extract_primitives(canonical)
    gen_all = extract_primitives(generated)
    
    if can_all and gen_all:
        overlap = can_all & gen_all
        if len(overlap) >= max(1, min(len(can_all), len(gen_all)) * 0.5):
            prec = len(overlap) / len(gen_all) if gen_all else 0
            rec = len(overlap) / len(can_all) if can_all else 0
            if prec >= 0.8 or rec >= 0.8:
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
                return ComparisonResult(
                    1.0, prec, rec, f1, len(overlap) / len(can_all | gen_all),
                    MatchType.SEMANTIC_ANSWER.value, 1, 1, len(overlap), 0, 0,
                    f"High value overlap: {len(overlap)}/{len(can_all)} match", True, False
                )
    return None


def compare_node_vs_property(canonical: list, generated: list, can_q: str, gen_q: str) -> Optional[ComparisonResult]:
    """Handle different projection styles: node vs property returns."""
    if not canonical or not generated:
        return None
    
    can_prims = extract_primitives(canonical)
    gen_prims = extract_primitives(generated)
    if not can_prims or not gen_prims:
        return None
    
    if can_prims.issubset(gen_prims):
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_NODE_PROPERTY.value,
            len(canonical), len(generated), len(can_prims), 0, 0,
            f"Generated returns full nodes ({len(can_prims)} values)", True, False
        )
    
    if gen_prims.issubset(can_prims):
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_NODE_PROPERTY.value,
            len(canonical), len(generated), len(gen_prims), 0, 0,
            f"Generated returns properties ({len(gen_prims)} values)", True, False
        )
    
    overlap = can_prims & gen_prims
    if overlap:
        jacc = len(overlap) / len(can_prims | gen_prims)
        if jacc >= 0.7:
            prec = len(overlap) / len(gen_prims)
            rec = len(overlap) / len(can_prims)
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            return ComparisonResult(
                1.0, prec, rec, f1, jacc,
                MatchType.SEMANTIC_NODE_PROPERTY.value, len(canonical), len(generated),
                len(overlap), len(gen_prims - can_prims), len(can_prims - gen_prims),
                f"High primitive overlap: {jacc:.1%}", True, False
            )
    return None


# Country code <-> name normalization
_COUNTRY_MAP: dict = {}


def _build_country_map() -> dict:
    """Lazy-build bidirectional country name/code normalization map."""
    if _COUNTRY_MAP:
        return _COUNTRY_MAP
    entries = {
        'JP': ['japan'], 'CN': ['china'],
        'US': ['united states of america', 'united states'],
        'SG': ['singapore'], 'IL': ['israel'], 'IN': ['india'],
        'AU': ['australia'], 'NL': ['netherlands'],
        'GB': ['united kingdom of great britain and northern ireland', 'united kingdom'],
        'DE': ['germany'], 'FR': ['france'], 'BR': ['brazil'], 'CA': ['canada'],
        'KR': ['republic of korea', 'south korea', 'korea'],
        'RU': ['russian federation', 'russia'], 'IT': ['italy'], 'ES': ['spain'],
        'MX': ['mexico'], 'ZA': ['south africa'], 'SE': ['sweden'],
        'CH': ['switzerland'], 'NO': ['norway'], 'DK': ['denmark'], 'FI': ['finland'],
        'BE': ['belgium'], 'AT': ['austria'], 'PT': ['portugal'], 'IE': ['ireland'],
        'NZ': ['new zealand'], 'PL': ['poland'], 'CZ': ['czechia', 'czech republic'],
        'HU': ['hungary'], 'RO': ['romania'], 'GR': ['greece'],
        'TR': ['turkey', 'türkiye'], 'HK': ['hong kong'], 'TW': ['taiwan'],
        'ID': ['indonesia'], 'MY': ['malaysia'], 'TH': ['thailand'],
        'PH': ['philippines'], 'VN': ['vietnam', 'viet nam'],
        'AR': ['argentina'], 'CL': ['chile'], 'CO': ['colombia'], 'PE': ['peru'],
        'EG': ['egypt'], 'NG': ['nigeria'], 'KE': ['kenya'], 'UA': ['ukraine'],
        'LU': ['luxembourg'], 'IS': ['iceland'], 'HR': ['croatia'],
        'BG': ['bulgaria'], 'RS': ['serbia'], 'SK': ['slovakia'],
        'SI': ['slovenia'], 'EE': ['estonia'], 'LV': ['latvia'],
        'LT': ['lithuania'], 'CY': ['cyprus'], 'MT': ['malta'],
        'SA': ['saudi arabia'], 'AE': ['united arab emirates'],
        'PK': ['pakistan'], 'BD': ['bangladesh'], 'IR': ['iran'],
        'IQ': ['iraq'], 'QA': ['qatar'],
    }
    for code, names in entries.items():
        _COUNTRY_MAP[code.lower()] = code
        for name in names:
            _COUNTRY_MAP[name.lower()] = code
    return _COUNTRY_MAP


def normalize_country(val: Any) -> Optional[str]:
    """Normalize a country name or code to ISO 2-letter code, or None."""
    if not isinstance(val, str):
        return None
    return _build_country_map().get(val.strip().lower())


def compare_normalized_values(canonical: list, generated: list) -> Optional[ComparisonResult]:
    """Compare results after normalizing domain-specific representations (e.g. 'JP' vs 'Japan')."""
    if not canonical or not generated or len(canonical) != len(generated):
        return None
    if not all(isinstance(r, dict) for r in canonical + generated):
        return None

    def _normalize_record(record: dict) -> tuple:
        vals = []
        for v in record.values():
            cc = normalize_country(v) if isinstance(v, str) else None
            vals.append(cc if cc else normalize(v))
        return tuple(sorted(vals, key=lambda x: (type(x).__name__, str(x))))

    can_set = frozenset(_normalize_record(r) for r in canonical)
    gen_set = frozenset(_normalize_record(r) for r in generated)

    if can_set == gen_set:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_EQUIVALENT.value,
            len(canonical), len(generated), len(can_set), 0, 0,
            f"Match after value normalization ({len(can_set)} distinct rows)", True, False
        )

    intersection = can_set & gen_set
    if intersection and len(intersection) / max(len(can_set), len(gen_set)) >= 0.8:
        prec = len(intersection) / len(gen_set) if gen_set else 0
        rec = len(intersection) / len(can_set) if can_set else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return ComparisonResult(
            1.0, prec, rec, f1, len(intersection) / len(can_set | gen_set),
            MatchType.SEMANTIC_EQUIVALENT.value, len(canonical), len(generated),
            len(intersection), len(gen_set - can_set), len(can_set - gen_set),
            f"High match after value normalization (F1={f1:.3f})", True, False
        )
    return None


def detect_projection_relationship(canonical: list, generated: list, can_q: str, gen_q: str) -> tuple:
    """Detect if generated is a valid projection of canonical."""
    can_f = parse_query(can_q)
    gen_f = parse_query(gen_q)
    common_props = ['name', 'asn', 'country_code', 'countrycode', 'prefix', 'ip']
    
    if gen_f.has_distinct and not can_f.has_distinct:
        for prop in common_props:
            can_vals = extract_property(canonical, prop)
            gen_vals = extract_property(generated, prop)
            if can_vals and gen_vals:
                if gen_vals.issubset(can_vals):
                    return True, f"Generated extracts DISTINCT {prop}", 1.0
                overlap = len(can_vals & gen_vals)
                if overlap and (prec := overlap / len(gen_vals)) >= 0.8:
                    return True, f"Generated extracts {prop} ({prec:.0%} precision)", prec
    
    if gen_f.returns_properties and can_f.returns_nodes:
        gen_vals = extract_values(generated)
        for prop in common_props:
            can_vals = extract_property(canonical, prop)
            if can_vals and gen_vals and (ratio := len(can_vals & gen_vals) / len(gen_vals)) >= 0.8:
                return True, f"Generated returns {prop} property", ratio
    
    return False, "", 0.0


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def results_structurally_equal(canonical: list, generated: list) -> bool:
    """Check if two result sets are structurally identical."""
    if len(canonical) != len(generated):
        return False
    try:
        can_sorted = sorted(canonical, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        gen_sorted = sorted(generated, key=lambda x: json.dumps(x, sort_keys=True, default=str))
        return can_sorted == gen_sorted
    except Exception:
        return False


def compare(canonical_results: list, generated_results: list, can_q: str = "", gen_q: str = "") -> ComparisonResult:
    """Compare canonical vs generated query results."""
    canonical = preprocess_results(remove_null_records(canonical_results))
    generated = preprocess_results(remove_null_records(generated_results))
    
    # Empty handling
    if not canonical and not generated:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.BOTH_EMPTY.value,
            details="Both queries returned no results", is_correct=True, is_strict_match=True
        )
    
    if not canonical:
        return ComparisonResult(
            match_type=MatchType.CANONICAL_EMPTY.value,
            generated_count=len(generated), false_positives=len(generated),
            details="Canonical empty, generated has results"
        )
    
    if not generated:
        return ComparisonResult(
            match_type=MatchType.GENERATED_EMPTY.value,
            canonical_count=len(canonical), false_negatives=len(canonical),
            details="Generated returned no results"
        )
    
    # Check for true structural equality
    structurally_equal = results_structurally_equal(canonical, generated)
    
    # Try specialized comparators
    for comparator in [compare_path_results, compare_multi_column, compare_single_numeric, compare_single_row_answer]:
        if result := comparator(canonical, generated):
            return result
    
    # Projection and set comparison
    is_proj, proj_desc, proj_score = detect_projection_relationship(canonical, generated, can_q, gen_q)
    can_set = extract_values(canonical)
    gen_set = extract_values(generated)
    prec, rec, f1 = precision_recall_f1(can_set, gen_set)
    jacc = jaccard(can_set, gen_set)
    
    intersection = can_set & gen_set
    tp = len(intersection)
    fp = len(gen_set - can_set)
    fn = len(can_set - gen_set)
    exact_match = can_set == gen_set
    
    gen_f = parse_query(gen_q)
    can_f = parse_query(can_q)
    
    # LIMIT semantic equivalence
    if gen_f.has_limit and not can_f.has_limit and prec == 1.0 and fp == 0 and fn > 0:
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_LIMIT.value,
            len(canonical), len(generated), tp, 0, 0,
            f"Generated has LIMIT, returns perfect subset ({tp} of {len(can_set)})", True, False
        )
    
    # LIMIT with different projection (identifiers still match)
    if gen_f.has_limit and not can_f.has_limit and fp > 0 and fn > 0:
        can_ids = extract_identifiers(canonical)
        gen_ids = extract_identifiers(generated)
        if gen_ids and can_ids and gen_ids.issubset(can_ids) and len(gen_ids) >= 2:
            return ComparisonResult(
                1.0, 1.0, 1.0, 1.0, 1.0, MatchType.SEMANTIC_LIMIT.value,
                len(canonical), len(generated), len(gen_ids), 0, 0,
                f"LIMIT subset with matching identifiers ({len(gen_ids)}/{len(can_ids)} ids)",
                True, False
            )
    
    # Projection equivalence
    if is_proj and proj_score >= 0.8:
        return ComparisonResult(
            1.0, prec, rec, f1, jacc,
            MatchType.SEMANTIC_PROJECTION.value, len(canonical), len(generated),
            tp, fp, fn, proj_desc, True, False
        )
    
    # Exact match - values match, check structure
    if exact_match:
        is_strict = structurally_equal
        match_type = MatchType.EXACT.value if is_strict else MatchType.SEMANTIC_EQUIVALENT.value
        return ComparisonResult(
            1.0, 1.0, 1.0, 1.0, 1.0, match_type,
            len(canonical), len(generated), tp, 0, 0,
            f"TP:{tp} FP:{fp} FN:{fn}", True, is_strict
        )
    
    # Node vs property fallback
    if result := compare_node_vs_property(canonical, generated, can_q, gen_q):
        return result
    
    # Value normalization fallback (country codes, etc.)
    if result := compare_normalized_values(canonical, generated):
        return result
    
    # Partial matches
    details = f"TP:{tp} FP:{fp} FN:{fn} | P:{prec:.3f} R:{rec:.3f} F1:{f1:.3f}"
    if f1 >= CONFIG.high_quality_threshold:
        match_type = MatchType.HIGH_OVERLAP.value
    elif f1 >= CONFIG.acceptable_threshold:
        match_type = MatchType.PARTIAL.value
    elif f1 > 0:
        match_type = MatchType.LOW_OVERLAP.value
    else:
        match_type = MatchType.NO_OVERLAP.value
        details = f"TP:{tp} FP:{fp} FN:{fn} | No common results"
    
    return ComparisonResult(
        0.0, prec, rec, f1, jacc, match_type, len(canonical), len(generated),
        tp, fp, fn, details
    )


# =============================================================================
# STATISTICS
# =============================================================================

def parse_difficulty(text: str) -> tuple:
    """Parse difficulty string into level and prompt type."""
    if not text:
        return "Unknown", "Unknown"
    parts = text.lower().split()
    level = next((d for d in ["Easy", "Medium", "Hard"] if d.lower() in parts), "Unknown")
    prompt = next((p for p in ["Technical", "General"] if p.lower() in parts), "Unknown")
    return level, prompt


def get_tier(comp: dict) -> str:
    """Determine performance tier for a comparison."""
    if comp.get('is_correct', False):
        return 'correct'
    f1 = comp.get('f1_score', 0)
    if f1 >= CONFIG.high_quality_threshold:
        return 'high'
    if f1 >= CONFIG.acceptable_threshold:
        return 'acceptable'
    return 'poor'


def calculate_statistics(comparisons: list) -> dict:
    """Calculate statistics by category."""
    stats = {
        'by_difficulty': defaultdict(CategoryStats),
        'by_prompt_type': defaultdict(CategoryStats),
        'by_combined': defaultdict(CategoryStats)
    }
    
    for comp in comparisons:
        level, prompt = parse_difficulty(comp.get('difficulty', ''))
        tier = get_tier(comp)
        
        for key, name in [('by_difficulty', level), ('by_prompt_type', prompt), ('by_combined', f"{level}_{prompt}")]:
            s = stats[key][name]
            s.total += 1
            setattr(s, tier, getattr(s, tier) + 1)
            if comp.get('is_strict_match', False):
                s.strict_match += 1
            s.sum_f1 += comp.get('f1_score', 0)
            s.sum_precision += comp.get('precision', 0)
            s.sum_recall += comp.get('recall', 0)
            q_sim = comp.get('query_similarity')
            if q_sim is not None:
                s.sum_query_similarity += q_sim
                s.query_similarity_count += 1
    
    return stats


def get_tier_counts(comparisons: list) -> dict:
    """Get overall tier counts."""
    correct = sum(1 for c in comparisons if c.get('is_correct', False))
    strict = sum(1 for c in comparisons if c.get('is_strict_match', False))
    non_correct = [c for c in comparisons if not c.get('is_correct', False)]
    high = sum(1 for c in non_correct if c.get('f1_score', 0) >= CONFIG.high_quality_threshold)
    acceptable = sum(1 for c in non_correct if CONFIG.acceptable_threshold <= c.get('f1_score', 0) < CONFIG.high_quality_threshold)
    poor = sum(1 for c in non_correct if c.get('f1_score', 0) < CONFIG.acceptable_threshold)
    
    return {'strict_match': strict, 'EX': correct, 'high': high, 'acceptable': acceptable, 'poor': poor}


# =============================================================================
# DISPLAY
# =============================================================================

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"


def status_symbol(comp: dict) -> str:
    """Get colored status symbol (ASCII-safe for Windows)."""
    if comp.get('is_correct', False):
        return f"{GREEN}+{RESET}"
    f1 = comp.get('f1_score', 0)
    if f1 >= CONFIG.high_quality_threshold:
        return f"{YELLOW}!{RESET}"
    if f1 >= CONFIG.acceptable_threshold:
        return f"{YELLOW}~{RESET}"
    return f"{RED}x{RESET}"


def print_statistics(stats: dict):
    """Print category statistics."""
    print(f"\n{'='*80}\nSTATISTICS BY DIFFICULTY\n{'='*80}")
    for level in ['Easy', 'Medium', 'Hard']:
        if level in stats['by_difficulty']:
            s = stats['by_difficulty'][level]
            print(f"\n{level}: (n={s.total})")
            print(f"  Strict: {s.strict_match:3d} ({s.rate('strict_match'):5.1%}) | "
                  f"EX: {s.correct:3d} ({s.rate('correct'):5.1%}) | "
                  f"High: {s.high:3d} | Acceptable: {s.acceptable:3d} | Poor: {s.poor:3d}")
            if s.total:
                print(f"  Avg F1: {s.sum_f1/s.total:.3f} | Avg Query Sim: {s.sum_query_similarity/s.query_similarity_count:.3f}" if s.query_similarity_count else f"  Avg F1: {s.sum_f1/s.total:.3f}")
    
    print(f"\n{'='*80}\nSTATISTICS BY PROMPT TYPE\n{'='*80}")
    for ptype in ['Technical', 'General']:
        if ptype in stats['by_prompt_type']:
            s = stats['by_prompt_type'][ptype]
            print(f"\n{ptype}: (n={s.total})")
            print(f"  Strict: {s.strict_match:3d} ({s.rate('strict_match'):5.1%}) | "
                  f"EX: {s.correct:3d} ({s.rate('correct'):5.1%}) | "
                  f"High: {s.high:3d} | Acceptable: {s.acceptable:3d} | Poor: {s.poor:3d}")


def print_summary(comparisons: list):
    """Print overall summary."""
    total = len(comparisons)
    if not total:
        print("\nNo comparisons to summarize.")
        return
    
    tiers = get_tier_counts(comparisons)
    avg_f1 = sum(c.get('f1_score', 0) for c in comparisons) / total
    q_sims = [c.get('query_similarity') for c in comparisons if c.get('query_similarity') is not None]
    match_types = Counter(c.get('match_type', 'unknown') for c in comparisons)
    
    print(f"\n{'='*80}\nOVERALL SUMMARY\n{'='*80}")
    print(f"Total: {total}\n\nPerformance Tiers:")
    print(f"  Strict Match:   {tiers['strict_match']:3d} ({tiers['strict_match']/total:5.1%})  <- structural exact match")
    print(f"  EX (Correct):   {tiers['EX']:3d} ({tiers['EX']/total:5.1%})  <- exact + semantic equivalent")
    print(f"  High Quality:   {tiers['high']:3d} ({tiers['high']/total:5.1%})")
    print(f"  Acceptable:     {tiers['acceptable']:3d} ({tiers['acceptable']/total:5.1%})")
    print(f"  Poor/Failed:    {tiers['poor']:3d} ({tiers['poor']/total:5.1%})")
    print(f"\nAverage F1: {avg_f1:.3f}")
    if q_sims:
        print(f"Average Query Similarity: {sum(q_sims)/len(q_sims):.3f}")
    print(f"\nMatch Types:")
    for mt, count in match_types.most_common(8):
        print(f"  {mt:35s}: {count:3d} ({count/total:5.1%})")


def print_failures(comparisons: list):
    """Print failure analysis."""
    failures = [c for c in comparisons if not c.get('is_correct', False)]
    if not failures:
        return
    
    print(f"\n{'='*80}\nFAILURE ANALYSIS ({len(failures)} tasks)\n{'='*80}")
    by_type = Counter(c.get('match_type', 'unknown') for c in failures)
    print("\nBy Type:")
    for t, count in by_type.most_common():
        print(f"  {t:30s}: {count:3d} ({count/len(failures):5.1%})")
    
    print("\nWorst 10:")
    for c in sorted(failures, key=lambda x: x.get('f1_score', 0))[:10]:
        print(f"  Task {c['task_id']:6s} | F1: {c['f1_score']:.3f} | {c.get('match_type', '')}")


# =============================================================================
# I/O
# =============================================================================

def load_results(path: str) -> dict:
    """Load results from directory or JSON file."""
    if os.path.isdir(path):
        results = {}
        for file in glob.glob(os.path.join(path, "task_*.json")):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if task_id := str(data.get('task_id', '')):
                    results[task_id] = data
        return results
    
    with open(path, 'r', encoding='utf-8') as f:
        return {str(item['task_id']): item for item in json.load(f)}


def save_json(comparisons: list, stats: dict, output: str):
    """Save results to JSON."""
    total = len(comparisons)
    tiers = get_tier_counts(comparisons)
    q_sims = [c.get('query_similarity') for c in comparisons if c.get('query_similarity') is not None]
    avg_q_sim = sum(q_sims) / len(q_sims) if q_sims else 0
    
    data = {
        "metadata": {
            "description": "Text-to-Cypher Evaluation",
            "thresholds": {
                "high_quality_f1": CONFIG.high_quality_threshold,
                "acceptable_f1": CONFIG.acceptable_threshold
            }
        },
        "summary": {
            "total_tasks": total,
            "strict_match": tiers['strict_match'],
            "strict_match_rate": tiers['strict_match'] / total if total else 0,
            "EX": tiers['EX'],
            "EX_rate": tiers['EX'] / total if total else 0,
            "tier_high": tiers['high'],
            "tier_acceptable": tiers['acceptable'],
            "tier_poor": tiers['poor'],
            "avg_f1": sum(c.get('f1_score', 0) for c in comparisons) / total if total else 0,
            "avg_query_similarity": avg_q_sim
        },
        "category_statistics": {
            key: {k: v.to_dict() for k, v in cat.items()}
            for key, cat in stats.items()
        },
        "comparisons": comparisons
    }
    
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n[SAVED] JSON: {output}")


def save_csv(comparisons: list, output: str):
    """Save results to CSV."""
    fields = [
        "task_id", "difficulty", "prompt",
        "canonical_query", "generated_query",
        "canonical_success", "generated_success",
        "is_correct", "is_strict_match", "match_type", "execution_accuracy",
        "f1_score", "precision", "recall", "jaccard_similarity", "query_similarity",
        "canonical_count", "generated_count",
        "true_positives", "false_positives", "false_negatives", "details"
    ]
    
    with open(output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparisons)
    print(f"[SAVED] CSV: {output}")


# =============================================================================
# MAIN
# =============================================================================

def process_all(canonical_data: dict, generated_data: dict, verbose: bool = False) -> list:
    """Process all comparisons."""
    all_ids = sorted(
        set(canonical_data.keys()) | set(generated_data.keys()),
        key=lambda x: (float(x.split('.')[0]), float(x.split('.')[1]) if '.' in x else 0)
    )
    
    print(f"\nComparing {len(all_ids)} tasks...\n" + "=" * 80)
    results = []
    
    for task_id in all_ids:
        canonical = canonical_data.get(task_id, {})
        generated = generated_data.get(task_id, {})
        can_ok = canonical.get('success', False)
        gen_ok = generated.get('success', False)
        can_results = canonical.get('results', [])
        gen_results = generated.get('results', [])
        can_q = canonical.get('query', '')
        gen_q = generated.get('query', '')
        
        # Compute query text similarity (useful even when execution fails)
        q_sim = compute_query_similarity(can_q, gen_q) if can_q and gen_q else None
        
        if can_ok and gen_ok:
            result = compare(can_results, gen_results, can_q, gen_q)
        elif not can_ok and not gen_ok:
            sim_note = f" | Query similarity: {q_sim:.1%}" if q_sim is not None else ""
            result = ComparisonResult(match_type=MatchType.BOTH_FAILED.value,
                                     details=f"Both queries failed{sim_note}")
        elif not can_ok:
            sim_note = f" | Query similarity: {q_sim:.1%}" if q_sim is not None else ""
            result = ComparisonResult(
                match_type=MatchType.CANONICAL_FAILED.value,
                generated_count=len(gen_results),
                details=f"Canonical failed: {canonical.get('error', 'N/A')[:50]}{sim_note}"
            )
        else:
            sim_note = f" | Query similarity: {q_sim:.1%}" if q_sim is not None else ""
            result = ComparisonResult(
                match_type=MatchType.GENERATED_FAILED.value,
                canonical_count=len(can_results),
                details=f"Generated failed: {generated.get('error', 'N/A')[:50]}{sim_note}"
            )
        
        # Set query similarity on all results
        result.query_similarity = q_sim
        
        record = {
            "task_id": task_id,
            "difficulty": canonical.get('difficulty', generated.get('difficulty', '')),
            "prompt": canonical.get('prompt', generated.get('prompt', '')),
            "canonical_query": can_q,
            "generated_query": gen_q,
            "canonical_success": can_ok,
            "generated_success": gen_ok,
            **result.to_dict()
        }
        
        q_sim_val = record.get('query_similarity', 0) or 0
        show = (verbose or record.get('is_correct') or
                'partial' in record.get('match_type', '') or
                record.get('f1_score', 0) >= 0.8 or
                (record.get('match_type', '').endswith('_failed') and q_sim_val >= 0.5))
        
        if show:
            symbol = status_symbol(record)
            strict = " [STRICT]" if record.get('is_strict_match') else ""
            sim_tag = f" [QSim:{q_sim_val:.0%}]" if q_sim_val > 0 and not record.get('is_correct') else ""
            print(f"\n[{symbol}] Task {task_id} - {record['match_type']}{strict}{sim_tag}")
            f1 = record.get('f1_score', 0)
            if 0 < f1 < 1:
                print(f"    F1: {f1:.3f} | {record['details']}")
            else:
                print(f"    {record['details']}")
        
        results.append(record)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Text-to-Cypher Comparison Tool")
    parser.add_argument("--canonical", required=True, help="Canonical results (directory or JSON)")
    parser.add_argument("--generated", required=True, help="Generated results (directory or JSON)")
    parser.add_argument("--output", default="comparison_results.json", help="Output file")
    parser.add_argument("--output-format", choices=["json", "csv", "both"], default="both")
    parser.add_argument("--verbose", action="store_true", help="Print all task details")
    args = parser.parse_args()
    
    print(f"Loading canonical from {args.canonical}...")
    canonical = load_results(args.canonical)
    print(f"Loading generated from {args.generated}...")
    generated = load_results(args.generated)
    
    comparisons = process_all(canonical, generated, args.verbose)
    stats = calculate_statistics(comparisons)
    
    print_statistics(stats)
    print_failures(comparisons)
    print_summary(comparisons)
    
    base = args.output.replace('.json', '').replace('.csv', '')
    if args.output_format in ["json", "both"]:
        save_json(comparisons, stats, f"{base}.json")
    if args.output_format in ["csv", "both"]:
        save_csv(comparisons, f"{base}.csv")


if __name__ == "__main__":
    main()
