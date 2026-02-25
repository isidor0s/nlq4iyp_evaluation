# Data

This directory holds the evaluation inputs and outputs.

## `eval_data/cypherEval/`

The **CypherEval** dataset from the [Pythia paper](https://www.iijlab.net/en/members/romain/pdf/dimitrios_lcn2025.pdf). Each row in the CSV contains:

| Column | Description |
|--------|-------------|
| `Task ID` | Unique identifier (e.g. `1.1`, `1.2`). The `.1` variants use technical phrasing, and the `.2` variants use general/natural phrasing. |
| `Difficulty Level` | `Easy`, `Medium`, or `Hard` with prompt style (`technical prompt` or `general prompt`). |
| `Prompt` | The natural-language question posed to the agent. |
| `Canonical Solution` | The reference Cypher query that produces the correct result. |

`variation-B-filtered.csv` (155 tasks) is the filtered version of CypherEval,doesn't include queries that return em pty results from live database.

## `eval_data/comparisons/`

Comparison reports from previous evaluation runs are stored as paired CSV and JSON files. Each pair captures the per-task scoring output of `compare_results.py`.

| File pattern | Description |
|--------------|-------------|
| `run_v2_2.5_1.csv / .json` | INVESTIGATOR_V2 prompt, Gemini 2.5 Flash |
| `run_v2_2.5_2.csv / .json` | INVESTIGATOR_V2 prompt, Gemini 2.5 Flash |
| `run_v2_temp0.3.csv / .json` | INVESTIGATOR_V2 prompt, temperature 0.3 (2 runs included) |
| `run_v2_temp0.7.csv / .json` | INVESTIGATOR_V2 prompt, temperature 0.7 |
| `run_v2_2.5-pro.csv / .json` | INVESTIGATOR_V2 prompt, Gemini 2.5 Pro |
| `run_pythia_agentic.csv / .json` | PYTHIA_AGENTIC_PROMPT |


### CSV columns

Each comparison CSV contains the following columns:

| Column | Description |
|--------|-------------|
| `task_id` | Matches the CypherEval task ID. |
| `difficulty` | Task difficulty level. |
| `prompt` | Original question posed. |
| `canonical_success` / `generated_success` | Indicates if each query executed without errors. |
| `is_correct` | Overall correctness flag. |
| `is_strict_match` | Exact structural match without semantic relaxation. |
| `match_type` | Classification label (refer to the Match types table). |
| `execution_accuracy` | 1.0 if correct, 0.0 otherwise. |
| `f1_score`, `precision`, `recall` | Set-overlap metrics. |
| `jaccard_similarity` | Intersection-over-union metric. |
| `query_similarity` | Textual similarity between generated and canonical Cypher queries. |
| `canonical_count` / `generated_count` | Row counts returned. |
| `true_positives`, `false_positives`, `false_negatives` | Confusion matrix values. |
| `details` | Human-readable explanation of the result. |

### Match types

| Label | Meaning |
|-------|---------|
| `exact_match` | Identical result sets. |
| `semantic_equivalent` | Same answers but different structure (e.g., extra columns, reordered rows). |
| `semantic_equivalent_projection` | Same identifiers but different column projection. |
| `semantic_equivalent_node_property` | One returns full nodes while the other returns extracted properties. |
| `semantic_equivalent_answer` | Core answer values match despite having different shapes. |
| `semantic_equivalent_limit` | The generated query adds a `LIMIT`. Every row is a valid subset of the canonical query. |
| `numeric_exact_match` | Single numeric value matches exactly. |
| `numeric_near_match` | Numeric values are within tolerance. |
| `superset` | Generated query is a strict superset of the canonical query. |
| `partial_overlap` / `low_overlap` | Some shared results exist but are insufficient for equivalence. |
| `no_overlap` | No common results are returned. |
| `generated_failed` | Generated query encountered an error or timed out. |
| `generated_empty` | Generated query returned no rows. |
