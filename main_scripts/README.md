# Main Scripts

This directory contains the Natural Language Querying (NlQ4IYP) for IYP agent (cli version) and the CypherEval evaluation pipeline.

## Structure

-   **`agent/`**: The ReAct agent implementation.
    -   `gemini_agent.py`: Interactive CLI to chat with the agent.
    -   `agent_core.py`: Main agent logic (LLM setup, tools).
    -   `agent_utils.py`: Shared Utilities for the evaluation.
    -   `prompts.py`: Prompt templates 
-   **`eval/`**: Evaluation tools.
    -   `batch_evaluate.py`: Batch evaluation script.
    -   `compare_results.py`: Script to score generated results against canonicals.
    -   `loocv.py`: Leave-One-Out Cross-Validation logic.
-   **`system-prompt`**: The prompt file defining the Neo4j schema and rules.

## Usage

### Interactive Agent
To chat with the agent interactively:
```bash
python main_scripts/agent/gemini_agent.py
```

### Batch Evaluation
To run the full evaluation benchmark with Leave-One-Out Cross-Validation:
```bash
python main_scripts/eval/batch_evaluate.py --csv data/eval_data/cypherEval/variation-B-filtered.csv --loocv --results-folder data/eval_data/results/demo_run --output data/eval_data/results/demo_run.csv
```

**Key options:**
- `--loocv`: Enable leave-one-out cross-validation 
- `--results-folder`: Directory to store individual JSON last query result
- `--output`: CSV format containing all extracted queries
- `--temperature`: Model temperature (default: 0.0)
- `--timeout`: Agent timeout in seconds (default: 300)

### Scoring Results
To compare generated results with canonicals:
```bash
python main_scripts/eval/compare_results.py --canonical data/eval_data/results/canonical --generated data/eval_data/results/demo --output data/eval_data/results/comparison_results.json --output-format both
```

**Key options:**
- `--canonical`: Directory or file containing canonical/expected results
- `--generated`: Directory or file containing generated query results
- `--output`: Output file for comparison results
- `--output-format`: Output format (json, csv, or both; default: both)
- `--verbose`: Print detailed results for each task
