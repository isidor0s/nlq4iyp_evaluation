# NLQ4IYP: Evaluation & CLI Agent


This repository offers a CLI agent and evaluation tools for the [NLQ4IYP project](https://github.com/isidor0s/NLQ_4_IYP), complementing the [web app](https://nlq4iyp.streamlit.app/):

**CypherEval datasets** ([codeberg.org/dimitrios/CypherEval](https://codeberg.org/dimitrios/CypherEval)) are used for benchmarking Cypher query generation. This benchmark currently evaluates only the Gemini agent version.

- CLI gemini agent version.
- Evaluate Cypher query generation automatically.
- Assess free-text answers with an LLM-based judge.


## Project Structure


## CypherEval Dataset

This repository uses the CypherEval datasets for benchmarking NL-to-Cypher models. See the [CypherEval project](https://codeberg.org/dimitrios/CypherEval) for more details about the dataset.

**Note:** This benchmark currently only evaluates the Gemini agent.

---

This repository is organized into three main components:

-   **[`main_scripts/`](main_scripts/README.md)**: The core **Text-to-Cypher Agent** and the **CypherEval** evaluation pipeline. This is the main part of the project.
-   **[`model_judge/`](model_judge/README.md)**: A separate **LLM Judge** tool for evaluating free-text answers from general models (like ChatGPT) against IYP ground truth.
-   **`data/`**: Shared data, including the CypherEval dataset and evaluation results.

## Quick Start
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure environment**:
    Copy `.env.example` to `.env` and add your API keys (Google GenAI, Neo4j, Groq).

3.  **Run the Agent** (Text-to-Cypher):
    ```bash
    python main_scripts/agent/gemini_agent.py
    ```

4.  **Run the Model Judge** (Text Evaluation):
    ```bash
    python model_judge/model_judge.py
    ```

For detailed instructions, please refer to the README files in each subdirectory.
