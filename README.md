# NLQ4IYP: Evaluation & CLI Agent


This repository offers a CLI agent and evaluation tools for the [NLQ4IYP project](https://github.com/isidor0s/NLQ_4_IYP), complementing the [web app](https://nlq4iyp.streamlit.app/):


- CLI gemini agent version.
- Evaluate Cypher query generation automatically.
- Assess free-text answers with an LLM-based judge.


## CypherEval Dataset

This repository uses the CypherEval datasets for benchmarking NL-to-Cypher models. See the [CypherEval](https://codeberg.org/dimitrios/CypherEval) for more details about the dataset.

**Note:** This benchmark currently only evaluates the Gemini agent.

---

## Project Structure

This repository is organized into three main components:

-   **[`main_scripts/`](main_scripts/README.md)**: The core **Text-to-Cypher Agent** and the **CypherEval** evaluation pipeline. This is the main part of the project.
-   **[`model_judge/`](model_judge/README.md)**: A separate **LLM Judge** tool for evaluating free-text answers from general models (like ChatGPT) against IYP ground truth.
-   **`data/`**: Shared data, including the CypherEval dataset and evaluation results.

## Quick Start


### Prerequisites

- **Python version:** Python 3.8 or higher
- **Google Gemini API key:** Get from [Google AI Studio](https://aistudio.google.com/api-keys)
- **Groq API key (for Model Judge):** Get from [Groq Console](https://console.groq.com/keys)

---

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure environment**:
    Copy `.env.example` to `.env` and add your API keys (Google Ai, Neo4j Server, Groq).
    ```
    # Google Gemini API
    #GOOGLE_GENAI_API_KEY=your_google_api_key_here

    # Neo4j Database Connection
    # Default Live server/ Change for local IYP DB
    NEO4J_URI=bolt://iyp-bolt.ihr.live:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=
    NEO4J_DATABASE=neo4j

    # For Model Judge
    GROQ_API_KEY=YOUR_GROQ_API_KEY_HERE

3.  **Run the Agent** (Text-to-Cypher):
    ```bash
    python main_scripts/agent/gemini_agent.py
    ```

4.  **Run the Model Judge** (Text Evaluation):
    ```bash
    python model_judge/model_judge.py
    ```

For detailed instructions, please refer to the README files in each subdirectory.
