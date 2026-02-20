# IYP Text-to-Cypher Evaluation

This project and Natural Language Querying agent for the Internet Yellow Pages (IYP) and provides tools to evaluate its performance.

## Project Structure

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
