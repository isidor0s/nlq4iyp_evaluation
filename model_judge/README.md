# Model Comparison (LLM Judge)

This tool evaluates **free-text answers** from general LLMs (ChatGPT, Claude, etc.) against the IYP knowledge graph's ground truth. It is separate from the main Cypher evaluation pipeline.

It shows why using validated resources like the IYP knowledge graph matters by checking how closely free‑text LLM answers, based on arbitrary external sources, match the ground‑truth Internet routing facts stored in IYP. 
## Setup
Requires `groq` and `python-dotenv`. Set `GROQ_API_KEY` in your `.env`.

## Usage
Run from the project root:
```bash
python model_judge/model_judge.py
```

## Structure
-   `model_judge.py`: The evaluation script.
-   `questions.json`: Questions and ground truth.
-   `answers/`: Directory for model answer text files.
-   `results.json`: Output of the evaluation.
