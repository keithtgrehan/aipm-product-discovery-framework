# AIPM Product Discovery Framework (Experiment 1 Prototype)

Minimal omission-benchmark prototype for fatigue-domain PubMed abstracts.

## Quickstart

```bash
source .venv/bin/activate
python scripts/fetch_pubmed.py
python scripts/build_index.py
python scripts/run_eval.py --llm ollama --model llama3.2:3b
```

Optional Groq backend:

```bash
python scripts/run_eval.py --llm groq --model <groq-model-name>
```
