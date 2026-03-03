# AIPM Bootcamp - Day 2: AI-Powered Discovery Framework

Do not make this material public.

## Goal
Transform Day 1 business assumptions into validated market insights using AI-powered discovery in under 40 minutes of focused work.

## Framework Overview
Total time: 40 minutes of individual AI work plus team synthesis.

- Step 0: Product Context Calibration (5 min)
- Step 1: Customer Research and Personas (10 min)
- Step 2: Pain Point Pattern Analysis (10 min)
- Step 3: AI-Powered Ideation (8 min)
- Step 4: Market Validation (5 min)
- Step 5: Stakeholder-Ready Outputs (3 min)
- Bonus: Stress-Test Your Discovery

## How to Use
1. Individual work: Complete Steps 0-5 using ChatGPT/Claude.
2. Team synthesis: Compare discoveries and make decisions.
3. Next steps: Use insights for Day 3 prototyping.

## Experiment 1 Prototype (Omission Benchmark)
This repo now also includes a minimal Python prototype for Experiment 1 using PubMed fatigue-domain abstracts.

Quickstart:

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

Key prototype paths:

- `benchmarks/`
- `scripts/`
- `src/structured_rag/`
- `data/corpus/`
- `data/index/`
- `reports/`
