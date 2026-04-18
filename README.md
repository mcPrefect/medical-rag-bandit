# Contextual Bandit Optimisation of Medical RAG Pipelines

BSc AI & Machine Learning Final Year Project — University of Limerick  
Michael Cronin · Supervisor: Dr. Abdul Razzaq

---

## Overview

An autonomous medical question-answering system that selects its own retrieval strategy per query, validates its own outputs for safety, and updates its own policy using off-policy evaluation — without human intervention in the loop.

Three retrieval arms compete for each query:
- **Fast** — BM25 keyword search
- **Deep** — Semantic search via biomedical Sentence-BERT
- **Graph** — Graph Attention Network over a UMLS subgraph (~1.2M nodes)

A LinUCB contextual bandit observes a 10-dimensional clinical context vector and learns which arm to use. A four-layer safety validator runs independently and can veto any answer. After each evaluation run, IPS-based off-policy evaluation automatically compares LinUCB against Thompson Sampling and saves the better policy's weights.

Evaluated on 1,000 PubMedQA expert-annotated questions with Qwen2.5-14B-Instruct.

---

## Requirements

- Python 3.10+
- CUDA GPU with ~40GB VRAM (for Qwen2.5-14B-Instruct)
- UMLS 2025AB licence (for knowledge graph data)

---

## Setup

```bash
git clone https://github.com/your-username/medical-rag-bandit.git
cd medical-rag-bandit

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Download the scispaCy model:
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

Place PubMedQA data at `data/pubmedqa/ori_pqal.json` and preprocessed UMLS files at `data/umls/`.

---

## Running

**Full evaluation (all strategies, 1,000 examples):**
```bash
python src/evaluation/full_evaluation.py
```

**Quick test run (50 examples, no oracle or ablations):**
```bash
python src/evaluation/full_evaluation.py --n 50 --skip-oracle --skip-ablations
```

**Reward sensitivity analysis:**
```bash
python src/evaluation/reward_sensitivity.py
```

**Adversarial safety test:**
```bash
python src/evaluation/adversarial_safety_test.py
```

**Test suite:**
```bash
pytest tests/test_system.py -v
```

**Demo UI:**
```bash
python src/ui/app.py
```

---

## Project structure

```
src/
├── bandit/
│   ├── linucb.py               # LinUCB contextual bandit
│   └── thompson_sampling.py    # Thompson Sampling baseline
├── retrieval/
│   ├── fast_arm.py             # BM25 keyword retrieval
│   ├── deep_arm.py             # Sentence-BERT dense retrieval
│   └── kg_arm.py               # GNN-based knowledge graph retrieval
├── llm/
│   └── llm_wrapper.py          # Qwen2.5-14B-Instruct interface
├── safety/
│   └── validator.py            # 4-layer safety validator
├── reward/
│   └── reward_function.py      # 4-component weighted reward
├── learning/
│   └── off_policy.py           # IPS estimator + policy comparison
├── graph/
│   ├── gnn_model.py            # MedicalGAT architecture
│   ├── train_gnn.py            # GNN training pipeline
│   ├── compute_node_features.py
│   └── umls_preprocessing.py
├── evaluation/
│   ├── full_evaluation.py      # Main evaluation script
│   ├── reward_sensitivity.py   # Latency weight sensitivity analysis
│   └── adversarial_safety_test.py
└── ui/
    └── app.py                  # Gradio demo interface

configs/
└── config.yaml                 # All hyperparameters
tests/
└── test_system.py              # 62 unit + integration tests
```

---

## Key results

todo
---

## Configuration

All hyperparameters live in `configs/config.yaml`. No code changes needed to adjust reward weights, bandit parameters, retrieval top-k, or safety thresholds.