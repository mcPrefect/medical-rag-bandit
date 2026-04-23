# Contextual Bandit Optimisation of Medical RAG Pipelines

BSc AI & Machine Learning Final Year Project — University of Limerick  
Michael Cronin · Supervisor: Dr. Abdul Razzaq

---

## What this is

An autonomous medical question-answering system that selects its own retrieval strategy per query, validates its own outputs for safety, and updates its own policy using off-policy evaluation — without human intervention in the loop.

Three retrieval arms compete for each query:
- **Fast** — BM25 keyword search
- **Deep** — Semantic search via biomedical Sentence-BERT
- **Graph** — Graph Attention Network over a UMLS subgraph (~1.2M nodes)

A LinUCB contextual bandit observes a 10-dimensional clinical context vector and learns which arm to use. A four-layer safety validator runs independently and can veto any answer. After each evaluation run, IPS-based off-policy evaluation automatically compares LinUCB against Thompson Sampling and saves the better policy's weights.

Evaluated on PubMedQA (1,000 expert-annotated questions) and MedQA-USMLE (1,000 questions, MedRAG textbook corpus).

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

---

## Data and Models

The following are required but not included due to size or licensing:

**UMLS Knowledge Graph** (free academic licence required):
- Register at: https://uts.nlm.nih.gov/uts/signup-login
- Then run: `python src/graph/umls_preprocessing.py`
- Then run: `python src/graph/compute_node_features.py`
- Place outputs at `data/umls/`

**Trained GNN:**
- Train from scratch: `python src/graph/train_gnn.py`
- Or place `gnn_model_best.pt` at `models/`

**PubMedQA:**
- Place `ori_pqal.json` at `data/pubmedqa/`
- Available at: https://pubmedqa.github.io

**MedRAG Textbook Corpus** (for MedQA evaluation only):
- Built automatically: `python src/evaluation/build_medrag_index.py`

Without UMLS and the GNN, the Graph arm falls back to returning the first-k context sentences. The Fast and Deep arms work fully without any additional data.

---

## Running

**Full PubMedQA evaluation:**
```bash
python src/evaluation/full_evaluation.py
```

**Quick test (50 examples, no oracle or ablations):**
```bash
python src/evaluation/full_evaluation.py --n 50 --skip-oracle --skip-ablations
```

**MedQA evaluation (requires MedRAG index):**
```bash
python src/evaluation/build_medrag_index.py   # run once
python src/evaluation/medqa_evaluation.py --n 1000
```

**Reward sensitivity analysis:**
```bash
python src/evaluation/reward_sensitivity.py
```

**Adversarial safety test (no GPU required):**
```bash
python src/evaluation/adversarial_safety_test.py
```

**Test suite (no GPU required):**
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
│   ├── linucb.py                  # LinUCB contextual bandit
│   └── thompson_sampling.py       # Thompson Sampling baseline
├── retrieval/
│   ├── fast_arm.py                # BM25 keyword retrieval
│   ├── deep_arm.py                # Sentence-BERT dense retrieval
│   └── kg_arm.py                  # GNN-based knowledge graph retrieval
├── llm/
│   └── llm_wrapper.py             # Qwen2.5-14B-Instruct interface
├── safety/
│   └── validator.py               # 4-layer safety validator
├── reward/
│   └── reward_function.py         # 4-component weighted reward
├── learning/
│   └── off_policy.py              # IPS estimator + policy comparison
├── graph/
│   ├── gnn_model.py               # MedicalGAT architecture
│   ├── train_gnn.py               # GNN training pipeline
│   ├── compute_node_features.py   # PubMedBERT node embeddings
│   └── umls_preprocessing.py      # UMLS subgraph extraction
├── evaluation/
│   ├── full_evaluation.py         # Main PubMedQA evaluation
│   ├── medqa_evaluation.py        # MedQA-USMLE evaluation
│   ├── reward_sensitivity.py      # Latency weight sensitivity analysis
│   ├── adversarial_safety_test.py # Safety validator stress test
│   └── build_medrag_index.py      # MedRAG FAISS index builder
└── ui/
    └── app.py                     # Gradio demo interface

configs/
└── config.yaml                    # All hyperparameters

tests/
└── test_system.py                 # 74 unit and integration tests

```

---

## Configuration

All hyperparameters are in `configs/config.yaml`. No code changes needed to adjust reward weights, bandit parameters, retrieval top-k, safety thresholds, or model paths.