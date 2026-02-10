# Autonomous Medical RAG - Progress Update as I go

## Still in MVP Phase so some features are hardcoded and currentl only working with PUBMEDQA data
## Completed 

**Two-Arm Bandit System:**
- Fast arm (BM25) and Deep arm (semantic retrieval)
- LinUCB bandit with latency-aware rewards
- Baseline evaluation: 44.6% accuracy, 26% efficiency improvement over best baseline

**Safety Validator:**
- 4-layer validation (confidence, evidence, contraindications, sanity)
- 100% sensitivity on edge cases (caught all dangerous recommendations)
- Automatic abstention when unsafe

**Infrastructure:**
- YAML configuration system
- Logging throughout pipeline
- End-to-end autonomous loop working

## Next Steps
- Attempt GNN/UMLS knowledge graph arm 

-  Off-policy learning (IPS estimator)

- Full evaluation

- UI


## Running
```bash
python src/main.py                  # Main pipeline
python tests/test_validator.py      # Edge case tests
python src/evaluate_baselines.py    # Baseline comparison
```

Edit `configs/config.yaml` to change settings (n_examples, model, thresholds, etc.)
