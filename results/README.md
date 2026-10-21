# Experiment Results

Commit the small JSON result files produced by the shared training pipeline. Do not commit dataset archives or PyTorch checkpoints.

Expected layout:

```text
results/preliminary/<model>/<group>/seed_42/
├── history.json
└── summary.json
```

After all three groups finish, the model folder also contains `aggregate_preliminary.json`.

Smoke-test outputs use a `seed_42_smoke/` directory and `aggregate_smoke.json`; do not include them in the comparison.

