# PHX_A_RED_Project

Refactored, reusable implementation of the ARED (Anomaly + Relevance + Discovery) pipeline for bioacoustic data.

This package consolidates the duplicated logic that previously lived across:
- `Perch/`
- `Spectrogram_A_RED/`
- `Spectrogram_DinoV3/`
- `Pretrain_DinoV3/`

**No changes were made to the original code.** This is a clean re-implementation following proper object-oriented design.

## Key Abstractions

- **Data Streams** (`data/`): `BaseDataStream` + concrete implementations for spectrograms, precomputed embeddings, Perch (live or precomp), and DinoV3.
- **Oracle** (`data/oracle.py`): Single `NoRelevanceOracle` — always returns `relevance=False` (the policy that produces near-zero queries after initial points).
- **Feature Extractors** (`features/`): `PerchEmbedder`, `DinoV3Extractor`.
- **Utilities** (`utils/`): `DiscoveryTracker`, `ClassDiscoveryCounter`, `ShiftingKappaController`.
- **Experiment Runner** (`experiment.py`): `AREDExperiment` — the unified high-level interface. Handles wiring, first point, loop, controller integration, and reporting.
- **Pretraining** (`pretraining/`): Refactored DINOv3 SSL training to produce domain-specific spectrogram embedders.

All streams implement the duck-typed interface expected by the untouched core ARED:
- `stream_new_data_point()`
- `get_remaining_num_points()`
- `get_true_label_for_idx(idx)`

## Quick Start (from project root)

```bash
# Basic spectrogram run (raw pooled features)
python -m PHX_A_RED_Project.runners.spectrogram --num-points 200

# Perch embeddings (uses precomputed if available, else live)
python -m PHX_A_RED_Project.runners.perch --num-points 200 --label-column class_name

# DinoV3 embeddings
python -m PHX_A_RED_Project.runners.dinov3 --num-points 200

# With live shifting-kappa controller (species level)
python -m PHX_A_RED_Project.runners.with_shifting_kappa \
    --num-points 1000 --label-column scientific_name --target 8
```

## Adding a New Frontend

1. Create (or subclass) a `XXXDataStream(BaseDataStream)`.
2. Implement vector loading (either precomputed fast path or live extractor).
3. (Optional) Add a thin extractor in `features/`.
4. Wire it in a new runner or directly via `AREDExperiment`.

Example skeleton:

```python
from PHX_A_RED_Project.data.base_stream import BaseDataStream
from PHX_A_RED_Project.data.oracle import NoRelevanceOracle
from PHX_A_RED_Project.experiment import AREDExperiment

stream = MyNewStream(max_samples=500, shuffle=True, seed=42)
oracle = NoRelevanceOracle(stream)
exp = AREDExperiment(stream, oracle, kappa=1.5, data_window_size=250)
exp.run(num_points=500)
exp.print_report()
```

## Important Behavioral Notes

- Oracle policy is deliberately `relevance=False` for every class. This implements "query only on true anomaly" and "do not re-query discovered classes".
- All vectors fed to ARED are L2-normalized unit vectors.
- Discovery counters and the kappa controller are optional and compose cleanly.
- The core ARED implementation lives in `A_REDimplementation/A_RED/` and is imported (never modified).

## Relationship to Original Code

This package re-uses the *design and logic* from the original frontends but is self-contained under `PHX_A_RED_Project/`. Original files continue to work exactly as before.

See the individual runner files for CLI flags that match historical usage (`--num-points`, `--label-column`, `--target`, `--fast`, etc.).
