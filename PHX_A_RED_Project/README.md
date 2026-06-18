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

## Understanding the Class Discovery Report (for Rare Event Detection)

When you call `exp.print_report()` (or the runner finishes) you will see a block that looks like this:

```
Class Discovery Report (vs Random Baseline):
  Classes seen: 87 | Discovered: 12 | Total queries: 14
  Dataset size processed: 500 points
  ...
  caucoo1  | prevalence= 0.20% | seen index=12 | queried index=34 | discovery query=3 | expected random≈500 | lift=166.7x | seen before=1 | (total=1)
  ...
Interpretation:
  • lift > 1.0  = discovered faster than random (good)
  • lift >> 1.0 = strong active discovery of rare classes
  • ...
```

### Key columns / metrics
- **prevalence**: fraction of the stream belonging to this label (after your shuffle + --num-points).
- **seen index / queried index**: ARED's internal processed point index (0-based "algorithm time").
- **discovery query**: the 1-based ordinal of the oracle query that first revealed the class. This is the budget cost.
- **expected random**: under a dumb uniform random labeling policy you would expect to hit a class of this prevalence on query number `1/prevalence`.
- **lift**: `expected_random / discovery_query`. How many times earlier (in labeling budget) A_RED found the class compared to random. This is the primary figure of merit for rare-event work.
- **seen before**: number of real examples of this class that arrived *before* we spent a query on it. They were absorbed into other clusters as "o_pts" (other points). For rare audio events this number tells you how many missed opportunities / how much unlabeled mass went by.

At the bottom of the report (new in the unified code) you also get a quick **Rare classes** aggregate using a default 1% prevalence cutoff:
- how many rare classes were seen vs. discovered
- median lift among the rare ones you *did* discover
- total unlabeled rare instances ("wasted" from a labeling perspective)

### Programmatic access
```python
exp = run_ared("perch", num_points=2000, label_column="scientific_name")
m = exp.get_discovery_metrics(rare_threshold=0.005)   # 0.5%
print(m["median_lift_rare_discovered"])
exp.print_rare_event_summary(rare_threshold=0.01)
```

### Why this matters for rare event detection in audio
Bioacoustic datasets are extremely long-tailed. A few common species dominate the recordings; the interesting events (uncommon species, unusual calls, anomalies) are rare.

Traditional random sampling or even confidence-based active learning wastes most of the labeling budget on the head of the distribution. A_RED's design (anomaly test + "relevance=False forever") + these metrics let you quantify:
- Did my embedding (Perch vs DinoV3 vs spectrogram) + A_RED actually surface the tail classes early?
- How many real rare clips did I let pass unlabeled under a given query budget?
- Is the lift on the rarest classes high enough to justify the pipeline in production?

Use `--label-column scientific_name` (or whatever your species column is) when you care about fine-grained rarity.

## Relationship to Original Code

This package re-uses the *design and logic* from the original frontends but is self-contained under `PHX_A_RED_Project/`. Original files continue to work exactly as before.

See the individual runner files for CLI flags that match historical usage (`--num-points`, `--label-column`, `--target`, `--fast`, etc.).
