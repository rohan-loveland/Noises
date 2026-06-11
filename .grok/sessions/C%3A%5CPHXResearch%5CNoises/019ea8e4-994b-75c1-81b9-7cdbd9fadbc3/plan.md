# Plan for True Random Streaming in Spectrogram A_RED

## Current State (from previous plan and implementation)
- SpectrogramDataStream uses pandas CSV (`train_5s_spectrograms.csv`) with `spectrogram_npy_path` column.
- `shuffle=True` + `pd.sample(frac=1, random_state=seed)` + `head(max_samples)` for pseudo-random order.
- `stream_new_data_point` loads from `(identifier)/(chunk).npy` subfolders in `5sSpectrograms_tensors/`.
- Oracle uses `class_name` for label/relevance (currently always False).
- A_RED processes stream sequentially; queries only on anomaly or new relevant class.
- Previous plan focused on high-dim preprocessing (pooling, L2 norm, comp_distance floor), debug, low-query tuning (REL_PROC_VAR=0, relevance=False, low KAPPA).

## User Request Evaluation
- "Create a more true reflection of the dataset, Grabbing truly random data points."
- Dataset: Thousands of .npy files in subfolders `(identifier)/(chunk).npy`, metadata in CSV.
- Goal: True random sampling (not just shuffled CSV order) to better simulate streaming/real-world arrival of spectrograms (avoids CSV bias, ensures uniform coverage of identifiers/chunks).
- Must maintain: Oracle-only labels, A_RED compatibility, no true labels until queried, debug ("Examining spectrogram"), low queries, relevance=False.

## Proposed Approaches (Trade-offs Explored)
1. **CSV + Full Shuffle (Current, Simple)**: Already does `df.sample`. Upgrade to `shuffle=True` + no `head` until streaming, or use `np.random.choice` per step. 
   - Pros: Fast, uses existing metadata.
   - Cons: Not "truly random" if CSV order biases certain identifiers; repeated runs with same seed are reproducible but not dynamic.
   - Trade-off: Good enough for most tests but not "true reflection" of folder structure.

2. **Directory Walk + Random Sample (Recommended)**: Use `list_dir` or `Path.rglob("**/*.npy")` to discover all .npy files, pair with CSV lookup for labels, then randomly sample indices or shuffle list on init. Stream by loading random remaining file.
   - Pros: True random from actual files (reflects subfolder structure), no CSV order bias, can weight by identifier if needed.
   - Cons: Slower init (many files ~100k+), needs to map npy_path back to class_name (use dict from CSV).
   - Matches "They are stored in the 5sSpectrograms_tensors folder, in a form of (identifier)/(chunk).npy."

3. **On-Demand Random (No Preload)**: Maintain set of unused paths, pick `random.choice` each `stream_new_data_point`, remove from set. Lookup label via CSV dict.
   - Pros: Truly random every run, memory efficient (no full list if huge dataset).
   - Cons: O(n) removal cost if not using set of indices; needs index-to-row mapping.
   - Best for "truly random data points" without replacement.

4. **DINOv2 Integration (Future, Out of Scope)**: Embed all spectrograms once for low-dim random stream. Not now (deps, time).

**Chosen Approach**: #3 On-Demand Random (best "true reflection"). 
- Load CSV to dict: npy_rel_path → class_name.
- On init: collect or sample list of all npy_rel_paths (or use glob on first call).
- `stream_new_data_point`: random.choice from remaining, load, remove, print "Examining...", return processed vector.
- Oracle uses dict lookup (no stream_counter dependency for labels).
- Keep max_samples support, add `random.seed` for reproducibility.
- Update main to `shuffle=False` (random now in stream), increase NUM_POINTS_TO_PROCESS if desired.
- No change to A_RED core (still follows papers: anomaly detection, o_pt/add_l_pt, no relevance).

## Critical Files
- `Spectrogram_A_RED/SpectrogramDataStream.py` (main changes to __init__ and stream_new_data_point; add remaining_paths set + label_dict).
- `Spectrogram_A_RED/main_spectrogram.py` (update call, NUM_POINTS_TO_PROCESS=100 for test, note on true random).
- `A_REDimplementation/A_RED/A_RED.py` (no change, but verify with new random stream).
- Test: Run with 100 points, check random files from different identifiers, low queries, relevance=False, algorithm logs.

## Verification Steps
1. Init loads all paths/labels without full preload if possible.
2. Each point is truly random (different runs show different order/files).
3. Debug prints show varied (identifier)/chunk.npy.
4. Oracle queries near-zero, clusters have relevance=False.
5. Matches A_RED (NN, kappa, add_o_pt for non-anomalous, no re-query on discovered classes).
6. Run full test, analyse output for algorithm fidelity (anomalous checks, add_o_pt vs QUERY OCCURRED, cluster summary).

## Risks/Trade-offs
- Large dataset (~100k files): Use lazy glob or limit to CSV rows (current CSV already filters to valid).
- Randomness: Use `random.Random(seed)` for reproducible "true" random.
- Performance: Random choice on list is fine for 1k-10k; for full use set of indices.

This plan creates a more true random stream reflecting the folder structure while preserving all prior work.

**Next**: Implement in SpectrogramDataStream (on-demand random from CSV paths for efficiency). Update main comment. Test and analyse output vs papers. 

(Plan written 2025-04-18. Previous plan was for initial spectrogram adaptation + low-query tuning; this extends it for true random sampling.)