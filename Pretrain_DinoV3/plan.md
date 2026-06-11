# Plan for Pretraining Custom DINO Model on Spectrogram Dataset

## Context
- **Current State**: We have successfully implemented:
  - Hidden `DiscoveryTracker` in `Spectrogram_A_RED/SpectrogramDataStream.py:88-167` (first_seen vs first_queried per-class metrics, JSON report, invisible to ARED algorithm).
  - `Spectrogram_DinoV3/` folder with `Dinov3DataStream.py` (uses HF `facebook/dinov2-small` for 384-dim semantic embeddings from existing `.npy` Mel-spectrograms via PIL.Image.fromarray + mean-pool last_hidden_state; reuses tracker, CSV loading, mmap).
  - `SpectrogramDinov3.py` runner (mirrors `main_spectrogram.py`, tested with ~26% query rate, discovery_report.json).
  - `Pretrain_DinoV3/` with `Dinov3PretrainDataset.py` (2-view SSL dataset, RGB conversion for ViT, mmap .npy) and `pretrain_dinov3.py` (timm `vit_small_patch16_224`, teacher/student momentum=0.996, simplified DINO loss, ImageNet transforms, CLI, tested with --epochs=1 --subset=0.005 producing `dinov3_pretrained_epoch1.pth` with loss ~2.57).
- **Goal**: Pretrain our own DINO (self-supervised ViT) model **on our specific dataset** (Mel-spectrograms from `5sSpectrograms_tensors/train_5s_spectrograms.csv` + `*.npy` files) to learn better domain-specific semantic embeddings. This should improve A_RED accuracy/speed over generic DINOv2 (better clustering of bird/insect/noise classes).
- **Constraints** (from history):
  - Use **existing .npy files** (no regeneration via create_spectrogram_tensors.py).
  - Isolated in `Pretrain_DinoV3/` folder (separate from A_RED and DinoV3 inference).
  - Hidden metrics only for testing (no algorithm access).
  - Reuse patterns from `MLBird.py:31` (dataset), `SpectrogramDataStream.py:27-66` (loading), `pretrain_dinov3.py` (training loop).
  - No extra features, error handling, abstractions, or docstrings beyond task.
  - Verify with runs/tests before claiming success.
  - Respect plan mode (read-only exploration, write plan.md, exit only after user review/approval).

## Approach
1. **Data Pipeline**: Treat Mel-spectrograms as images (grayscale→RGB via `Image.fromarray(...).convert('RGB')` as fixed in `Dinov3PretrainDataset.py:42-43`). Use 2-view augmentations (global/local crops) for DINO SSL. Full dataset or subset via pandas/CLI.
2. **Model**: timm `vit_small_patch16_224` (384-dim output, matches DINOv2-small). Student/teacher setup with momentum update.
3. **Loss & Training**: Simplified DINO loss (`dino_loss` in pretrain script: sharpened teacher softmax + CE). AdamW, AMP/GradScaler, multi-crop views. Checkpoints every 10 epochs + final.
4. **Integration**: After pretraining, update `Dinov3DataStream.py` to optionally load custom `dinov3_pretrained.pth` (extract student/teacher or use as feature extractor backbone) instead of HF model. Test downstream in `SpectrogramDinov3.py` for improved query rate / discovery metrics.
5. **Verification**: Run short pretrain (`--epochs=3 --subset=0.01`), check loss decrease, checkpoint creation. Then integrate + run full A_RED pipeline. Compare reports (e.g. fewer queries on Insecta/Amphibia).
6. **Why This Works**: Spectrograms-as-RGB viable (channel fix applied). Low-dim embeddings avoid curse-of-dimensionality in A_RED BallTree. Self-supervised on our acoustic data captures domain semantics better than generic ImageNet-pretrained DINOv2.

## Key Files (with line references from current state)
- **Pretrain_DinoV3/pretrain_dinov3.py:1-168** (main script, transforms:43-63, dataset:100-101, model:105-109, loop:123-148, checkpoints:152-159).
- **Pretrain_DinoV3/Dinov3PretrainDataset.py:1-63** (loading:37-38 mmap, RGB:42-43, 2-views:48-51).
- **Pretrain_DinoV3/__init__.py** (docstring only).
- **Spectrogram_DinoV3/Dinov3DataStream.py:57-63** (current HF load; to be updated for custom weights ~lines 56-70), `stream_new_data_point:107-117` (embedding extraction).
- **Spectrogram_DinoV3/SpectrogramDinov3.py:36-50** (runner, report).
- Reference: `MLBird.py:31-54` (dataset base), `create_spectrogram_tensors.py` (Mel details, not modified), `Spectrogram_A_RED/SpectrogramDataStream.py:80` (label helper).

## Implementation Steps
1. [ ] **Finalize Plan**: Write this `plan.md` with full details (done via tool). Explore codebase if needed (grep for timm/transforms, read MLBird.py sections).
2. [ ] **Run/Verify Pretrain**: Execute `python Pretrain_DinoV3/pretrain_dinov3.py --epochs=3 --batch-size=16 --subset=0.02` (small scale for Windows/CPU+CUDA). Confirm loss drops, checkpoints saved (`dinov3_pretrained_epoch*.pth`).
3. [ ] **Update DataStream for Custom Model**: Edit `Dinov3DataStream.py` to add `load_pretrained` option, replace HF with custom ViT + state_dict load (use student or averaged weights). Keep fallback.
4. [ ] **Update Runner**: Minor edits to `SpectrogramDinov3.py` to use custom model path, print "Custom Pretrained DINO" in report.
5. [ ] **Test Full Pipeline**: Run `SpectrogramDinov3.py` with custom model. Generate new `dinov3_discovery_report.json`. Compare query % and per-class metrics vs HF DINOv2 run.
6. [ ] **Verification & Metrics**: Use `todo_write` to track. Run tests via terminal. Check embedding dim=384, valid A_RED clusters, discovery tracker hidden. Save final model as `dinov3_pretrained.pth`.
7. [ ] **Cleanup**: No new files beyond plan if possible. Respect "minimum complexity" — edit existing where feasible.

## Risks & Mitigations
- CUDA OOM on full data: Use --subset, small batch, CPU fallback.
- ViT input mismatch: Already fixed (224x224, 3-channel RGB).
- Loss not converging: Monitor with tqdm; use official DINO temp/sinkhorn if needed (but keep simplified per current script).
- Windows paths: Use Pathlib (already in files).

## Success Criteria
- Custom model pretrained on **our dataset** (not just HF weights).
- Checkpoint loads successfully in Dinov3DataStream.
- A_RED runs with embeddings, produces discovery report showing improved (lower) query rates on rare classes.
- All tests pass; no breakage to prior DiscoveryTracker or Spectrogram_A_RED.
- Plan approved by user before any further code changes (per plan mode).

**Next Action**: Exit plan mode for user review/approval of this plan. Then implement steps above using search_replace for edits, run_terminal_command for training/tests, todo_write for progress.

Plan created: [Pretrain_DinoV3/plan.md](/Pretrain_DinoV3/plan.md)
