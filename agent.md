# LAVA Agent Documentation

## Project Overview

LAVA (Latent Action Video Analyzer) is a streaming video understanding framework built on V-JEPA 2, designed for real-time video comprehension with autonomous deferral capability.

## Architecture

```
Input Video Frames
    │
    ▼
┌──────────────────────────────────────┐
│  V-JEPA 2 Encoder (frozen)            │
│  Frame → z_t (2048-dim latent)        │
└──────────────────────────────────────┘
    │
    ├─→ LAE (Linear 2048→256, trainable)
    │       │
    │       ▼
    │   s_t (256-dim state)
    │       │
    │       ▼
    │   ┌─────────────────┐
    │   │  WorldStateGRU  │ ← temporal fusion → h_t
    │   └─────────────────┘
    │           │
    │           ├─→ TemporalClassifier
    │           └─→ EvidenceScorer (confidence C)
    │
    └─→ V-JEPA 2 Predictor (frozen)
            │
            ▼
        Rollout N steps → z_{t+1:t+N}
```

## Query Routing

| Tag | Path | Context Source |
|-----|------|----------------|
| PAST / NOW / THV | Retrieval | Memory M top-k slots |
| FP (Future Prediction) | Prediction | V-JEPA rollout N=5 |
| IA (Intention Analysis) | Prediction | V-JEPA rollout N=10 |

## Training

**Phase 3 (active)** — joint end-to-end training:

```
L_total = L_QA + L_tag + L_defer
```

- `L_QA`: answer quality loss (cross-entropy)
- `L_tag`: temporal tag classification loss
- `L_defer`: deferral decision loss (binary)

## File Structure

```
lava/
  config.py              # hyperparameters
  train.py               # entry point (--mode train/eval)
  models/
    vjepa_wrapper.py     # V-JEPA 2 encode + rollout
    components.py        # LAE, GRU, TemporalClassifier, EvidenceScorer
    memory.py            # ActionStateMemory with surprise gate
    icd.py               # Qwen2-VL-7B + LoRA decoder
    lava.py              # main model (step() + query())
  data/dataset.py        # streaming pair construction
  training/trainer.py    # Phase 3 training loop
  inference/pipeline.py # streaming inference + benchmark evaluation
```

## Key Design Decisions

1. **No separate Phase 2**: LAE trained jointly in Phase 3 — contrastive pre-training not required
2. **V-JEPA encoder handles past, V-JEPA predictor handles future** — clean separation of temporal reasoning
3. **Predictor rollout provides visual grounding for FP/IA** — not language-prior guessing

## Commands

```bash
# Train (8 GPUs)
torchrun --nproc_per_node=8 train.py --mode train

# Resume
torchrun --nproc_per_node=8 train.py --mode train --resume ./checkpoints/lava_phase3/step_10000

# Evaluate
python train.py --mode eval --ckpt ./checkpoints/lava_phase3/step_30000 --bench all
```

## External Dependencies

- V-JEPA 2 encoder + predictor (Meta, 1M hour video pretrained)
- Qwen2-VL-7B (Alibaba, instruction-tuned)
- OVBench / OVO-Bench datasets

## Notes

- V-JEPA 2 predictor attribute name may vary by HF version — check `raw.predictor` vs `raw.predictor_model`
- Dataset JSON requires `answer_timestamp` field — write conversion script for OVBench/OVO-Bench (~50 lines)
- ICD label masking in trainer.py uses rough token counting — consider switching to `"Answer:"` token position for precision
