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

## Training Datasets

### Recommended Training Data Combinations

#### Pretraining / Large-Scale
| Dataset | Description | Link |
|---------|-------------|------|
| **WebVid-10M** | 10M captioned short videos — largest video-text dataset | [GitHub](https://github.com/m-bain/webvid) |
| **HowTo100M** | 1M instructional videos with captions, long temporal span | [GitHub](https://github.com/LandyGuo/Download_HowTo100M) |

#### Downstream / LVU Task Training
| Dataset | Description | Link |
|---------|-------------|------|
| **MLVU** | CVPR 2025 — Multi-Task Long Video Understanding, 9 tasks, 3min–2hr videos | [HuggingFace](https://huggingface.co/datasets/MLVU/MVLU) |
| **YouCook2** | 89 recipe categories, temporal annotations for video grounding | widely used in video grounding literature |
| **ActivityNet-QA** | QA pairs on ActivityNet videos | [GitHub](https://github.com/MILVLG/activitynet-qa) |

#### For Streaming / Online特性
| Dataset | Description | Link |
|---------|-------------|------|
| **OVO-Bench** | Online Video Understanding benchmark — streaming video comprehension | [GitHub](https://github.com/JoeLeelyf/OVO-Bench) |
| **OVBench** | Streaming video understanding evaluation | benchmark for LAVA's deferral mechanism |

### Data Preparation Priority
1. **Pretraining**: WebVid-10M or HowTo100M — establish video-text alignment
2. **LVU downstream**: MLVU training set — multi-task LVU objectives
3. **Temporal reasoning**: YouCook2 — rich temporal ground truth for FP/IA paths
4. **Streaming evaluation**: OVO-Bench / OVBench — validate deferral mechanism

## Related Work: WeaveTime (CVPR 2026)

**Paper**: [WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs](https://arxiv.org/abs/2602.22142)

WeaveTime addresses **Time-Agnosticism** in Video-LLMs — treating videos as unordered bags of evidence rather than causally ordered sequences. It introduces:

- **Temporal Reconstruction objective** (Streaming Order Perception) — lightweight fine-tuning to instill order-aware representations
- **Past-Current Dynamic Focus Cache** — uncertainty-triggered, coarse-to-fine retrieval that expands history only when needed

### WeaveTime Training Datasets (LoRA fine-tuning)
| Dataset | Description |
|---------|-------------|
| **LLaVA-Video-178K** | lmms-lab/LLaVA-Video-178K |
| **VCR-Bench** | VLM-Reasoning/VCR-Bench (visual commonsense reasoning) |
| **Ego4D-MC** | Becomebright/QAEgo4D-MC-test (egocentric video QA) |

Evaluated on: StreamingBench, **OVOBench**, **MLVU**, ActivityNet-QA, EgoSchema, ETBench, CGbench, Eventhal, VidHalluc

### WeaveTime vs LAVA

| | WeaveTime | LAVA |
|--|-----------|------|
| Core mechanism | Temporal Reconstruction + Dynamic Focus Cache | V-JEPA 2 predictor rollout + deferral |
| Training data | 178K + VCR + Ego4D | MLVU + WebVid-10M (proposed) |
| Goal | Memory efficiency in streaming Video-LLMs | LVU + future prediction + autonomous deferral |
| Base models | LLaVA-OneVision / Qwen2-VL | V-JEPA 2 + Qwen2-VL-7B |
| Orthogonal? | ✓ — addresses memory/order | ✓ — addresses future grounding + deferral |

**Conclusion**: WeaveTime and LAVA solve different dimensions of streaming video understanding and are complementary. WeaveTime excels at memory-efficient streaming inference; LAVA excels at LVU tasks with V-JEPA 2's future imagination for FP/IA queries.

## Key Design Decisions

1. **No separate Phase 2**: LAE trained jointly in Phase 3 — contrastive pre-training not required
2. **V-JEPA encoder handles past, V-JEPA predictor handles future** — clean separation of temporal reasoning
3. **Predictor rollout provides visual grounding for FP/IA** — not language-prior guessing
4. **Focus: LVU (Long Video Understanding), not VLA** — WAM's future imagination complements LVU's temporal reasoning

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
- MLVU / OVBench / OVO-Bench datasets
- WebVid-10M or HowTo100M (for pretraining)

## Notes

- V-JEPA 2 predictor attribute name may vary by HF version — check `raw.predictor` vs `raw.predictor_model`
- Dataset JSON requires `answer_timestamp` field — write conversion script for OVBench/OVO-Bench (~50 lines)
- ICD label masking in trainer.py uses rough token counting — consider switching to `"Answer:"` token position for precision
- FP queries use N=5 (~2.5s lookahead); IA queries use N=10 (~5s lookahead) — validate on dev set
