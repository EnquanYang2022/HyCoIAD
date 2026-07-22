# HyCo-IAD

**HyCo-IAD: Hyperbolic Collaborative Reinforcement Fine-Tuning for Multimodal Industrial Anomaly Detection**

Enquan Yang, Zhengqin Xu, Xilin Xu, Peng Xing, Yuanwei Ma, Zechao Li, and Dan Zeng

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Backbone-Qwen2.5--VL--3B-7E57C2.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
[![Benchmark](https://img.shields.io/badge/Benchmark-MMAD-green.svg)](https://github.com/jam-cc/MMAD)

> The trained model checkpoint is now available. The paper and complete environment specifications will be released upon publication.

## News

* **[2026-07-22]** The trained HyCo-IAD checkpoint is available on [Google Drive].
* **[2026-07-21]** The complete code will be made publicly available upon publication of the paper.

## Model Checkpoint

The trained HyCo-IAD checkpoint based on Qwen2.5-VL-3B-Instruct can be downloaded from:

* [HyCo-IAD Checkpoint (Google Drive)](https://drive.google.com/file/d/12hCT3gkhJ3FkGMwzRuYI7lCQzBKZTTgS/view?usp=sharing)

## Overview

Multimodal Large Language Models (MLLMs) enable industrial anomaly detection systems to move beyond numerical anomaly scores and provide semantic interpretation and diagnostic reasoning. However, standard supervised fine-tuning can overfit severely when industrial anomaly samples are scarce. Reinforcement fine-tuning offers a promising alternative, but its application to industrial anomaly detection is limited by two structural challenges: the mismatch between Euclidean representations and hierarchical defect semantics, and the attenuation of sparse reward signals through deep networks.

HyCo-IAD is a reinforcement fine-tuning framework that addresses these challenges through two complementary components:

- **Low-rank Hyperbolic Adaptation (LHA):** projects visual representations into a hyperbolic space using Lorentz transformations, enabling the model to capture fine-grained and hierarchical defect semantics.
- **Cross-Layer Collaborative Optimization (CCO):** models layer-wise adaptation parameters as a high-order tensor and builds direct cross-layer communication, improving reward-signal propagation to earlier visual layers.

HyCo-IAD is built on Qwen2.5-VL-3B and optimized with Group Relative Policy Optimization (GRPO). On the MMAD benchmark under the standard one-shot protocol, HyCo-IAD achieves an average accuracy of **79.09%**.

## Highlights

- A hyperbolic reinforcement fine-tuning framework tailored to multimodal industrial anomaly detection.
- Parameter-efficient visual adaptation through learnable low-rank hyperbolic transformations.
- Cross-layer collaborative optimization for more effective sparse-reward propagation.
- Data-efficient training using **319 images and 1,164 question-answer pairs**.
- State-of-the-art average accuracy on the MMAD benchmark in the one-shot setting.

## Repository Structure

```text
HyCoIAD/
├── data_config/
│   └── IAD.yaml                         # Training-data configuration
├── eval/
│   ├── evaluate_batch_mmad_choice_mymodel.py
│   ├── qwen2_5_vl_base_query.py
│   ├── qwen2_5vl_module_HpyCo.py
│   ├── helper/                          # MMAD metric utilities
│   └── GPT4/                            # API-based baseline evaluation
├── open_r1_multimodal/
│   └── open_r1/                         # GRPO trainer and HyCo-IAD model modules
├── run_scripts/
│   ├── run_grpo.sh                      # Distributed training entry point
│   ├── accelerate_config/
│   └── deepspeed_config/
├── train_data/
│   └── train_data_grpo.jsonl            # GRPO training annotations
└── grpo.py                              # Training and reward functions
```

## Environment Setup

The experiments reported in the paper use four NVIDIA RTX 4090 GPUs, DeepSpeed, FlashAttention-2, and the Qwen2.5-VL-3B backbone. We recommend creating an isolated Python environment:

```bash
conda create -n hycoiad python=3.10 -y
conda activate hycoiad

pip install torch torchvision
pip install transformers trl accelerate datasets deepspeed peft vllm \
    qwen-vl-utils pyyaml pillow numpy tqdm packaging \
    pandas matplotlib seaborn opencv-python
pip install flash-attn --no-build-isolation
```

Exact package versions will be added in a pinned environment file before the official release.

## Data Preparation

### 1. Training data

The included `train_data/train_data_grpo.jsonl` contains 1,164 question-answer pairs. Each record follows this format:

```json
{
  "id": 0,
  "question": "Is there any defect in the object? ...",
  "image": ["path/to/query_image.png"],
  "object_type": "metal_plate",
  "mask_path": "path/to/ground_truth_mask.png",
  "task_type": "Anomaly Detection",
  "solution": "A"
}
```

The released annotation file contains paths from the data-generation environment. Before training, replace the values in `image` and `mask_path` with paths valid on your machine, or set `--image_root` so that the resolved image locations match your dataset layout.

The local training annotations cover the following task types:

| Task type | Number of QA pairs |
|:--|--:|
| Anomaly Detection | 305 |
| Defect Classification | 227 |
| Defect Localization | 225 |
| Object Classification | 88 |
| Defect Description | 104 |
| Defect Analysis | 104 |
| Object Structure | 39 |
| Object Analysis | 38 |
| Object Details | 34 |
| **Total** | **1,164** |

The default data configuration is:

```yaml
datasets:
  - json_path: train_data/train_data_grpo.jsonl
```

### 2. MMAD benchmark

Download and prepare MMAD by following the instructions in the [official MMAD repository](https://github.com/jam-cc/MMAD). The evaluation script expects the following layout:

```text
/path/to/MMAD/
├── mmad.json
├── domain_knowledge.json                # Optional
└── ...                                  # Benchmark images
```

## Training

Download [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), then update the following variables in `run_scripts/run_grpo.sh`:

```bash
image_folders="/path/to/your/image/root"
model_path="Qwen/Qwen2.5-VL-3B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3
```

Launch distributed GRPO training from the repository root:

```bash
bash run_scripts/run_grpo.sh
```

The default script uses four GPUs, DeepSpeed ZeRO-2, FlashAttention-2, bfloat16 precision, a per-device batch size of 8, eight generations per prompt, and a learning rate of `5e-6` for the main trainable parameters. Adjust these settings according to your hardware.

Training checkpoints and debug logs are written to:

```text
checkpoints/rl/<experiment_name>/
runs/<experiment_name>/log/
```

## Evaluation

Evaluate a HyCo-IAD checkpoint on MMAD with:

```bash
python eval/evaluate_batch_mmad_choice_mymodel.py \
    --checkpoint /path/to/hycoiad_checkpoint \
    --data-root /path/to/MMAD \
    --batch-size 8 \
    --out-dir results/benchmark \
    --think True
```

The script performs multi-GPU inference, saves the predictions as JSON, and reports task-wise MMAD accuracy. Ensure that `mmad.json` exists directly under `--data-root`.

## Results

Performance on MMAD under the standard one-shot setting. The Anomaly Discrimination score is the mean accuracy over normal and anomalous samples. All values are percentages.

| Model | Scale | Anomaly Discrimination | Defect Classification | Localization | Description | Analysis | Object Classification | Object Analysis | Average |
|:--|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| GPT-4o | - | 68.63 | 65.80 | 55.62 | 73.21 | 83.41 | 94.98 | 82.80 | 74.92 |
| AnomalyR1 | 3B | 60.62 | 63.56 | 70.14 | 80.47 | 85.28 | 92.48 | 86.15 | 76.96 |
| Qwen2.5-VL (base) | 3B | 62.82 | 47.65 | 54.40 | 67.00 | 80.74 | 86.89 | 82.34 | 68.83 |
| Qwen2.5-VL (SFT) | 3B | 58.19 | 59.68 | 71.05 | 81.03 | 86.23 | 90.68 | 86.20 | 76.15 |
| **HyCo-IAD (ours)** | **3B** | **64.45** | **68.50** | **71.86** | **83.41** | **85.90** | **92.47** | **87.06** | **79.09** |

HyCo-IAD improves the average accuracy of the Qwen2.5-VL-3B backbone by **10.26 percentage points**, exceeds standard supervised fine-tuning by **2.94 points**, and outperforms AnomalyR1 by **2.13 points**.

## Citation

If you find this work useful, please cite:

```bibtex
@article{yang2026hycoiad,
  title   = {HyCo-IAD: Hyperbolic Collaborative Reinforcement Fine-Tuning for Multimodal Industrial Anomaly Detection},
  author  = {Yang, Enquan and Xu, Zhengqin and Xu, Xilin and Xing, Peng and Ma, Yuanwei and Li, Zechao and Zeng, Dan},
  year    = {2026}
}
```

The citation entry will be updated with the final venue, volume, pages, and DOI after publication.

## Acknowledgements

This repository builds upon [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [TRL](https://github.com/huggingface/trl), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL). We thank the authors of [MMAD](https://github.com/jam-cc/MMAD) for providing the industrial anomaly detection benchmark.

## License

A project license will be added before the public release. Third-party code, models, and datasets remain subject to their original licenses and terms of use.
