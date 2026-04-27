# COL761 Assignment 3 — Setup & Usage

## Installation

Install the updated dependencies first:

```bash
pip install -r requirements.txt
```

> **Note:** The original `requirements.txt` provided with the assignment did not work as-is. The following changes were made to produce the working `requirements.txt` included here:
>
> - **PyTorch downgraded** from `2.7.1+cu118` → `2.4.1` (with matching `torchaudio==2.4.1`, `torchvision==0.19.1`, `triton==3.0.0`) — the newer build caused compatibility issues.
> - **CUDA 12 runtime libraries added** — all `nvidia-*-cu12` packages were added alongside the existing `cu11` ones to satisfy driver-level dependencies on the target environment.
> - `pip`, `setuptools`, and `wheel` were removed from the pinned list (these are installer tools, not project dependencies).

### ⚠️ Additional Dependencies: `torch-sparse` and `torch-scatter`

`torch-sparse` and `torch-scatter` are **required** for `NeighborLoader` and are not included in `requirements.txt`. Install them separately after the step above:

```bash
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html

export PYG_USE_PYG_LIB=0
```

> **Note:** Run this **after** installing `requirements.txt` so that `torch` is already available for version detection. The `PYG_USE_PYG_LIB=0` export disables `pyg-lib` (which was the original approach but proved unnecessary) and ensures PyG falls back to the `torch-sparse`/`torch-scatter` backend.

---

## Training

```bash
python train.py --dataset A|B|C --task node|link \
    --data_dir /absolute/path/to/datasets \
    --model_dir /path/to/models \
    --kerberos YOUR_KERBEROS
```

---

## Prediction

```bash
python predict.py --dataset A|B|C --task node|link \
    --data_dir /absolute/path/to/datasets \
    --model_dir /path/to/models \
    --output_dir /path/to/outputs \
    --kerberos YOUR_KERBEROS
```

---

## Evaluation

```bash
python evaluate.py --dataset A|B|C --task node|link \
    --data_dir /absolute/path/to/datasets \
    --output_dir /path/to/outputs \
    --kerberos YOUR_KERBEROS
```

| Dataset | Task | Metric |
|---------|------|--------|
| A | Node classification (multi-class) | Accuracy |
| B | Node classification (binary) | AUC-ROC |
| C | Link prediction | Hits@50 |

---
