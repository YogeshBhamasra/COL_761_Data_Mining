# COL761 Assignment 3 — Setup & Usage

## Installation

Install the provided dependencies first:

```bash
pip install -r requirements.txt
```

### ⚠️ Additional Dependency: `pyg-lib`

`pyg-lib` is **required** for `NeighborLoader` and is not included in `requirements.txt`. Install it separately:

```bash
pip install pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)").html
```

> **Note:** Run this **after** installing `requirements.txt` so that `torch` is already available for version detection.

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

