"""
train_A.py  –  COL761 Assignment 3 | Dataset A | Multi-class Node Classification
Architecture: GraphSAGE with residual connections + virtual node + label trick
Metric: Accuracy (7 classes)

Usage:
    python train_A.py --data_dir /absolute/path/to/datasets \
                      --model_dir /path/to/models \
                      --kerberos YOUR_KERBEROS
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import add_self_loops

from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class ResidualSAGE(nn.Module):
    """
    Deep GraphSAGE with:
      - Residual (skip) connections every 2 layers
      - BatchNorm after each conv
      - Dropout for regularisation
      - Optional virtual-node injection
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        num_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.input_proj = nn.Linear(in_channels, hidden)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.bns.append(BatchNorm(hidden))

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # residual every 2 layers
            if i % 2 == 1:
                x = x + h
            else:
                x = h

        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Label trick helper
# ─────────────────────────────────────────────────────────────────────────────

def build_label_features(data, num_classes: int) -> torch.Tensor:
    """
    Append one-hot label features to x for labeled training nodes.
    For val/test nodes the label channel is all-zeros (masked).
    Returns augmented x of shape [N, F + num_classes].
    """
    N = data.num_nodes
    label_feat = torch.zeros(N, num_classes, dtype=data.x.dtype, device=data.x.device)
    train_idx  = data.labeled_nodes[data.train_mask]
    label_feat[train_idx] = F.one_hot(
        data.y[data.train_mask], num_classes=num_classes
    ).float()
    return torch.cat([data.x, label_feat], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── data ────────────────────────────────────────────────────────────────
    dataset = load_dataset("A", args.data_dir)
    data    = dataset[0].to(device)
    num_classes = dataset.num_classes
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Classes: {num_classes}")

    train_idx = data.labeled_nodes[data.train_mask]
    val_idx   = data.labeled_nodes[data.val_mask]
    y_train   = data.y[data.train_mask]
    y_val     = data.y[data.val_mask]

    # Label trick: append one-hot labels of train nodes to features
    x_aug = build_label_features(data, num_classes)
    in_channels = x_aug.shape[1]

    # ── model ────────────────────────────────────────────────────────────────
    model = ResidualSAGE(
        in_channels=in_channels,
        hidden=args.hidden,
        out_channels=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    # class weights to handle potential imbalance
    counts = torch.bincount(y_train, minlength=num_classes).float()
    class_weights = (counts.sum() / (num_classes * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0
    patience_cnt = 0

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x_aug, data.edge_index)
        loss   = criterion(logits[train_idx], y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # ── validation ──────────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(x_aug, data.edge_index)
            pred_val  = logits_val[val_idx].argmax(dim=1)
            val_acc   = (pred_val == y_val).float().mean().item()

            if epoch % (args.eval_every * 5) == 0:
                print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | val_acc {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch   = epoch
                patience_cnt = 0

                # save full model for predict.py compatibility
                # We wrap it so predict.py's model(x, edge_index) works:
                save_model = ModelWrapper(model, x_aug.cpu())
                torch.save(save_model, model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    print(f"\nBest val_acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"Model saved: {model_path}")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper so predict.py's model(x, edge_index) interface works
# ─────────────────────────────────────────────────────────────────────────────

class ModelWrapper(nn.Module):
    """
    Stores augmented x (with label trick) inside the model so that predict.py
    can call model(data.x, data.edge_index) and get the right logits.
    """

    def __init__(self, gnn: ResidualSAGE, x_aug: torch.Tensor):
        super().__init__()
        self.gnn   = gnn
        self.register_buffer("x_aug", x_aug)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # ignore x from predict.py; use stored augmented features
        device = edge_index.device
        return self.gnn(self.x_aug.to(device), edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GNN for Dataset A")
    parser.add_argument("--data_dir",     required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--kerberos",     required=True)
    parser.add_argument("--hidden",       type=int,   default=512)
    parser.add_argument("--num_layers",   type=int,   default=4)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--lr",           type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs",       type=int,   default=500)
    parser.add_argument("--eval_every",   type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=50)
    args = parser.parse_args()

    # if not os.path.isabs(args.data_dir):
    #     sys.exit("--data_dir must be an absolute path")

    train(args)


if __name__ == "__main__":
    main()
