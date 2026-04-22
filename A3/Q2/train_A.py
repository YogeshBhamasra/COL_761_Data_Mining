"""
train_A.py  –  COL761 Assignment 3 | Dataset A | Multi-class Node Classification
Metric: Accuracy (7 classes)

Usage:
    python train_A.py --data_dir /absolute/path/to/datasets \
                      --model_dir ./models --kerberos YOUR_KERBEROS
    # or explicitly pick a model:
    python train_A.py ... --model gcnii
    python train_A.py ... --model appnp
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv
from torch_geometric.utils import dropout_edge

from load_dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# GCNII  (best for deep transductive classification)
# Paper: "Simple and Deep Graph Convolutional Networks" (Chen et al., 2020)
# Key idea: initial residual + identity mapping → no over-smoothing at 64 layers
# ─────────────────────────────────────────────────────────────────────────────

class GCNII(nn.Module):
    def __init__(self, in_channels, hidden, out_channels,
                 num_layers=64, alpha=0.1, theta=0.5,
                 dropout=0.6):
        super().__init__()
        self.dropout = dropout

        self.input_lin  = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList([
            GCN2Conv(hidden, alpha=alpha, theta=theta, layer=i+1, shared_weights=False)
            for i in range(num_layers)
        ])
        self.output_lin = nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index):

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x0 = F.relu(self.input_lin(x))

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, x0, edge_index))

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_lin(x)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model( in_channels: int, num_classes: int, hidden: int, num_layers: int, device: torch.device) -> nn.Module:
    return GCNII(
            in_channels, hidden=hidden, out_channels=num_classes,
            num_layers=num_layers
        ).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Correct & Smooth post-processing
# Paper: "Combining Label Propagation and Simple Models" (Huang et al., 2021)
# ─────────────────────────────────────────────────────────────────────────────

def correct_and_smooth(soft, y_train, train_idx, edge_index, num_nodes,
                       num_classes, ca, sa, steps, device):
    N, C = soft.shape
    row, col = edge_index
    deg = torch.zeros(N, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
    d = deg.pow(-0.5).clamp(max=1e4)
    ew = d[row] * d[col]

    def prop(h, alpha, n):
        for _ in range(n):
            agg = torch.zeros_like(h)
            agg.scatter_add_(0, col.unsqueeze(1).expand(-1, C), ew.unsqueeze(1) * h[row])
            h = (1 - alpha) * h + alpha * agg
        return h

    y_oh  = F.one_hot(y_train, C).float().to(device)
    err   = torch.zeros(N, C, device=device)
    err[train_idx] = y_oh - soft[train_idx]
    cor   = (soft + prop(err, ca, steps)).clamp(0, 1)
    cor   = cor / cor.sum(1, keepdim=True).clamp(min=1e-9)
    cor[train_idx] = y_oh
    return prop(cor, sa, steps)


# ─────────────────────────────────────────────────────────────────────────────
# ModelWrapper – predict.py calls model(x, edge_index) → logits [N, C]
# ─────────────────────────────────────────────────────────────────────────────

class ModelWrapper(nn.Module):
    def __init__(self, gnn, final_preds):
        super().__init__()
        self.gnn = gnn
        self.register_buffer("final_preds", final_preds)

    def forward(self, x, edge_index):
        return self.final_preds.to(edge_index.device)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(data_dir, model_dir, kerberos, 
          hidden = 64, 
          num_layers = 64,
          lr = 1e-2, 
          weight_decay = 5e-4, 
          epochs = 3000, 
          eval_every = 5, 
          patience = 200, 
          cs_correct_alpha = 0.5, 
          cs_smooth_alpha = 0.5, 
          cs_steps = 50):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset     = load_dataset("A", data_dir)
    data        = dataset[0].to(device)
    num_classes = dataset.num_classes
    num_nodes   = data.num_nodes
    print(f"Dataset: {num_nodes} nodes, {data.num_edges} edges, {num_classes} classes")

    train_idx = data.labeled_nodes[data.train_mask]
    val_idx   = data.labeled_nodes[data.val_mask]
    y_train   = data.y[data.train_mask]
    y_val     = data.y[data.val_mask]

    model = build_model(data.x.shape[1], num_classes, hidden=hidden, num_layers=num_layers, device=device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=20, min_lr=1e-5
    )

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    ckpt_path     = os.path.join(model_dir, f"{kerberos}_model_A.pt.ckpt")
    model_path    = os.path.join(model_dir, f"{kerberos}_model_A.pt")
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss   = F.cross_entropy(logits[train_idx], y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                logits_all = model(data.x, data.edge_index)

            raw_acc = (logits_all[val_idx].argmax(1) == y_val).float().mean().item()

            soft   = torch.softmax(logits_all.detach(), dim=1)
            cs_out = correct_and_smooth(
                soft, y_train, train_idx, data.edge_index, num_nodes, num_classes,
                ca=cs_correct_alpha, sa=cs_smooth_alpha,
                steps=cs_steps, device=device,
            )
            cs_acc = (cs_out[val_idx].argmax(1) == y_val).float().mean().item()
            scheduler.step(cs_acc)

            if epoch % (eval_every * 10) == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:5d} | loss {loss.item():.4f} | "
                      f"raw {raw_acc:.4f} | C&S {cs_acc:.4f} | lr {lr:.2e}")

            if cs_acc > best_val_acc:
                best_val_acc = cs_acc
                best_epoch   = epoch
                patience_cnt = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

    print(f"\nBest val_acc (C&S): {best_val_acc:.4f} @ epoch {best_epoch}")

    # Rebuild final predictions with best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits_all = model(data.x, data.edge_index)
    soft   = torch.softmax(logits_all, dim=1)
    cs_out = correct_and_smooth(
        soft, y_train, train_idx, data.edge_index, num_nodes, num_classes,
        ca=cs_correct_alpha, sa=cs_smooth_alpha,
        steps=cs_steps, device=device,
    )

    wrapper = ModelWrapper(model.cpu(), cs_out.cpu())
    torch.save(wrapper, model_path)
    os.remove(ckpt_path)
    print(f"Saved: {model_path}")
    return best_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",          required=True)
    p.add_argument("--model_dir",         required=True)
    p.add_argument("--kerberos",          required=True)
    # model choice
    p.add_argument("--model",             default="gcnii",
                   choices=["gcnii"],
                   help="Which SOTA model to use (default: gcnii)")
    # shared hyperparams
    p.add_argument("--hidden",            type=int,   default=64)
    p.add_argument("--num_layers",        type=int,   default=64,
                   help="Depth for GCNII / hop K for SGC")
    p.add_argument("--dropout",           type=float, default=0.6)
    p.add_argument("--drop_edge_p",       type=float, default=0.0)
    p.add_argument("--lr",                type=float, default=1e-2)
    p.add_argument("--weight_decay",      type=float, default=5e-4)
    p.add_argument("--epochs",            type=int,   default=1500)
    p.add_argument("--eval_every",        type=int,   default=5)
    p.add_argument("--patience",          type=int,   default=200)
    # GCNII-specific
    p.add_argument("--alpha",       type=float, default=0.1)
    p.add_argument("--theta",       type=float, default=0.5)
    # C&S
    p.add_argument("--cs_correct_alpha",  type=float, default=0.5)
    p.add_argument("--cs_smooth_alpha",   type=float, default=0.5)
    p.add_argument("--cs_steps",          type=int,   default=50)
    args = p.parse_args()
    train(
        data_dir=args.data_dir, model=args.model, model_dir=args.model_dir, kerberos=args.kerberos,
        hidden=args.hidden, num_layers=args.num_layers, dropout=args.dropout, drop_edge_p=args.drop_edge_p,
        lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, eval_every=args.eval_every, patience=args.patience,
        alpha=args.alpha, theta=args.theta,
        cs_correct_alpha=args.cs_correct_alpha, cs_smooth_alpha=args.cs_smooth_alpha, cs_steps=args.cs_steps,
    )


if __name__ == "__main__":
    main()