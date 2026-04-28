import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch.utils import data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score
from load_dataset import load_dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.ops import sigmoid_focal_loss
from evaluate import hits_at_k

import os
import time

from torch_geometric.data import Data

try:
    torch.serialization.add_safe_globals([Data])
except Exception as e:
    print(f"Warning: Could not add Data to safe globals for torch serialization: {e}")

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.3, num_layers=3):
        super().__init__()
        self.dropout = dropout
        self.convlist = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList()
        self.convlist.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convlist.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.batchnorms.append(nn.BatchNorm1d(hidden_channels))

        self.net = nn.Sequential(
            nn.Linear(4 * hidden_channels + 1, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )

    def encode(self, x, edge_index):
        for conv, bn in zip(self.convlist, self.batchnorms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def decode(self, z, edge_pairs):
        src, dst = edge_pairs[:, 0], edge_pairs[:, 1]
        dot = (z[src] * z[dst]).sum(dim=1, keepdim=True)
        edge_repr = torch.cat([
            z[src],
            z[dst],
            z[src] * z[dst],
            torch.abs(z[src] - z[dst]),
            dot
        ], dim=1)
        return self.net(edge_repr).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)
    
def train(data_dir, model_dir, kerberos,
          learning_rate=1e-3,
          hidden_channels=384,
          num_epochs=300,
          dropout=0.2,
          num_layers=3,
          resume=False, 
          checkpoint_path=None):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(model_dir, exist_ok=True)
    data = load_dataset("C", data_dir)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_pos = data.train_pos.to(device)
    train_neg = data.train_neg.to(device)
    valid_pos = data.valid_pos.to(device)
    valid_neg = data.valid_neg.to(device)

    P_value, K_value, _ = valid_neg.shape
    print(f"Nodes: {data.num_nodes} | Edges: {data.edge_index.size(1)} | Train Pos: {train_pos.shape[0]} | Train Neg: {train_neg.shape[0]} | Valid Pos: {valid_pos.shape[0]} | Valid Neg: {valid_neg.shape[0]} (P={P_value}, K={K_value})")

    model = LinkPredictor(
        in_channels=x.shape[1],
        hidden_channels=hidden_channels,
        dropout=dropout,
        num_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_hits = 0
    patience, p_counter = 15, 0
    N_train = train_pos.shape[0]

    if resume:
        if checkpoint_path is None:
            raise NotImplementedError("Checkpoint path must be provided when --resume is set.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_hits = checkpoint.get("best_hits", 0)
        print(f"Resumed training from checkpoint: {checkpoint_path} with best_hits={best_hits:.4f}")

    print(f"{'=' * 50} Starting training {'=' * 50}")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        n = min(train_pos.shape[0], train_neg.shape[0])

        perm_pos = torch.randperm(train_neg.shape[0], device=device)[:n]
        perm_neg = torch.randperm(train_neg.shape[0], device=device)[:n]

        neg_batch = train_neg[perm_neg]

        pos_batch = train_pos[perm_pos]

        z = model.encode(x, edge_index)

        pos_scores = model.decode(z, pos_batch)
        neg_scores = model.decode(z, neg_batch)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)

        bce_loss = F.binary_cross_entropy_with_logits(scores, labels)


        neg_scores = neg_scores[:pos_scores.shape[0]]
        rank_loss = F.margin_ranking_loss(pos_scores, neg_scores, torch.ones_like(pos_scores), margin=0.5)

        loss = 1.0 * rank_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                valid_pos_scores = model(x, edge_index, valid_pos)

                P, K, _ = valid_neg.shape

                valid_neg_scores = model(x, edge_index, valid_neg.view(P * K, 2)).view(P, K)

                hits = hits_at_k(valid_pos_scores.cpu(), valid_neg_scores.cpu(), k=50)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} (BCE {bce_loss.item():.4f} + Rank {rank_loss.item():.4f}) | Valid Hits@50: {hits:.4f}")

                if hits > best_hits:
                    best_hits = hits
                    p_counter = 0
                    if model_dir:
                        save_path = os.path.join(model_dir, f"{kerberos}_model_C.pt")
                        torch.save(model, save_path)
                        print(f"New best model saved to {save_path}")
                else:
                    p_counter += 1
                    if p_counter >= patience:
                        print(f"No improvement for {patience} consecutive evaluations. Early stopping.")
                        break

        torch.save(model, os.path.join(model_dir, f"{kerberos}_model_C_checkpoint.pt"))


        
