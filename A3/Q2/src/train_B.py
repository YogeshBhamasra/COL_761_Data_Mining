import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score
from load_dataset import load_dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.ops import sigmoid_focal_loss

from torch_geometric.data import Data

try:
    torch.serialization.add_safe_globals([Data])
except Exception as e:
    print(f"Warning: Could not add Data to safe globals for torch serialization: {e}")

import os
import time

try:
    num_cpus = len(os.sched_getaffinity(0))  # Linux (HPC + local)
except AttributeError:
    num_cpus = os.cpu_count()  # macOS / Windows fallback

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.3, num_layers=3):
        super().__init__()
        self.dropout = dropout
        self.convlist = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList()
        self.convlist.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convlist.append(SAGEConv(hidden_channels, hidden_channels))
        self.convlist.append(SAGEConv(hidden_channels, hidden_channels))

        for _ in range(num_layers):
            self.batchnorms.append(nn.BatchNorm1d(hidden_channels))

        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convlist, self.batchnorms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x

    @torch.no_grad()
    def inference(self, loader, device):
        all_preds = []
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                out = self.forward(batch.x, batch.edge_index)[:batch.batch_size].squeeze(1)
            all_preds.append(out)
        return torch.cat(all_preds, dim=0).cpu()

def train(data_dir, model_dir, kerberos,
          hidden_channels=256, 
          lr=1e-5, 
          weight_decay=5e-4, 
          epochs=100,
          batch_size=1024,
          num_layers=3,
          resume=False,
          checkpoint_path=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:        
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    dataset = load_dataset("B", data_dir)
    data = dataset[0]
    os.makedirs(model_dir, exist_ok=True)
    print(f"    Dataset loaded with {data.num_nodes} nodes, {data.num_edges} edges, and {dataset.num_node_features} features.")

    y_full = torch.full((data.num_nodes,), float('nan'))
    y_full[data.labeled_nodes] = data.y
    data.y = y_full

    train_nodes = data.labeled_nodes[data.train_mask]
    val_nodes   = data.labeled_nodes[data.val_mask]

    # num_workers = max(8, num_cpus // 2)
    num_workers = 8

    train_loader = NeighborLoader(
        data,
        input_nodes=train_nodes,
        num_neighbors=[10]*num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = NeighborLoader(
        data,
        input_nodes=val_nodes,
        num_neighbors=[10]*num_layers,  # Use all neighbors for validation
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"    Train loader created with {len(train_loader)} batches, batch size {batch_size}, num_workers {num_workers}")
    print(f"    Validation loader created with {len(val_loader)} batches, batch size {batch_size}, num_workers {num_workers}")

    model = SAGE(
        in_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(device)
    print(f"    Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    criterion = lambda logits, labels: sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0, reduction='mean')

    best_auc = 0
    patience, p_counter = 20, 0

    if resume:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided if --resume is set")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if 'scheduler_state_dict' in checkpoint else None
        best_auc = checkpoint['best_auc'] if 'best_auc' in checkpoint else 0
        p_counter = checkpoint['p_counter'] if 'p_counter' in checkpoint else 0
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    print(f"\n\n{'='*50}\nStarting training...\n{'='*50}")
    t_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        total_examples = 0


        for batch in train_loader:
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            optimizer.zero_grad()
            labeled_out = model(x, edge_index)[:batch.batch_size].squeeze(1)  # Only consider the output for the target node
            labeled_y = batch.y[:batch.batch_size].float().to(device)  # Get the labels for the target nodes

            # print(f"labeled_out for first 10 nodes: {labeled_out[:10].cpu().detach().numpy()}")
            # print(f"labeled_y for first 10 nodes: {labeled_y[:10].cpu().detach().numpy()}")

            # print(f"Unique labels in batch: {torch.unique(batch.y)}")

            loss = criterion(labeled_out, labeled_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        
        scheduler.step()
        loss_avg = total_loss / total_examples

        if epoch % 10 == 0 or epoch == epochs - 1:
            end_time = time.time()
            elapsed = end_time - t_start
            val_pred = model.inference(val_loader, device)
            val_labels = data.y[val_nodes].cpu()
            auc = roc_auc_score(val_labels.numpy(), val_pred.numpy())
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {loss_avg:.4f}, Val AUC: {auc:.4f}, Time Elapsed: {elapsed:.2f}s")
            model.eval()
            if auc > best_auc:
                best_auc = auc
                p_counter = 0
                torch.save(model, os.path.join(model_dir, f"{kerberos}_model_B.pt"))
                print(f"New best model at epoch {epoch+1} saved with AUC: {best_auc:.4f}")
            else:
                p_counter += 1
            
            if p_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} with best AUC: {best_auc:.4f}")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_auc': best_auc,
            'p_counter': p_counter,
        }, os.path.join(model_dir, f"{kerberos}_model_B_checkpoint.pt"))

    print(f"Best validation AUC: {best_auc:.4f}")
