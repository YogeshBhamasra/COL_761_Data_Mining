import os

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score
from load_dataset import load_dataset

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x.view(-1)

def train(data_dir, model_dir, kerberos,
          hidden_channels=64, 
          lr=0.01, 
          weight_decay=5e-4, 
          epochs=500,
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

    train_idx = data.labeled_nodes[data.train_mask]
    val_idx   = data.labeled_nodes[data.val_mask]

    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=True,
    )

    val_loader = NeighborLoader(
        data,
        input_nodes=val_idx,
        num_neighbors=[15, 10],
        batch_size=1024,
        shuffle=False,
    )

    print(f"Dataset loaded with {data.num_nodes} nodes, {data.num_edges} edges, and {dataset.num_node_features} features.")
    model = SAGE(dataset.num_node_features, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if resume and checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0

    best_auc = 0
    patience, p_counter = 50, 0

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            labeled_out = model(batch.x, batch.edge_index)[:batch.batch_size]  # Only consider the output for the target node
            labeled_y = batch.y[:batch.batch_size].float()

            loss = F.binary_cross_entropy_with_logits(labeled_out, labeled_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                all_preds.append(out.cpu())
                all_labels.append(batch.y[:batch.batch_size].cpu())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        auc = roc_auc_score(all_labels.numpy(), all_preds.numpy())

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val AUC: {auc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(model_dir, f"{kerberos}_B_checkpoint.pt"))
        
        if auc > best_auc:
            best_auc = auc
            p_counter = 0
            torch.save(model, os.path.join(model_dir, f"{kerberos}_B.pt"))
            print(f"New best model saved with AUC: {best_auc:.4f}")
        else:
            p_counter += 1
    print(f"Best validation AUC: {best_auc:.4f}")