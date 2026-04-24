import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
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
    data = dataset[0].to(device)

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
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        labeled_out = out[data.labeled_nodes]
        labeled_y = data.y.float()

        loss = F.binary_cross_entropy_with_logits(labeled_out[data.train_mask], labeled_y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = out[data.labeled_nodes][data.val_mask]
            val_labels = labeled_y[data.val_mask]

            auc = roc_auc_score(val_labels.cpu(), val_out.cpu())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': auc,
            }, f"{model_dir}/checkpoint_epoch_{epoch}.pt")
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val AUC: {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model, f"{model_dir}/best_model.pt")
                p_counter = 0
                print(f"New best model saved at epoch {epoch} with AUC: {auc:.4f}")
            else:
                p_counter += 1
                if p_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    print(f"Best validation AUC: {best_auc:.4f}")