
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import accuracy_score, f1_score

class TacticalGNN(nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads=num_heads, edge_dim=2, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, edge_dim=2, add_self_loops=True)
        self.conv3 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=1, edge_dim=2, add_self_loops=True)
        self.receiver_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1)
        )
        self.shot_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        global_x = global_mean_pool(x, batch)
        shot_probs = torch.sigmoid(self.shot_head(global_x))
        receiver_probs = torch.sigmoid(self.receiver_head(x))
        return receiver_probs, shot_probs

class TacticalVAE(nn.Module):
    def __init__(self, num_features, hidden_channels=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        self.fc_mu = nn.Linear(hidden_channels, latent_dim)
        self.fc_var = nn.Linear(hidden_channels, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, num_features)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=0.01):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss

class TacticalAI(nn.Module):
    def __init__(self, num_features, hidden_channels=64, num_heads=4, latent_dim=32):
        super().__init__()
        self.gnn = TacticalGNN(num_features, hidden_channels, num_heads)
        self.vae = TacticalVAE(num_features, hidden_channels, latent_dim)

    def forward(self, data):
        receiver_probs, shot_probs = self.gnn(data)
        recon_x, mu, logvar = self.vae(data.x)
        return receiver_probs, shot_probs, recon_x, mu, logvar

def evaluate_model(model, loader):
    model.eval()
    all_receiver_preds = []
    all_receiver_labels = []
    all_shot_preds = []
    all_shot_labels = []

    with torch.no_grad():
        for batch in loader:
            receiver_probs, shot_probs, _, _, _ = model(batch)
            attacker_mask = batch.x[:, -1] == 1
            receiver_target = batch.receiver_labels

            shot_pred = (shot_probs > 0.5).long().squeeze()
            shot_label = batch.y[:, 1].long()
            all_shot_preds.extend(shot_pred.cpu().numpy())
            all_shot_labels.extend(shot_label.cpu().numpy())

            if attacker_mask.any():
                receiver_pred = (receiver_probs.squeeze() > 0.5).long()[attacker_mask]
                receiver_true = receiver_target[attacker_mask].long()
                all_receiver_preds.extend(receiver_pred.cpu().numpy())
                all_receiver_labels.extend(receiver_true.cpu().numpy())

    receiver_acc = accuracy_score(all_receiver_labels, all_receiver_preds) if all_receiver_labels else None
    receiver_f1 = f1_score(all_receiver_labels, all_receiver_preds, zero_division=0) if all_receiver_labels else None
    shot_acc = accuracy_score(all_shot_labels, all_shot_preds)
    return receiver_acc, receiver_f1, shot_acc

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            receiver_probs, shot_probs, recon_x, mu, logvar = model(batch)
            attacker_mask = batch.x[:, -1] == 1
            receiver_target = batch.receiver_labels.unsqueeze(1)
            if attacker_mask.any():
                receiver_loss = F.binary_cross_entropy(receiver_probs[attacker_mask], receiver_target[attacker_mask])
            else:
                receiver_loss = torch.tensor(0.0, device=batch.x.device)
            shot_loss = F.binary_cross_entropy(shot_probs, batch.y[:, 1:2])
            vae_loss = model.vae.loss_function(recon_x, batch.x, mu, logvar)
            loss = receiver_loss + shot_loss + vae_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                receiver_probs, shot_probs, recon_x, mu, logvar = model(batch)
                attacker_mask = batch.x[:, -1] == 1
                receiver_target = batch.receiver_labels.unsqueeze(1)
                if attacker_mask.any():
                    receiver_loss = F.binary_cross_entropy(receiver_probs[attacker_mask], receiver_target[attacker_mask])
                else:
                    receiver_loss = torch.tensor(0.0, device=batch.x.device)
                shot_loss = F.binary_cross_entropy(shot_probs, batch.y[:, 1:2])
                vae_loss = model.vae.loss_function(recon_x, batch.x, mu, logvar)
                val_loss += (receiver_loss + shot_loss + vae_loss).item()

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Average Training Loss: {total_loss / len(train_loader):.4f}")
        print(f"Average Validation Loss: {val_loss / len(val_loader):.4f}")

    # Final evaluation after training
    receiver_acc, receiver_f1, shot_acc = evaluate_model(model, val_loader)
    print(f"Final Shot Accuracy: {shot_acc:.4f}")
    if receiver_acc is not None:
        print(f"Final Receiver Accuracy: {receiver_acc:.4f}, F1 Score: {receiver_f1:.4f}")
    else:
        print("Receiver evaluation skipped due to no attacker labels.")
