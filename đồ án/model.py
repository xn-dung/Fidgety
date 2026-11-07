from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, levels, codebook_size, latent_dim, commitment_weight):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, latent_dim) for _ in range(levels)])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, values):
        residual = values
        quantized_parts = []
        indices = []
        loss = values.new_tensor(0.0)
        for codebook in self.codebooks:
            distances = residual.pow(2).sum(-1, keepdim=True) + codebook.weight.pow(2).sum(-1) - 2 * residual @ codebook.weight.t()
            index = distances.argmin(-1)
            quantized = codebook(index)
            loss = loss + torch.mean((quantized - residual.detach()).pow(2))
            loss = loss + self.commitment_weight * torch.mean((quantized.detach() - residual).pow(2))
            quantized_parts.append(quantized)
            indices.append(index)
            residual = residual - quantized.detach()
        quantized_sum = torch.stack(quantized_parts).sum(0)
        straight_through = values + (quantized_sum - values).detach()
        return straight_through, loss, torch.stack(indices, dim=-1)


class RVQVAE(nn.Module):
    def __init__(self, features=60, frames=250, patch_size=10, hidden_dim=256, latent_dim=64, codebook_size=64, quantizers=2, commitment_weight=0.25):
        super().__init__()
        self.features = features
        self.frames = frames
        self.patch_size = patch_size
        patch_dim = features * patch_size
        self.encoder = nn.Sequential(nn.Linear(patch_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.quantizer = ResidualVectorQuantizer(quantizers, codebook_size, latent_dim, commitment_weight)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, patch_dim))

    def patches(self, values):
        return values.reshape(values.shape[0], self.frames // self.patch_size, self.patch_size * self.features)

    def forward(self, values):
        patches = self.patches(values)
        latent = self.encoder(patches)
        quantized, quantization_loss, indices = self.quantizer(latent)
        reconstruction = self.decoder(quantized).reshape_as(values)
        return reconstruction, quantization_loss, indices

    @torch.no_grad()
    def tokens(self, values):
        latent = self.encoder(self.patches(values))
        return self.quantizer(latent)[2]


def _loader(values, batch_size, shuffle):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def train_rvqvae(model, x_fit, x_val, device, epochs, patience, batch_size, learning_rate, weight_decay):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    fit_loader = _loader(x_fit, batch_size, True)
    val_loader = _loader(x_val, batch_size, False)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for (batch,) in fit_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, quantization_loss, _ = model(batch)
            loss = torch.mean((reconstruction - batch).pow(2)) + quantization_loss
            loss.backward()
            optimizer.step()
            train_total += float(loss.detach()) * len(batch)
            train_count += len(batch)
        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device, non_blocking=True)
                reconstruction, quantization_loss, _ = model(batch)
                loss = torch.mean((reconstruction - batch).pow(2)) + quantization_loss
                val_total += float(loss) * len(batch)
                val_count += len(batch)
        train_loss = train_total / train_count
        val_loss = val_total / val_count
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def encode_tokens(model, values, device, batch_size):
    model.eval()
    batches = []
    for (batch,) in _loader(values, batch_size, False):
        batches.append(model.tokens(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(batches)
