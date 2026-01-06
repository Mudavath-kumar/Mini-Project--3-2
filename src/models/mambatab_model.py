"""MambaTab: Selective State Space Model for Tabular Fraud Detection

Implements a simplified Mamba-style architecture with:
- Structured State Space (S4/S6) blocks for sequential modeling
- Selective scan mechanism (input-dependent state transitions)
- Efficient CPU-friendly implementation for tabular data
- Feature-wise attention and gating mechanisms

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
Adapted for tabular fraud detection with transaction sequences.
"""

from __future__ import annotations

import os

# Avoid slow/problematic torch._dynamo initialization on some environments.
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class TabularSequenceDataset(Dataset):
    """Minimal dataset that treats each row as a length-1 sequence.

    This is a simplification so we can plug into a GRU without having to
    rebuild the entire credit-card dataset as true sequences per card.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int):  # type: ignore[override]
        # (seq_len=1, feature_dim)
        x = self.X[idx].unsqueeze(0)
        y = self.y[idx]
        return x, y


class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) Block
    
    Implements the core Mamba mechanism:
    - Discretization of continuous-time state space
    - Input-dependent state transitions (selectivity)
    - Efficient sequential processing via parallel scan
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # Projection layers for input-dependent parameters
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # State space parameters (learnable initialization)
        # Delta (Δ): Controls discretization timestep (input-dependent)
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, self.d_inner),
            nn.Softplus()  # Ensure positive timesteps
        )
        
        # B and C: Input and output matrices (input-dependent via projection)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        
        # A: State transition matrix (structured initialization)
        # Initialized as diagonal for stability
        A = torch.randn(self.d_inner, d_state) * 0.01
        self.A = nn.Parameter(A)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input-dependent projections
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_ssm, gate = x_proj.chunk(2, dim=-1)  # Split for gating
        
        # Compute input-dependent SSM parameters
        delta = self.delta_proj(x)  # (batch, seq_len, d_inner) - discretization timestep
        B = self.B_proj(x)  # (batch, seq_len, d_state) - input matrix
        C = self.C_proj(x)  # (batch, seq_len, d_state) - output matrix
        
        # Selective scan (simplified CPU-friendly version)
        y = self.selective_scan(x_ssm, delta, B, C)
        
        # Gating mechanism (inspired by Mamba's gating)
        y = y * F.silu(gate)
        
        # Output projection and residual
        output = self.out_proj(y)
        return self.norm(output + x)  # Residual connection
    
    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor, 
                      B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """Simplified selective scan algorithm (CPU-friendly sequential version)
        
        State space equation:
            h[t] = A * h[t-1] + B[t] * x[t]  (state update)
            y[t] = C[t] * h[t]               (output)
        
        Delta controls the discretization (how much of new input to incorporate)
        """
        batch, seq_len, d_inner = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # Discretize A using delta (Zero-Order Hold discretization)
            # A_discrete = exp(delta * A)
            # delta[t] is (batch, d_inner); expand to (batch, d_inner, 1) so it can
            # broadcast with A (d_inner, d_state) -> (batch, d_inner, d_state).
            delta_t = delta[:, t, :].unsqueeze(-1)  # (batch, d_inner, 1)
            A_discrete = torch.exp(delta_t * self.A.unsqueeze(0))  # (batch, d_inner, d_state)
            
            # Input modulation
            B_t = B[:, t]  # (batch, d_state)
            x_t = x[:, t:t+1]  # (batch, 1, d_inner)
            
            # State update: h = A*h + B*x
            # Simplified: use mean pooling for d_inner -> d_state projection
            x_pooled = x_t.mean(dim=-1, keepdim=True).expand(-1, -1, self.d_state)  # (batch, 1, d_state)
            # Collapse the per-channel A to a single (batch, d_state) factor.
            h = (A_discrete.mean(dim=1) * h) + (B_t * x_pooled.squeeze(1))
            
            # Output: y = C*h
            C_t = C[:, t]  # (batch, d_state)
            y_t = (C_t * h).sum(dim=-1, keepdim=True)  # (batch, 1)
            y_t = y_t.expand(-1, d_inner)  # Broadcast to d_inner
            
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)


class MambaBlock(nn.Module):
    """Complete Mamba block with SSM + MLP"""
    
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, expand_factor)
        
        # MLP block (like in Transformer)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM block with residual
        x = self.ssm(x)
        # MLP block with residual
        x = x + self.mlp(self.norm(x))
        return x


class MambaTab(nn.Module):
    """MambaTab: State Space Model for Tabular Fraud Detection
    
    Architecture:
    1. Feature embedding layer
    2. Stack of Mamba blocks (Selective SSM)
    3. Global pooling + classification head
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Stack of Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, expand_factor)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - tabular features as sequence
        Returns:
            logits: (batch,) - fraud probability logits
        """
        # Embed features
        x = self.embedding(x)  # (batch, seq_len, hidden_dim)
        
        # Apply Mamba blocks
        for block in self.mamba_blocks:
            x = block(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(x)  # (batch, 1)
        return logits.squeeze(-1)


# Legacy alias for compatibility
class GRUTabularModel(MambaTab):
    """Backward compatibility alias - now uses MambaTab"""
    pass


@dataclass
class TrainConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2  # Number of Mamba blocks
    d_state: int = 16  # SSM state dimension
    expand_factor: int = 2  # Inner expansion factor
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 5
    device: str = "cpu"


def train_gru_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Tuple[MambaTab, dict]:
    """Train MambaTab state space model; return model and metrics dict."""

    device = torch.device(config.device)

    train_ds = TabularSequenceDataset(X_train, y_train)
    val_ds = TabularSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Initialize MambaTab model with SSM architecture
    model = MambaTab(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        d_state=config.d_state,
        expand_factor=config.expand_factor
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    # Reduce LR when validation loss plateaus.
    # Note: Some PyTorch builds don't accept a 'verbose=' kwarg here, so keep it minimal.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * len(x_batch)

        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(x_batch)

        val_loss /= len(val_ds)
        print(f"[MambaTab SSM] Epoch {epoch+1}/{config.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    metrics = {"val_loss": float(best_val_loss)}
    return model, metrics
