"""
Mamba Block implementation for selective state space modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MambaBlock(nn.Module):
    """
    Mamba Block: Selective State Space Model for sequence modeling.
    
    This implements the core Mamba architecture with selective scan mechanism.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        """
        Initialize Mamba Block.
        
        Args:
            d_model: Model dimension
            d_state: State dimension (N in Mamba paper)
            d_conv: Convolution kernel size
            expand_factor: Expansion factor for inner dimension
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            dt_min: Minimum value for dt
            dt_max: Maximum value for dt
            dt_init: Initialization method for dt ("random" or "constant")
            dt_scale: Scale for dt initialization
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Delta (dt) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A parameter (state matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Layer norm
        x = self.norm(x)
        
        # Input projection and split
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)  # [B, L, d_inner]
        
        # Activation
        x = F.silu(x)
        
        # SSM
        x = self._ssm(x)
        
        # Gating
        z = F.silu(z)
        x = x * z
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # Residual connection
        return x + residual
    
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model computation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_inner]
            
        Returns:
            Output tensor [batch_size, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # Project to get B, C, and dt
        x_proj = self.x_proj(x)  # [B, L, d_state * 2 + 1]
        
        # Split into B, C, and dt_input
        B = x_proj[:, :, :self.d_state]  # [B, L, d_state]
        C = x_proj[:, :, self.d_state:self.d_state * 2]  # [B, L, d_state]
        dt_input = x_proj[:, :, -1:]  # [B, L, 1]
        
        # Compute dt
        dt = F.softplus(self.dt_proj(dt_input))  # [B, L, d_inner]
        
        # Discretize A and B
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, d_inner, d_state]
        
        # Selective scan
        y = self._selective_scan(x, dA, dB, C)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dA: torch.Tensor,
        dB: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform selective scan operation.
        
        Args:
            x: Input [B, L, d_inner]
            dA: Discretized A [B, L, d_inner, d_state]
            dB: Discretized B [B, L, d_inner, d_state]
            C: C matrix [B, L, d_state]
            
        Returns:
            Output [B, L, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = dA.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Update state: h = dA * h + dB * x
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            
            # Output: y = C * h
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)
