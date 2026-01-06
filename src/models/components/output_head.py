"""
Output Head for MambaTab model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class OutputHead(nn.Module):
    """
    Output head for classification or regression tasks.
    """
    
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        task_type: str = "classification",
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize Output Head.
        
        Args:
            d_model: Input dimension
            num_classes: Number of output classes (1 for regression)
            task_type: "classification" or "regression"
            dropout: Dropout rate
            hidden_dim: Hidden dimension for MLP (default: d_model * 2)
        """
        super().__init__()
        
        self.task_type = task_type
        self.num_classes = num_classes
        hidden_dim = hidden_dim or d_model * 2
        
        # Determine output dimension
        if task_type == "classification":
            if num_classes == 2:
                output_dim = 1  # Binary classification with sigmoid
            else:
                output_dim = num_classes  # Multi-class with softmax
        else:
            output_dim = num_classes  # Regression
        
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, d_model]
            
        Returns:
            Logits [batch_size, num_classes] or [batch_size, 1]
        """
        return self.head(x)
