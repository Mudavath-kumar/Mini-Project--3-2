

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math

from .components.mamba_block import MambaBlock
from .components.feature_encoder import FeatureEncoder
from .components.output_head import OutputHead


class MambaTab(nn.Module):
    """
    MambaTab: A Mamba-based model for tabular data.
    
    This model uses the Mamba architecture (selective state space model)
    adapted for tabular data processing.
    """
    
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        task_type: str = "classification",
        num_numerical: int = 0,
        num_categorical: int = 0,
        categorical_cardinalities: Optional[List[int]] = None,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ):
        """
        Initialize MambaTab model.
        
        Args:
            num_features: Total number of input features
            num_classes: Number of output classes (1 for regression)
            d_model: Model dimension
            n_layers: Number of Mamba layers
            d_state: State dimension for Mamba
            d_conv: Convolution dimension for Mamba
            expand_factor: Expansion factor for inner dimension
            dropout: Dropout rate
            task_type: "classification" or "regression"
            num_numerical: Number of numerical features
            num_categorical: Number of categorical features
            categorical_cardinalities: List of cardinalities for each categorical feature
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            activation: Activation function ("gelu", "relu", "silu")
        """
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        self.task_type = task_type
        self.num_numerical = num_numerical if num_numerical > 0 else num_features
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities or []
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(
            num_numerical=self.num_numerical,
            num_categorical=self.num_categorical,
            categorical_cardinalities=self.categorical_cardinalities,
            d_model=d_model,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        
        # Calculate input dimension after encoding
        encoded_dim = self.feature_encoder.output_dim
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(encoded_dim, d_model),
            nn.LayerNorm(d_model) if use_layer_norm else nn.Identity(),
            self._get_activation(activation),
            nn.Dropout(dropout),
        )
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        
        # Output head
        self.output_head = OutputHead(
            d_model=d_model,
            num_classes=num_classes,
            task_type=task_type,
            dropout=dropout,
            hidden_dim=d_model * 2,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
        }
        return activations.get(activation.lower(), nn.GELU())
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x_numerical: Optional[torch.Tensor] = None,
        x_categorical: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_numerical: Numerical features [batch_size, num_numerical]
            x_categorical: Categorical features [batch_size, num_categorical]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary containing logits and optionally embeddings
        """
        # Encode features
        encoded = self.feature_encoder(x_numerical, x_categorical)
        
        # Project to model dimension
        x = self.input_projection(encoded)
        
        # Add sequence dimension for Mamba [batch_size, 1, d_model]
        x = x.unsqueeze(1)
        
        # Apply Mamba layers
        embeddings = []
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
            if return_embeddings:
                embeddings.append(x.squeeze(1))
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Output head
        logits = self.output_head(x)
        
        output = {"logits": logits}
        if return_embeddings:
            output["embeddings"] = embeddings
            output["final_embedding"] = x
        
        return output
    
    def predict(
        self,
        x_numerical: Optional[torch.Tensor] = None,
        x_categorical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            x_numerical: Numerical features
            x_categorical: Categorical features
            
        Returns:
            Predictions (probabilities for classification, values for regression)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x_numerical, x_categorical)
            logits = output["logits"]
            
            if self.task_type == "classification":
                if self.num_classes == 2:
                    return torch.sigmoid(logits)
                else:
                    return F.softmax(logits, dim=-1)
            else:
                return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
