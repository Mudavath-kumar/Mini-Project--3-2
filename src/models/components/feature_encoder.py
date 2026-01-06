"""
Feature Encoder for tabular data preprocessing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FeatureEncoder(nn.Module):
    """
    Encodes numerical and categorical features for the MambaTab model.
    """
    
    def __init__(
        self,
        num_numerical: int = 0,
        num_categorical: int = 0,
        categorical_cardinalities: Optional[List[int]] = None,
        d_model: int = 128,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        embedding_dim: Optional[int] = None,
    ):
        """
        Initialize Feature Encoder.
        
        Args:
            num_numerical: Number of numerical features
            num_categorical: Number of categorical features
            categorical_cardinalities: List of cardinalities for each categorical feature
            d_model: Model dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            embedding_dim: Embedding dimension for categorical features (default: d_model // 4)
        """
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities or []
        self.d_model = d_model
        self.embedding_dim = embedding_dim or max(d_model // 4, 8)
        
        # Numerical feature encoder
        if num_numerical > 0:
            self.numerical_encoder = NumericalEncoder(
                num_features=num_numerical,
                d_model=d_model,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            )
            self.numerical_output_dim = d_model
        else:
            self.numerical_encoder = None
            self.numerical_output_dim = 0
        
        # Categorical feature encoder
        if num_categorical > 0 and len(self.categorical_cardinalities) > 0:
            self.categorical_encoder = CategoricalEncoder(
                cardinalities=self.categorical_cardinalities,
                embedding_dim=self.embedding_dim,
                dropout=dropout,
            )
            self.categorical_output_dim = num_categorical * self.embedding_dim
        else:
            self.categorical_encoder = None
            self.categorical_output_dim = 0
        
        # Combined output dimension
        self.output_dim = self.numerical_output_dim + self.categorical_output_dim
        
        # Ensure output_dim is at least d_model for the model to work
        if self.output_dim == 0:
            self.output_dim = d_model
            self.fallback_projection = nn.Linear(1, d_model)
        else:
            self.fallback_projection = None
    
    def forward(
        self,
        x_numerical: Optional[torch.Tensor] = None,
        x_categorical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_numerical: Numerical features [batch_size, num_numerical]
            x_categorical: Categorical features [batch_size, num_categorical]
            
        Returns:
            Encoded features [batch_size, output_dim]
        """
        encoded_parts = []
        
        # Encode numerical features
        if self.numerical_encoder is not None and x_numerical is not None:
            numerical_encoded = self.numerical_encoder(x_numerical)
            encoded_parts.append(numerical_encoded)
        
        # Encode categorical features
        if self.categorical_encoder is not None and x_categorical is not None:
            categorical_encoded = self.categorical_encoder(x_categorical)
            encoded_parts.append(categorical_encoded)
        
        # Combine encodings
        if len(encoded_parts) > 0:
            return torch.cat(encoded_parts, dim=-1)
        elif self.fallback_projection is not None:
            # Fallback for when no features are provided
            batch_size = x_numerical.shape[0] if x_numerical is not None else 1
            device = x_numerical.device if x_numerical is not None else torch.device('cpu')
            dummy = torch.zeros(batch_size, 1, device=device)
            return self.fallback_projection(dummy)
        else:
            raise ValueError("No features provided and no fallback available")


class NumericalEncoder(nn.Module):
    """
    Encoder for numerical features with normalization and projection.
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        
        self.num_features = num_features
        
        # Batch normalization for input
        self.input_norm = nn.BatchNorm1d(num_features) if use_batch_norm else nn.Identity()
        
        # Multi-layer projection
        self.projection = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        
        # Feature-wise attention
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Numerical features [batch_size, num_features]
            
        Returns:
            Encoded features [batch_size, d_model]
        """
        # Handle NaN values
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Batch normalization
        if x.shape[0] > 1:
            x = self.input_norm(x)
        
        # Feature-wise attention
        attention = self.feature_attention(x)
        x = x * attention
        
        # Project to model dimension
        return self.projection(x)


class CategoricalEncoder(nn.Module):
    """
    Encoder for categorical features using embeddings.
    """
    
    def __init__(
        self,
        cardinalities: List[int],
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cardinalities = cardinalities
        self.embedding_dim = embedding_dim
        
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, embedding_dim, padding_idx=0)  # +1 for unknown
            for cardinality in cardinalities
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim * len(cardinalities))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Categorical features [batch_size, num_categorical]
            
        Returns:
            Encoded features [batch_size, num_categorical * embedding_dim]
        """
        # Ensure indices are within valid range
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            # Clamp indices to valid range
            indices = x[:, i].long()
            indices = torch.clamp(indices, 0, self.cardinalities[i])
            embedded.append(embedding(indices))
        
        # Concatenate all embeddings
        x = torch.cat(embedded, dim=-1)
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x
