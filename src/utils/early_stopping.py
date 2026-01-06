"""
Early stopping utility for training.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when a monitored metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "min" or "max" - whether to minimize or maximize the metric
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        if mode == "min":
            self.is_better = lambda a, b: a < b - min_delta
            self.best_value = np.inf
        else:
            self.is_better = lambda a, b: a > b + min_delta
            self.best_value = -np.inf
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric_value: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(metric_value, self.best_value):
            self.best_value = metric_value
            self.counter = 0
            if self.verbose:
                logger.info(f"Early stopping: metric improved to {metric_value:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"Early stopping: no improvement for {self.counter}/{self.patience} epochs"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_value = np.inf
        else:
            self.best_value = -np.inf
