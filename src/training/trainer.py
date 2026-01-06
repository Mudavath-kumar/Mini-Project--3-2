"""
Trainer for MambaTab model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, List, Tuple
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json

from ..models.mamba_tab import MambaTab
from ..utils.metrics import compute_metrics
from ..utils.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for MambaTab model.
    """
    
    def __init__(
        self,
        model: MambaTab,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "auto",
        task_type: str = "classification",
        num_classes: int = 2,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        mixed_precision: bool = True,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = "mambatab",
    ):
        """
        Initialize Trainer.
        
        Args:
            model: MambaTab model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to use ("auto", "cuda", "cpu", "mps")
            task_type: "classification" or "regression"
            num_classes: Number of classes
            gradient_clip: Gradient clipping value
            accumulation_steps: Gradient accumulation steps
            mixed_precision: Whether to use mixed precision training
            checkpoint_dir: Directory for saving checkpoints
            experiment_name: Name of the experiment
        """
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_type = task_type
        self.num_classes = num_classes
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        self.experiment_name = experiment_name
        
        # Optimizer
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        # Scheduler
        self.scheduler = scheduler
        
        # Loss function
        if criterion is not None:
            self.criterion = criterion
        elif task_type == "classification":
            if num_classes == 2:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Mixed precision
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Checkpoint directory
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": [],
            "lr": [],
        }
        
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_metric = 0.0
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get data
            x_numerical = batch.get("numerical")
            x_categorical = batch.get("categorical")
            labels = batch["labels"]
            
            # Move to device
            if x_numerical is not None:
                x_numerical = x_numerical.to(self.device)
            if x_categorical is not None:
                x_categorical = x_categorical.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(x_numerical, x_categorical)
                    logits = output["logits"]
                    loss = self._compute_loss(logits, labels)
                    loss = loss / self.accumulation_steps
            else:
                output = self.model(x_numerical, x_categorical)
                logits = output["logits"]
                loss = self._compute_loss(logits, labels)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            # Collect predictions
            with torch.no_grad():
                preds = self._get_predictions(logits)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
            
            pbar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})
        
        # Compute metrics
        avg_loss = total_loss / len(self.train_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_preds, all_labels, self.task_type, self.num_classes)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model.
        
        Args:
            data_loader: Data loader to evaluate on (default: validation loader)
            
        Returns:
            Tuple of (loss, metrics)
        """
        data_loader = data_loader or self.val_loader
        if data_loader is None:
            return 0.0, {}
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in data_loader:
            x_numerical = batch.get("numerical")
            x_categorical = batch.get("categorical")
            labels = batch["labels"]
            
            if x_numerical is not None:
                x_numerical = x_numerical.to(self.device)
            if x_categorical is not None:
                x_categorical = x_categorical.to(self.device)
            labels = labels.to(self.device)
            
            output = self.model(x_numerical, x_categorical)
            logits = output["logits"]
            loss = self._compute_loss(logits, labels)
            
            total_loss += loss.item()
            
            preds = self._get_predictions(logits)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
        
        avg_loss = total_loss / len(data_loader)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics(all_preds, all_labels, self.task_type, self.num_classes)
        
        return avg_loss, metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: int = 10,
        early_stopping_metric: str = "val_loss",
        early_stopping_mode: str = "min",
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            early_stopping_metric: Metric to monitor for early stopping
            early_stopping_mode: "min" or "max"
            save_best: Whether to save the best model
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            verbose=verbose,
        )
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_metrics"].append(train_metrics)
            
            # Evaluate
            if self.val_loader is not None:
                val_loss, val_metrics = self.evaluate()
                self.history["val_loss"].append(val_loss)
                self.history["val_metrics"].append(val_metrics)
            else:
                val_loss, val_metrics = train_loss, train_metrics
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Logging
            if verbose:
                metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"LR: {current_lr:.6f}, {metric_str}"
                )
            
            # Get early stopping metric value
            if early_stopping_metric == "val_loss":
                metric_value = val_loss
            elif early_stopping_metric in val_metrics:
                metric_value = val_metrics[early_stopping_metric]
            else:
                metric_value = val_loss
            
            # Save best model
            if save_best and self.checkpoint_dir is not None:
                if early_stopping_mode == "min" and metric_value < self.best_val_loss:
                    self.best_val_loss = metric_value
                    self.save_checkpoint("best_model.pt")
                elif early_stopping_mode == "max" and metric_value > self.best_val_metric:
                    self.best_val_metric = metric_value
                    self.save_checkpoint("best_model.pt")
            
            # Early stopping check
            if early_stopping(metric_value):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        return self.history
    
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss based on task type."""
        if self.task_type == "classification":
            if self.num_classes == 2:
                return self.criterion(logits.squeeze(-1), labels.float())
            else:
                return self.criterion(logits, labels.long())
        else:
            return self.criterion(logits.squeeze(-1), labels.float())
    
    def _get_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """Get predictions from logits."""
        if self.task_type == "classification":
            if self.num_classes == 2:
                return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            else:
                return torch.argmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_loss": self.best_val_loss,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
        logger.info(f"Saved checkpoint to {self.checkpoint_dir / filename}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.history = checkpoint["history"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {filepath}")
