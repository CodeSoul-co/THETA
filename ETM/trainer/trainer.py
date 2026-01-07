"""
ETM Trainer

Handles training loop, evaluation, and checkpoint management.
Follows the rule: train mode and eval mode must be synchronized.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class TrainerConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    optimizer: str = 'adam'           # adam, adamw
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    clip_grad: float = 2.0            # Gradient clipping
    
    # Learning rate scheduling
    lr_scheduler: str = 'plateau'     # plateau, cosine, none
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Logging
    log_interval: int = 10            # Log every N batches
    eval_interval: int = 1            # Evaluate every N epochs
    
    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    
    # Warm-up freezing for word embeddings (rho matrix)
    freeze_word_embeddings_epochs: int = 20  # Freeze rho for first N epochs
    word_embedding_lr: float = 1e-5          # Very small LR for rho after unfreezing
    
    # KL divergence weight scheduling
    kl_weight_start: float = 0.1      # Initial KL weight
    kl_weight_end: float = 1.0        # Final KL weight
    kl_weight_epochs: int = 30        # Epochs to anneal KL weight


class ETMTrainer:
    """
    Trainer for ETM model.
    
    Handles:
    - Training loop with proper logging
    - Validation and early stopping
    - Checkpoint saving (versioned, no overwrite)
    - Metric tracking
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None,
        device: Optional[torch.device] = None,
        result_dir: str = "/root/autodl-tmp/result",
        checkpoint_dir: str = "/root/autodl-tmp/ETM/checkpoints",
        dev_mode: bool = False
    ):
        """
        Initialize trainer.
        
        Args:
            model: ETM model
            config: Training configuration
            device: Device to use
            result_dir: Directory for results (versioned)
            checkpoint_dir: Directory for checkpoints
            dev_mode: Print debug information
        """
        self.model = model
        self.config = config or TrainerConfig()
        self.dev_mode = dev_mode
        self.result_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Create directories
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history: List[Dict] = []
        
        # Timestamp for this training run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.dev_mode:
            print(f"[DEV] Trainer initialized:")
            print(f"[DEV]   device={self.device}")
            print(f"[DEV]   optimizer={self.config.optimizer}")
            print(f"[DEV]   lr={self.config.learning_rate}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config"""
        if self.config.optimizer == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                min_lr=self.config.min_lr
            )
        elif self.config.lr_scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        else:
            return None
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            doc_emb = batch['doc_embedding'].to(self.device)
            bow = batch['bow'].to(self.device)
            
            # Calculate KL weight for annealing (linear schedule)
            if hasattr(self.config, 'kl_weight_start') and hasattr(self.config, 'kl_weight_end') and hasattr(self.config, 'kl_weight_epochs'):
                if self.current_epoch < self.config.kl_weight_epochs:
                    kl_weight = self.config.kl_weight_start + (self.config.kl_weight_end - self.config.kl_weight_start) * \
                                (self.current_epoch / self.config.kl_weight_epochs)
                else:
                    kl_weight = self.config.kl_weight_end
            else:
                kl_weight = 1.0
                
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(doc_emb, bow, compute_loss=True, kl_weight=kl_weight)
            
            # Backward pass
            loss = output['total_loss']
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad
                )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += output['recon_loss'].item()
            total_kl_theta_loss += output['kl_theta_loss'].item() if 'kl_theta_loss' in output else 0.0
            total_kl_bow_loss += output['kl_bow_loss'].item() if 'kl_bow_loss' in output else 0.0
            total_kl_loss += output['kl_loss'].item()
            num_batches += 1
            
            # Logging
            if self.dev_mode and batch_idx % self.config.log_interval == 0:
                print(f"[DEV] Batch {batch_idx}/{len(train_loader)}: "
                      f"loss={loss.item():.4f}, "
                      f"recon={output['recon_loss'].item():.4f}, "
                      f"kl_theta={output.get('kl_theta_loss', 0.0):.4f}, "
                      f"kl_bow={output.get('kl_bow_loss', 0.0):.4f}, "
                      f"kl_total={output['kl_loss'].item():.4f}")
        
        epoch_time = time.time() - epoch_start
        
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_theta_loss': total_kl_theta_loss / num_batches if 'total_kl_theta_loss' in locals() else 0.0,
            'kl_bow_loss': total_kl_bow_loss / num_batches if 'total_kl_bow_loss' in locals() else 0.0,
            'kl_loss': total_kl_loss / num_batches,
            'time': epoch_time,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_theta_loss = 0.0
        total_kl_bow_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        for batch in data_loader:
            doc_emb = batch['doc_embedding'].to(self.device)
            bow = batch['bow'].to(self.device)
            
            output = self.model(doc_emb, bow, compute_loss=True)
            
            total_loss += output['total_loss'].item()
            total_recon_loss += output['recon_loss'].item()
            total_kl_theta_loss += output['kl_theta_loss'].item() if 'kl_theta_loss' in output else 0.0
            total_kl_bow_loss += output['kl_bow_loss'].item() if 'kl_bow_loss' in output else 0.0
            total_kl_loss += output['kl_loss'].item()
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_theta_loss': total_kl_theta_loss / num_batches if total_kl_theta_loss > 0 else 0.0,
            'kl_bow_loss': total_kl_bow_loss / num_batches if total_kl_bow_loss > 0 else 0.0,
            'kl_loss': total_kl_loss / num_batches
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        dataset_name: str = "unknown"
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Training history and final metrics
        """
        print("=" * 70)
        print(f"Training ETM on {dataset_name}")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 70)
        
        # Check if model has word embeddings (rho)
        has_word_embeddings = (
            hasattr(self.model, 'decoder') and 
            hasattr(self.model.decoder, 'rho')
        )
        
        if has_word_embeddings:
            print(f"\n[Warm-up Freezing] Detected word embeddings (rho)")
            print(f"  First {self.config.freeze_word_embeddings_epochs} epochs: rho frozen")
            print(f"  Later epochs: rho unfrozen (lr={self.config.word_embedding_lr})")
            print("=" * 70)
        
        training_start = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Warm-up freezing logic
            if has_word_embeddings:
                if epoch < self.config.freeze_word_embeddings_epochs:
                    # Stage 1: Freeze rho
                    if self.model.decoder.rho.requires_grad:
                        self.model.decoder.rho.requires_grad = False
                        if epoch == 0:
                            print(f"\n[Epoch {epoch+1}] Rho frozen (warm-up phase)")
                else:
                    # Stage 2: Unfreeze rho
                    if not self.model.decoder.rho.requires_grad:
                        print(f"\n[Epoch {epoch+1}] Unfreezing rho, starting fine-tuning")
                        self.model.decoder.rho.requires_grad = True
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = None
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau) and val_metrics:
                    self.scheduler.step(val_metrics['loss'])
                elif isinstance(self.scheduler, CosineAnnealingLR):
                    self.scheduler.step()
            
            # Log epoch results
            self._log_epoch(epoch, train_metrics, val_metrics)
            
            # Save history
            history_entry = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            }
            self.training_history.append(history_entry)
            
            # Check for improvement
            if val_metrics is not None:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.config.save_best:
                        self._save_checkpoint(dataset_name, 'best')
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.config.early_stopping:
                    if self.epochs_without_improvement >= self.config.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
        
        # Save final model
        if self.config.save_last:
            self._save_checkpoint(dataset_name, 'last')
        
        total_time = time.time() - training_start
        
        # Save training history
        self._save_history(dataset_name)
        
        print("\n" + "=" * 70)
        print(f"Training completed in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 70)
        
        return {
            'history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time,
            'final_epoch': self.current_epoch
        }
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Optional[Dict]
    ):
        """Log epoch results"""
        msg = (f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
               f"Time: {train_metrics['time']:.1f}s | "
               f"LR: {train_metrics['lr']:.6f} | "
               f"Train Loss: {train_metrics['loss']:.4f} "
               f"(recon: {train_metrics['recon_loss']:.4f}, "
               f"kl_theta: {train_metrics.get('kl_theta_loss', 0.0):.4f}, "
               f"kl_bow: {train_metrics.get('kl_bow_loss', 0.0):.4f}, "
               f"kl_total: {train_metrics['kl_loss']:.4f})")
        
        if val_metrics is not None:
            msg += (f" | Val Loss: {val_metrics['loss']:.4f} "
                   f"(recon: {val_metrics['recon_loss']:.4f}, "
                   f"kl_total: {val_metrics['kl_loss']:.4f})")
        
        print(msg)
    
    def _save_checkpoint(self, dataset_name: str, tag: str):
        """Save model checkpoint to dataset-specific directory"""
        # Create dataset-specific checkpoint directory
        dataset_ckpt_dir = os.path.join(self.checkpoint_dir, dataset_name)
        os.makedirs(dataset_ckpt_dir, exist_ok=True)
        
        filename = f"etm_{tag}_{self.run_timestamp}.pt"
        path = os.path.join(dataset_ckpt_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'num_topics': self.model.num_topics,
                'doc_embedding_dim': self.model.doc_embedding_dim,
                'word_embedding_dim': self.model.word_embedding_dim,
                'hidden_dim': self.model.hidden_dim
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def _save_history(self, dataset_name: str):
        """Save training history to dataset-specific result directory"""
        # Create dataset-specific result directory
        dataset_result_dir = os.path.join(self.result_dir, dataset_name, "etm")
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        filename = f"training_history_{self.run_timestamp}.json"
        path = os.path.join(dataset_result_dir, filename)
        
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Saved history: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    @torch.no_grad()
    def get_document_topics(
        self,
        data_loader: DataLoader
    ) -> np.ndarray:
        """
        Get topic distributions for all documents.
        
        Args:
            data_loader: Data loader
            
        Returns:
            theta: Document-topic matrix (N, K)
        """
        self.model.eval()
        
        all_theta = []
        for batch in data_loader:
            doc_emb = batch['doc_embedding'].to(self.device)
            theta = self.model.get_theta(doc_emb)
            all_theta.append(theta.cpu().numpy())
        
        return np.vstack(all_theta)
    
    @torch.no_grad()
    def get_topic_words(
        self,
        vocab: List[str],
        top_k: int = 10
    ) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """
        Get top words for each topic.
        
        Args:
            vocab: Vocabulary list
            top_k: Number of top words
            
        Returns:
            List of (topic_idx, [(word, prob), ...])
        """
        self.model.eval()
        return self.model.get_topic_words(top_k, vocab)
