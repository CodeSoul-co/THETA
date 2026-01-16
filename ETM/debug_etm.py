#!/usr/bin/env python
"""
Debug script to verify ETM model is working correctly.
Tests on socialTwitter dataset with detailed monitoring.
Only uses GPU 1 (CUDA_VISIBLE_DEVICES=1).
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Force GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def debug_model_computation():
    """Verify ETM computation is correct"""
    print("\n" + "=" * 60)
    print("DEBUG: Verifying ETM Model Computation")
    print("=" * 60)
    
    from model.etm import ETM
    from model.encoder import ETMEncoder
    from model.decoder import ETMDecoder
    
    # Test parameters
    N, D, V, K = 100, 1024, 5000, 20
    
    print(f"Test parameters: N={N}, D={D}, V={V}, K={K}")
    
    # Create model
    model = ETM(
        vocab_size=V,
        num_topics=K,
        doc_embedding_dim=D,
        word_embedding_dim=D,
        hidden_dim=512,
        dev_mode=True
    )
    
    # Create test data
    doc_embeddings = torch.randn(N, D)
    bow_targets = torch.abs(torch.randn(N, V))  # Positive BOW counts
    bow_targets = bow_targets / bow_targets.sum(dim=1, keepdim=True) * 100  # Normalize
    
    # Forward pass
    model.train()
    output = model(doc_embeddings, bow_targets, kl_weight=1.0)
    
    print("\n--- Output Shapes ---")
    print(f"theta shape: {output['theta'].shape}")  # Should be (N, K)
    print(f"beta shape: {output['beta'].shape}")    # Should be (K, V)
    print(f"word_dist shape: {output['word_dist'].shape}")  # Should be (N, V)
    
    print("\n--- Theta (Topic Distribution) Statistics ---")
    theta = output['theta'].detach()
    print(f"theta sum per doc (should be 1.0): {theta.sum(dim=1).mean():.6f}")
    print(f"theta mean: {theta.mean():.6f}")
    print(f"theta std: {theta.std():.6f}")
    print(f"theta min: {theta.min():.6f}")
    print(f"theta max: {theta.max():.6f}")
    
    # Check if theta is diverse (not all same)
    theta_per_topic = theta.mean(dim=0)
    print(f"theta per topic std (should be > 0.01): {theta_per_topic.std():.6f}")
    
    print("\n--- Beta (Topic-Word Distribution) Statistics ---")
    beta = output['beta'].detach()
    print(f"beta sum per topic (should be 1.0): {beta.sum(dim=1).mean():.6f}")
    print(f"beta mean: {beta.mean():.6f}")
    print(f"beta std: {beta.std():.6f}")
    print(f"beta min: {beta.min():.6f}")
    print(f"beta max: {beta.max():.6f}")
    
    # Check topic diversity
    topic_similarity = torch.mm(beta, beta.t())
    off_diag = topic_similarity[~torch.eye(K, dtype=bool)]
    print(f"topic similarity (off-diagonal mean, should be < 0.5): {off_diag.mean():.6f}")
    
    print("\n--- Loss Statistics ---")
    print(f"recon_loss: {output['recon_loss'].item():.4f}")
    print(f"kl_theta_loss: {output['kl_theta_loss'].item():.4f}")
    print(f"kl_loss: {output['kl_loss'].item():.4f}")
    print(f"total_loss: {output['total_loss'].item():.4f}")
    
    # Verify loss is reasonable
    if output['total_loss'].item() < 0:
        print("WARNING: Total loss is negative!")
    if output['recon_loss'].item() < 0:
        print("WARNING: Reconstruction loss is negative!")
    
    print("\n--- Gradient Check ---")
    output['total_loss'].backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"WARNING: Zero gradient for {name}")
            elif grad_norm > 100:
                print(f"WARNING: Large gradient for {name}: {grad_norm:.4f}")
    
    print("\nModel computation verification complete!")
    return True


def train_small_test():
    """Train on a small subset to verify training works"""
    print("\n" + "=" * 60)
    print("DEBUG: Training on Small Subset")
    print("=" * 60)
    
    import subprocess
    
    # Run pipeline with dev mode on socialTwitter
    cmd = [
        sys.executable, "main.py", "pipeline",
        "--dataset", "socialTwitter",
        "--mode", "zero_shot",
        "--num_topics", "10",  # Fewer topics for faster testing
        "--epochs", "30",
        "--batch_size", "64",
        "--vocab_size", "3000",  # Smaller vocab for faster testing
        "--learning_rate", "0.0005",
        "--kl_warmup", "15",
        "--patience", "10",
        "--dev"  # Enable dev mode for more logging
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(
        cmd,
        cwd="/root/autodl-tmp/ETM"
    )
    
    return result.returncode == 0


def check_topic_quality():
    """Check topic quality metrics after training"""
    print("\n" + "=" * 60)
    print("DEBUG: Checking Topic Quality")
    print("=" * 60)
    
    import json
    from pathlib import Path
    
    # Find latest metrics file
    result_dir = Path("/root/autodl-tmp/result/socialTwitter/zero_shot/evaluation")
    if not result_dir.exists():
        print("No evaluation results found yet")
        return False
    
    metrics_files = list(result_dir.glob("metrics_*.json"))
    if not metrics_files:
        print("No metrics files found")
        return False
    
    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading metrics from: {latest_metrics}")
    
    with open(latest_metrics, 'r') as f:
        metrics = json.load(f)
    
    print("\n--- Topic Quality Metrics ---")
    
    # Topic Diversity
    td = metrics.get('topic_diversity_td', 0)
    irbo = metrics.get('topic_diversity_irbo', 0)
    print(f"Topic Diversity (TD): {td:.4f} (target: > 0.7)")
    print(f"Topic Diversity (iRBO): {irbo:.4f} (target: > 0.8)")
    
    # Topic Coherence
    coherence = metrics.get('topic_coherence_avg', 0)
    print(f"Topic Coherence (NPMI): {coherence:.4f} (target: > 0.4)")
    
    # Topic Significance
    significance = metrics.get('topic_significance', [])
    if significance:
        sig_nonzero = sum(1 for s in significance if s > 0.01)
        print(f"Topics with significance > 0.01: {sig_nonzero}/{len(significance)}")
    
    # Quality assessment
    print("\n--- Quality Assessment ---")
    issues = []
    if td < 0.3:
        issues.append("Topic Diversity (TD) is very low")
    if irbo < 0.5:
        issues.append("Topic Diversity (iRBO) is very low")
    if coherence < 0.1:
        issues.append("Topic Coherence is very low")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All metrics are within acceptable range!")
        return True


def main():
    print("=" * 60)
    print("ETM Debug Script")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)
    
    # Step 1: Verify model computation
    try:
        debug_model_computation()
    except Exception as e:
        print(f"ERROR in model computation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Train on small subset
    print("\nStarting training test...")
    success = train_small_test()
    
    if not success:
        print("Training failed!")
        return
    
    # Step 3: Check topic quality
    check_topic_quality()
    
    print("\n" + "=" * 60)
    print(f"Debug complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
