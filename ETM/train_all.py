#!/usr/bin/env python
"""
Batch training script for ETM model on all datasets and modes.
Only uses GPU 1 (CUDA_VISIBLE_DEVICES=1).
"""

import os
import sys
import subprocess
from datetime import datetime

# Force GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training configurations
TRAINING_CONFIGS = [
    # socialTwitter
    {"dataset": "socialTwitter", "mode": "zero_shot", "num_topics": 20},
    {"dataset": "socialTwitter", "mode": "supervised", "num_topics": 20},
    
    # hatespeech
    {"dataset": "hatespeech", "mode": "zero_shot", "num_topics": 20},
    {"dataset": "hatespeech", "mode": "supervised", "num_topics": 20},
    {"dataset": "hatespeech", "mode": "unsupervised", "num_topics": 20},
    
    # mental_health
    {"dataset": "mental_health", "mode": "zero_shot", "num_topics": 20},
    {"dataset": "mental_health", "mode": "supervised", "num_topics": 20},
    
    # FCPB
    {"dataset": "FCPB", "mode": "zero_shot", "num_topics": 20},
    
    # germanCoal
    {"dataset": "germanCoal", "mode": "zero_shot", "num_topics": 20},
]

def run_training(config):
    """Run training for a single configuration"""
    dataset = config["dataset"]
    mode = config["mode"]
    num_topics = config["num_topics"]
    
    print("=" * 70)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: {dataset} - {mode}")
    print("=" * 70)
    
    cmd = [
        sys.executable, "main.py", "pipeline",
        "--dataset", dataset,
        "--mode", mode,
        "--num_topics", str(num_topics),
        "--epochs", "100",
        "--batch_size", "64",
        "--vocab_size", "5000",
        "--learning_rate", "0.0005",
        "--kl_warmup", "30",
        "--patience", "15"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd="/root/autodl-tmp/ETM",
            check=True
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed: {dataset} - {mode}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] FAILED: {dataset} - {mode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {dataset} - {mode}")
        print(f"Exception: {e}")
        return False


def main():
    print("=" * 70)
    print("ETM Batch Training Script")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Total configurations: {len(TRAINING_CONFIGS)}")
    print("=" * 70)
    
    results = []
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n[{i}/{len(TRAINING_CONFIGS)}] Processing...")
        success = run_training(config)
        results.append({
            "dataset": config["dataset"],
            "mode": config["mode"],
            "success": success
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"Total: {len(results)}, Success: {success_count}, Failed: {fail_count}")
    print()
    
    for r in results:
        status = "SUCCESS" if r["success"] else "FAILED"
        print(f"  {r['dataset']:20s} {r['mode']:15s} {status}")
    
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
