#!/usr/bin/env python3
"""
Result Directory Migration Script

Migrates old result directory structure to new standardized structure:

Old Structure:
    result/{user_id}/{dataset}/models/{model_name}/exp_{timestamp}/
        ├── visualization/
        │   ├── en/
        │   │   ├── global/
        │   │   └── topics/
        │   └── global/  (duplicate)
        └── {model_name}/
            └── theta_k{K}.npy, beta_k{K}.npy

New Structure:
    result/{user_id}/{dataset}/{model_name}/{task_name}/
        ├── model/              # Model parameters
        │   ├── theta_k{K}.npy
        │   ├── beta_k{K}.npy
        │   └── config.json
        ├── metrics/            # Evaluation metrics
        │   └── metrics_k{K}.json
        ├── en/                 # English visualizations
        │   ├── global/
        │   └── topic/
        └── cn/                 # Chinese visualizations
            ├── global/
            └── topic/

Usage:
    python migrate_result_structure.py --user_id test_user --dataset test_data
    python migrate_result_structure.py --user_id test_user --dataset test_data --dry_run
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASE_RESULT, get_result_path


def migrate_model_experiment(old_exp_dir: Path, new_task_dir: Path, dry_run: bool = False):
    """Migrate a single model experiment directory."""
    print(f"\n  Migrating: {old_exp_dir.name}")
    
    if not dry_run:
        new_task_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Migrate model parameters
    model_subdir = None
    for subdir in old_exp_dir.iterdir():
        if subdir.is_dir() and subdir.name not in ('visualization', 'global'):
            # This is likely the model subdirectory (e.g., lda/, ctm/, etc.)
            model_subdir = subdir
            break
    
    if model_subdir:
        new_model_dir = new_task_dir / 'model'
        if not dry_run:
            new_model_dir.mkdir(parents=True, exist_ok=True)
        
        for f in model_subdir.glob('*.npy'):
            dst = new_model_dir / f.name
            print(f"    {f.name} -> model/")
            if not dry_run:
                shutil.copy2(f, dst)
        
        for f in model_subdir.glob('*.json'):
            dst = new_model_dir / f.name
            print(f"    {f.name} -> model/")
            if not dry_run:
                shutil.copy2(f, dst)
    
    # 2. Migrate config.json from root
    config_file = old_exp_dir / 'config.json'
    if config_file.exists():
        dst = new_task_dir / 'config.json'
        print(f"    config.json -> ./")
        if not dry_run:
            shutil.copy2(config_file, dst)
    
    # 3. Migrate metrics
    for metrics_file in old_exp_dir.glob('metrics_k*.json'):
        new_metrics_dir = new_task_dir / 'metrics'
        if not dry_run:
            new_metrics_dir.mkdir(parents=True, exist_ok=True)
        dst = new_metrics_dir / metrics_file.name
        print(f"    {metrics_file.name} -> metrics/")
        if not dry_run:
            shutil.copy2(metrics_file, dst)
    
    # 4. Migrate visualizations
    old_viz_dir = old_exp_dir / 'visualization'
    if old_viz_dir.exists():
        # Check for language subdirectories
        for lang_dir in ['en', 'zh']:
            old_lang_dir = old_viz_dir / lang_dir
            if old_lang_dir.exists():
                # Map 'zh' to 'cn' for new structure
                new_lang = 'cn' if lang_dir == 'zh' else lang_dir
                new_lang_dir = new_task_dir / new_lang
                
                if not dry_run:
                    new_lang_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy global directory
                old_global = old_lang_dir / 'global'
                if old_global.exists():
                    new_global = new_lang_dir / 'global'
                    print(f"    visualization/{lang_dir}/global/ -> {new_lang}/global/")
                    if not dry_run:
                        if new_global.exists():
                            shutil.rmtree(new_global)
                        shutil.copytree(old_global, new_global)
                
                # Copy topics directory (rename to topic)
                old_topics = old_lang_dir / 'topics'
                if old_topics.exists():
                    new_topic = new_lang_dir / 'topic'
                    print(f"    visualization/{lang_dir}/topics/ -> {new_lang}/topic/")
                    if not dry_run:
                        if new_topic.exists():
                            shutil.rmtree(new_topic)
                        shutil.copytree(old_topics, new_topic)
        
        # Also check for README.md in visualization root
        readme = old_viz_dir / 'README.md'
        if readme.exists():
            dst = new_task_dir / 'README.md'
            print(f"    visualization/README.md -> ./")
            if not dry_run:
                shutil.copy2(readme, dst)


def migrate_user_dataset(user_id: str, dataset: str, dry_run: bool = False):
    """Migrate all models for a user/dataset combination."""
    old_base = BASE_RESULT / user_id / dataset / 'models'
    
    if not old_base.exists():
        print(f"Old structure not found: {old_base}")
        return
    
    print(f"\n{'='*70}")
    print(f"Migrating: {user_id}/{dataset}")
    print(f"Old path: {old_base}")
    print(f"{'='*70}")
    
    for model_dir in old_base.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        print(f"\nModel: {model_name}")
        
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith('exp_'):
                continue
            
            task_name = exp_dir.name
            new_task_dir = get_result_path(user_id, dataset, model_name, task_name)
            
            migrate_model_experiment(exp_dir, new_task_dir, dry_run)
    
    # After migration, optionally remove old 'models' directory
    if not dry_run:
        print(f"\n[INFO] Migration complete. Old 'models/' directory preserved.")
        print(f"[INFO] To remove old structure, run:")
        print(f"       rm -rf {old_base}")


def cleanup_old_structure(user_id: str, dataset: str, confirm: bool = False):
    """Remove old 'models/' directory after migration."""
    old_base = BASE_RESULT / user_id / dataset / 'models'
    
    if not old_base.exists():
        print(f"Old structure already removed: {old_base}")
        return
    
    if not confirm:
        print(f"Would remove: {old_base}")
        print("Run with --confirm to actually delete")
        return
    
    shutil.rmtree(old_base)
    print(f"Removed: {old_base}")


def main():
    parser = argparse.ArgumentParser(description='Migrate result directory structure')
    parser.add_argument('--user_id', type=str, required=True, help='User ID')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--dry_run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--cleanup', action='store_true', help='Remove old structure after migration')
    parser.add_argument('--confirm', action='store_true', help='Confirm cleanup deletion')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("[DRY RUN] No changes will be made")
    
    migrate_user_dataset(args.user_id, args.dataset, args.dry_run)
    
    if args.cleanup:
        cleanup_old_structure(args.user_id, args.dataset, args.confirm)


if __name__ == '__main__':
    main()
