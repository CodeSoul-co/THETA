#!/usr/bin/env python3
"""
ETM Unified Visualization Runner
Unified visualization script - Generate all visualizations after training

Usage:
    python run_visualization.py --result_dir /path/to/result --dataset socialTwitter --mode zero_shot
    
    # Or use the convenience function:
    from visualization.run_visualization import run_all_visualizations
    run_all_visualizations(result_dir, dataset, mode)
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_file(directory, pattern):
    """Find the latest file matching pattern in directory."""
    from glob import glob
    files = glob(str(Path(directory) / pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_visualization_data(result_dir, dataset, mode, model_size=None, model_exp=None):
    """
    Load all data needed for visualization from result directory.
    
    Supports flexible directory structures:
    - New structure: result/{model_size}/{dataset}/models/exp_*/
    - Legacy structure: result/{dataset}/{mode}/
    - Mixed structure: model in one location, data in another
    
    Args:
        result_dir: Base result directory (e.g., ./result)
        dataset: Dataset name (e.g., socialTwitter)
        mode: Training mode (zero_shot, supervised, unsupervised)
        model_size: Optional model size subdirectory (e.g., 0.6B)
        model_exp: Optional specific experiment ID (e.g., exp_20260326_141527)
    
    Returns:
        dict with all visualization data
    """
    from scipy import sparse
    
    result_dir = Path(result_dir)
    result_base = result_dir.parent if model_size and model_size in str(result_dir) else result_dir
    
    # Auto-discover directories based on actual structure
    model_dir = None
    topic_words_dir = None
    bow_dir = None
    evaluation_dir = None
    
    # Strategy 1: Find model files (theta_*.npy, beta_*.npy) by searching
    def find_model_dir():
        """Search for directory containing theta_*.npy files"""
        # Unified structure: result/{model_size}/{dataset}/models/exp_*/theta/ or model/
        search_paths = [
            result_dir / dataset / 'models' if model_size and model_size in str(result_dir) else None,
            result_base / model_size / dataset / 'models' if model_size else None,
            result_base / dataset / 'models',
            result_dir / dataset,
        ]
        search_paths = [p for p in search_paths if p is not None]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            # Find latest exp_* directory with model files
            exp_dirs = sorted(search_path.glob('exp_*'), key=os.path.getmtime, reverse=True)
            for exp_dir in exp_dirs:
                # Try both theta/ and model/ subdirectories
                for subdir in ['theta', 'model']:
                    model_path = exp_dir / subdir
                    if model_path.exists() and list(model_path.glob('theta_*.npy')):
                        if model_exp is None or exp_dir.name == model_exp:
                            return model_path, exp_dir
        return None, None
    
    # Strategy 2: Find topic_words directory
    def find_topic_words_dir():
        """Search for directory containing topic_words_*.json files"""
        search_paths = [
            result_dir / dataset / 'models' if model_size and model_size in str(result_dir) else result_base / model_size / dataset / 'models' if model_size else None,
            result_base / model_size / dataset / 'models' if model_size else None,
            result_dir / 'models',
        ]
        search_paths = [p for p in search_paths if p is not None]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            exp_dirs = sorted(search_path.glob('exp_*'), key=os.path.getmtime, reverse=True)
            for exp_dir in exp_dirs:
                tw_path = exp_dir / 'topic_words'
                if tw_path.exists() and list(tw_path.glob('topic_words_*.json')):
                    if model_exp is None or exp_dir.name == model_exp:
                        return tw_path
        return None
    
    # Strategy 3: Find BOW directory
    def find_bow_dir():
        """Search for directory containing bow_matrix.npy or vocab.json"""
        search_paths = [
            result_dir / dataset / 'data' if model_size and model_size in str(result_dir) else result_base / model_size / dataset / 'data' if model_size else None,
            result_base / model_size / dataset / 'data' if model_size else None,
            result_dir / 'data',
            result_base / dataset / 'bow',
        ]
        search_paths = [p for p in search_paths if p is not None]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            # Check for exp_* subdirectories
            exp_dirs = sorted(search_path.glob('exp_*'), key=os.path.getmtime, reverse=True)
            for exp_dir in exp_dirs:
                bow_path = exp_dir / 'bow'
                if bow_path.exists() and (list(bow_path.glob('vocab.*')) or list(bow_path.glob('bow_matrix.*'))):
                    return bow_path
            # Check direct bow directory
            if (search_path / 'bow').exists():
                return search_path / 'bow'
        return None
    
    # Execute discovery
    model_dir, exp_dir = find_model_dir()
    topic_words_dir = find_topic_words_dir()
    bow_dir = find_bow_dir()
    
    if exp_dir:
        evaluation_dir = exp_dir / 'evaluation'
    
    # Fallback to legacy structure if discovery failed
    if model_dir is None:
        # Try legacy paths
        if model_size:
            base_dir = result_dir / model_size / dataset / mode if model_size not in str(result_dir) else result_dir / dataset / mode
        else:
            base_dir = result_dir / dataset / mode
        
        if (base_dir / 'model').exists():
            model_dir = base_dir / 'model'
            evaluation_dir = base_dir / 'evaluation'
            topic_words_dir = base_dir / 'topic_words'
            bow_dir = base_dir / 'bow' if (base_dir / 'bow').exists() else None
    
    if model_dir is None:
        raise FileNotFoundError(f"Could not find model directory with theta_*.npy files. Searched in: {result_dir}, {result_base}")
    
    # Legacy config-based BOW resolution as final fallback
    if bow_dir is None:
        config_file = model_dir.parent / 'config.json' if model_dir else None
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    train_config = json.load(f)
                data_exp = train_config.get('data_exp', '')
                ms = train_config.get('model_size', model_size or '')
                ds = train_config.get('dataset', dataset)
                if data_exp and data_exp != 'legacy':
                    # New exp structure: find bow in data exp dir
                    candidate = result_dir.parent.parent / 'data' / data_exp / 'bow'
                    if not candidate.exists() and ms:
                        result_base_env = Path(os.environ.get('RESULT_DIR', 'result'))
                        candidate = result_base_env / ms / ds / 'data' / data_exp / 'bow'
                    if candidate.exists():
                        bow_dir = candidate
            except Exception:
                pass
    
    print(f"\n{'='*60}")
    print(f"Loading visualization data")
    print(f"{'='*60}")
    print(f"Model directory: {model_dir}")
    if topic_words_dir:
        print(f"Topic words directory: {topic_words_dir}")
    if bow_dir:
        print(f"BOW directory: {bow_dir}")
    if evaluation_dir:
        print(f"Evaluation directory: {evaluation_dir}")
    
    data = {}
    
    # Load theta (document-topic distribution)
    theta_file = find_latest_file(model_dir, "theta_*.npy")
    if theta_file:
        data['theta'] = np.load(theta_file)
        print(f"✓ Loaded theta: {data['theta'].shape}")
    else:
        raise FileNotFoundError(f"theta not found in {model_dir}")
    
    # Load beta (topic-word distribution)
    beta_file = find_latest_file(model_dir, "beta_*.npy")
    if beta_file:
        data['beta'] = np.load(beta_file)
        print(f"✓ Loaded beta: {data['beta'].shape}")
    else:
        raise FileNotFoundError(f"beta not found in {model_dir}")
    
    # Load topic embeddings
    emb_file = find_latest_file(model_dir, "topic_embeddings_*.npy")
    if emb_file:
        data['topic_embeddings'] = np.load(emb_file)
        print(f"✓ Loaded topic_embeddings: {data['topic_embeddings'].shape}")
    
    # Load topic words - try topic_words_dir first, then model_dir
    words_file = find_latest_file(topic_words_dir, "topic_words_*.json")
    if not words_file:
        words_file = find_latest_file(model_dir, "topic_words_*.json")
    
    if words_file:
        with open(words_file, 'r', encoding='utf-8') as f:
            topic_words_raw = json.load(f)
        
        # Convert to standard format: [(topic_id, [(word, weight), ...]), ...]
        if isinstance(topic_words_raw, list):
            # Format: [[topic_id, [[word, weight], ...]], ...]
            data['topic_words'] = [
                (item[0], [(w[0], w[1]) for w in item[1]])
                for item in topic_words_raw
            ]
        elif isinstance(topic_words_raw, dict):
            # Format: {"0": [[word, weight], ...], ...}
            data['topic_words'] = [
                (int(k), [(w[0], w[1]) for w in v])
                for k, v in sorted(topic_words_raw.items(), key=lambda x: int(x[0]))
            ]
        print(f"✓ Loaded topic_words: {len(data['topic_words'])} topics")
    else:
        # Generate from beta
        n_topics = data['beta'].shape[0]
        data['topic_words'] = []
        for i in range(n_topics):
            top_indices = np.argsort(data['beta'][i])[-20:][::-1]
            words = [(f"word_{idx}", float(data['beta'][i, idx])) for idx in top_indices]
            data['topic_words'].append((i, words))
        print(f"⚠ Generated topic_words from beta: {len(data['topic_words'])} topics")
    
    # Load training history
    history_file = find_latest_file(model_dir, "training_history_*.json")
    if history_file:
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"✓ Loaded training_history: {len(data['training_history'].get('train_loss', []))} epochs")
    
    # Load evaluation metrics
    metrics_file = find_latest_file(evaluation_dir, "metrics_*.json")
    if metrics_file:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data['metrics'] = json.load(f)
        print(f"✓ Loaded metrics")
    
    # Load vocab
    vocab_file = bow_dir / 'vocab.txt'
    if vocab_file.exists():
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data['vocab'] = [line.strip() for line in f.readlines()]
        print(f"✓ Loaded vocab: {len(data['vocab'])} words")
    else:
        # Generate placeholder vocab
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"⚠ Generated placeholder vocab: {len(data['vocab'])} words")
    
    # Load BOW matrix (optional)
    if bow_dir:
        bow_file = bow_dir / 'bow_matrix.npy'
        if bow_file.exists():
            data['bow_matrix'] = np.load(bow_file)
            print(f"✓ Loaded bow_matrix: {data['bow_matrix'].shape}")
    
    # Load timestamps (optional) - check in model_dir parent or evaluation_dir
    ts_file = model_dir.parent / 'timestamps.npy' if model_dir else None
    if ts_file and ts_file.exists():
        data['timestamps'] = np.load(ts_file, allow_pickle=True)
        print(f"✓ Loaded timestamps: {len(data['timestamps'])}")
    
    # Load config
    config_file = find_latest_file(model_dir, "config_*.json")
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            data['config'] = json.load(f)
        print(f"✓ Loaded config")
    
    print(f"{'='*60}\n")
    
    return data


def run_all_visualizations(
    result_dir,
    dataset,
    mode,
    model_size=None,
    output_dir=None,
    language='en',
    dpi=300
):
    """
    Run all visualizations for ETM results.
    
    Args:
        result_dir: Base result directory
        dataset: Dataset name
        mode: Training mode
        model_size: Optional model size subdirectory
        output_dir: Output directory for visualizations (default: result_dir/.../visualization)
        language: Language for labels ('en' or 'zh')
        dpi: DPI for saved figures
    
    Returns:
        Path to output directory
    """
    # Load data
    data = load_visualization_data(result_dir, dataset, mode, model_size)
    
    # Determine output directory
    # New structure: result/{model_size}/{dataset}/{language}/
    if output_dir is None:
        result_dir = Path(result_dir)
        result_base = result_dir.parent if model_size and model_size in str(result_dir) else result_dir
        
        # Find latest experiment directory
        theta_base = result_base / dataset / 'models' if model_size and model_size in str(result_dir) else result_base / model_size / dataset / 'models' if model_size else result_base / dataset / 'models'
        if theta_base.exists():
            exp_dirs = sorted(theta_base.glob('exp_*'), key=lambda p: p.stat().st_mtime, reverse=True)
            if exp_dirs:
                # Output to result/{model_size}/{dataset}/{lang}/
                if model_size and model_size in str(result_dir):
                    output_dir = result_dir / dataset / language
                elif model_size:
                    output_dir = result_base / model_size / dataset / language
                else:
                    output_dir = result_base / dataset / language
            else:
                output_dir = theta_base / language
        else:
            if model_size and model_size in str(result_dir):
                output_dir = result_dir / dataset / language
            elif model_size:
                output_dir = result_base / model_size / dataset / language
            else:
                output_dir = result_dir / dataset / language
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running ETM Visualizations")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Language: {language}")
    print(f"DPI: {dpi}")
    print(f"{'='*60}\n")
    
    # Import visualization generator
    from visualization.visualization_generator import VisualizationGenerator
    
    # Create generator
    generator = VisualizationGenerator(
        theta=data['theta'],
        beta=data['beta'],
        vocab=data['vocab'],
        topic_words=data['topic_words'],
        topic_embeddings=data.get('topic_embeddings'),
        timestamps=data.get('timestamps'),
        bow_matrix=data.get('bow_matrix'),
        training_history=data.get('training_history'),
        metrics=data.get('metrics'),
        output_dir=str(output_dir),
        language=language,
        dpi=dpi
    )
    
    # Generate all visualizations
    generator.generate_all()
    
    # Also run topic visualizer for additional charts
    try:
        from visualization.topic_visualizer import TopicVisualizer
        
        print(f"\n[Additional Visualizations]")
        
        viz = TopicVisualizer(output_dir=str(output_dir / 'global'), dpi=dpi, language=language)
        
        # Topic word bars (split into individual per-topic charts)
        viz.visualize_topic_words(
            data['topic_words'],
            num_words=10
        )
        print(f"  ✓ Per-topic word distribution charts generated")
        
        # Topic similarity heatmap
        topic_sim_filename = '主题相似度.png' if language == 'zh' else 'topic_similarity.png'
        viz.visualize_topic_similarity(
            data['beta'],
            data['topic_words'],
            filename=topic_sim_filename
        )
        print(f"  ✓ {topic_sim_filename}")
        
        # Document-topic distribution
        doc_topic_filename = '文档主题分布_UMAP.png' if language == 'zh' else 'doc_topic_umap.png'
        viz.visualize_document_topics(
            data['theta'],
            method='umap',
            max_docs=5000,
            filename=doc_topic_filename
        )
        print(f"  ✓ {doc_topic_filename}")
        
        # Training history composite chart removed (single charts already generated by generator)

        # Metrics
        if data.get('metrics'):
            viz.visualize_metrics(
                data['metrics'],
                filename='metrics.png'
            )
            print(f"  ✓ metrics.png")
        
        # Word clouds (if wordcloud package available)
        try:
            viz.visualize_all_wordclouds(
                data['topic_words'],
                num_words=30,
                filename='topic_wordclouds.png'
            )
            print(f"  ✓ topic_wordclouds.png")
        except Exception as e:
            print(f"  ⚠ topic_wordclouds skipped: {e}")
        
        # pyLDAvis-style visualization (split into two separate charts)
        try:
            viz.visualize_intertopic_distance(
                data['theta'],
                data['beta']
            )
            print(f"  ✓ Intertopic Distance Map generated")
        except Exception as e:
            print(f"  ⚠ intertopic_distance skipped: {e}")
        
        try:
            viz.visualize_topic_word_frequency(
                data['beta'],
                data['topic_words'],
                selected_topic=0,
                n_words=30
            )
            print(f"  ✓ Top Salient Terms chart generated")
        except Exception as e:
            print(f"  ⚠ topic_word_frequency skipped: {e}")
        
        # pyLDAvis-style combined visualization
        try:
            if data.get('bow_matrix') is not None:
                pyldavis_filename = 'pyLDAvis风格图.png' if language == 'zh' else 'pyldavis_style.png'
                viz.visualize_pyldavis_style(
                    data['theta'],
                    data['beta'],
                    data['bow_matrix'],
                    data['vocab'],
                    filename=pyldavis_filename
                )
                print(f"  ✓ {pyldavis_filename}")
        except Exception as e:
            print(f"  ⚠ pyldavis_style skipped: {e}")
        
        # Also generate interactive HTML version if pyLDAvis is available
        try:
            from visualization.topic_visualizer import generate_pyldavis_visualization
            html_path = generate_pyldavis_visualization(
                theta=data['theta'],
                beta=data['beta'],
                bow_matrix=data.get('bow_matrix'),
                vocab=data['vocab'],
                output_path=str(output_dir / 'global' / 'pyldavis_interactive.html')
            )
            if html_path:
                print(f"  ✓ pyldavis_interactive.html")
        except Exception as e:
            print(f"  ⚠ pyldavis_interactive.html skipped: {e}")
        
    except Exception as e:
        print(f"  ⚠ Additional visualizations error: {e}")
    
    # Generate summary report
    generate_summary_report(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def generate_summary_report(data, output_dir):
    """Generate a summary report of the visualization."""
    output_dir = Path(output_dir)
    
    report = []
    report.append("# ETM Visualization Summary Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data summary
    report.append("## Data Summary")
    report.append(f"- Documents: {data['theta'].shape[0]:,}")
    report.append(f"- Topics: {data['theta'].shape[1]}")
    report.append(f"- Vocabulary size: {len(data['vocab']):,}")
    report.append(f"- Has timestamps: {'Yes' if data.get('timestamps') is not None else 'No'}")
    report.append(f"- Has training history: {'Yes' if data.get('training_history') is not None else 'No'}")
    report.append(f"- Has metrics: {'Yes' if data.get('metrics') is not None else 'No'}")
    report.append("")
    
    # Topic summary
    report.append("## Topic Summary")
    report.append("")
    for topic_id, words in data['topic_words']:
        top_words = [w[0] for w in words[:10]]
        strength = data['theta'][:, topic_id].mean()
        report.append(f"### Topic {topic_id + 1}")
        report.append(f"- **Strength**: {strength:.6f}")
        report.append(f"- **Top words**: {', '.join(top_words)}")
        report.append("")
    
    # Metrics summary
    if data.get('metrics'):
        report.append("## Evaluation Metrics")
        metrics = data['metrics']
        if 'topic_diversity_td' in metrics:
            report.append(f"- Topic Diversity (TD): {metrics['topic_diversity_td']:.4f}")
        if 'topic_diversity_irbo' in metrics:
            report.append(f"- Topic Diversity (iRBO): {metrics['topic_diversity_irbo']:.4f}")
        if 'topic_coherence_npmi_avg' in metrics:
            report.append(f"- Coherence (NPMI): {metrics['topic_coherence_npmi_avg']:.4f}")
        if 'topic_coherence_cv_avg' in metrics:
            report.append(f"- Coherence (C_V): {metrics['topic_coherence_cv_avg']:.4f}")
        if 'perplexity' in metrics and metrics['perplexity'] is not None:
            report.append(f"- Perplexity: {metrics['perplexity']:.2f}")
        report.append("")
    
    # Training summary
    if data.get('training_history'):
        history = data['training_history']
        report.append("## Training Summary")
        if 'epochs_trained' in history:
            report.append(f"- Epochs trained: {history['epochs_trained']}")
        if 'best_val_loss' in history:
            report.append(f"- Best validation loss: {history['best_val_loss']:.4f}")
        if 'test_loss' in history:
            report.append(f"- Test loss: {history['test_loss']:.4f}")
        report.append("")
    
    # Generated files
    report.append("## Generated Visualizations")
    report.append("")
    report.append("### Global Charts")
    global_dir = output_dir / 'global'
    if global_dir.exists():
        for f in sorted(global_dir.glob('*.png')):
            report.append(f"- `{f.name}`")
    report.append("")
    
    report.append("### Per-Topic Charts")
    topics_dir = output_dir / 'topics'
    if topics_dir.exists():
        topic_dirs = sorted(topics_dir.glob('topic_*'))
        if topic_dirs:
            report.append(f"- {len(topic_dirs)} topic directories with individual charts")
    report.append("")
    
    # Write report
    report_path = output_dir / 'README.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  ✓ README.md (summary report)")


def load_baseline_data(result_dir, dataset, model, num_topics=20):
    """
    Load baseline model data (LDA, ETM, CTM) for visualization.
    
    Args:
        result_dir: Result directory - can be:
            - New structure: full experiment path (e.g., .../models/lda/exp_xxx)
            - Old structure: base result directory (e.g., ./result/baseline)
        dataset: Dataset name (e.g., socialTwitter)
        model: Model name (lda, etm, ctm_zeroshot)
        num_topics: Number of topics
    
    Returns:
        dict with all visualization data
    """
    from scipy import sparse
    
    result_dir = Path(result_dir)
    
    # Check if result_dir is already the experiment directory (new structure)
    # New structure: result/{user}/{dataset}/{model}/{task_name}/
    data_exp_dir = None  # For loading vocab from data experiment
    
    # Helper function to find data_exp_dir from config
    def find_data_exp_dir(config_path, base_dir):
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            data_exp = config.get('data_exp')
            if data_exp:
                # Try multiple possible paths for data experiment
                possible_data_dirs = [
                    base_dir.parent.parent / 'data' / data_exp,  # .../models/../data/
                    base_dir.parent / 'data' / data_exp,  # .../data/
                    base_dir.parent.parent.parent / 'data' / data_exp,  # deeper nesting
                ]
                for d in possible_data_dirs:
                    if d.exists():
                        return d
        return None
    
    # New structure: result_dir is the task directory (e.g., .../ctm/bilingual_test/)
    # Model files are directly in result_dir or in model-specific subdirs (e.g., ctm_zeroshot/)
    if result_dir.name.startswith('exp_') or (result_dir / 'config.json').exists():
        # result_dir is the task/experiment directory
        model_dir = result_dir
        dataset_dir = result_dir
        data_exp_dir = find_data_exp_dir(result_dir / 'config.json', result_dir)
    elif (result_dir / model).exists():
        model_dir = result_dir / model
        dataset_dir = result_dir
        data_exp_dir = find_data_exp_dir(result_dir / 'config.json', result_dir)
    else:
        # Old structure: result_dir / dataset / model
        dataset_dir = result_dir / dataset
        model_dir = dataset_dir / model
    
    print(f"  Model dir: {model_dir}")
    print(f"  Data exp dir: {data_exp_dir}")
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"\n{'='*60}")
    print(f"Loading baseline data: {dataset} / {model}")
    print(f"{'='*60}")
    
    data = {}
    
    # Load theta from various possible subdirectories
    theta_path = None
    beta_path = None
    
    # Check paths in priority order
    possible_paths = [
        model_dir / 'model' / f'theta_k{num_topics}.npy',  # HDP/neural
        model_dir / f'theta_k{num_topics}.npy',  # direct
        model_dir / model / f'theta_k{num_topics}.npy',  # model subdir
        model_dir / model / 'model' / f'theta_k{num_topics}.npy',  # model/model subdir
    ]
    
    # CTM saves to ctm_zeroshot/ or ctm_combined/
    if model == 'ctm':
        possible_paths.insert(0, model_dir / 'ctm_zeroshot' / f'theta_k{num_topics}.npy')
        possible_paths.insert(1, model_dir / 'ctm_combined' / f'theta_k{num_topics}.npy')
    
    for p in possible_paths:
        if p.exists():
            theta_path = p
            beta_path = p.parent / f'beta_k{num_topics}.npy'
            break
    
    if theta_path and theta_path.exists():
        data['theta'] = np.load(theta_path)
        print(f"✓ Loaded theta: {data['theta'].shape}")
    else:
        raise FileNotFoundError(f"theta not found in any of: {[str(p) for p in possible_paths]}")
    
    if beta_path and beta_path.exists():
        data['beta'] = np.load(beta_path)
        print(f"✓ Loaded beta: {data['beta'].shape}")
    else:
        raise FileNotFoundError(f"beta not found: {beta_path}")
    
    # Load vocab - try multiple locations
    vocab_path = None
    # 1. Try data_exp_dir first (new structure)
    if data_exp_dir and (data_exp_dir / 'vocab.json').exists():
        vocab_path = data_exp_dir / 'vocab.json'
    # 2. Try model_dir / bow / vocab.json (old structure)
    elif (model_dir / 'bow' / 'vocab.json').exists():
        vocab_path = model_dir / 'bow' / 'vocab.json'
    # 3. Try dataset_dir / vocab.json
    elif (dataset_dir / 'vocab.json').exists():
        vocab_path = dataset_dir / 'vocab.json'
    
    if vocab_path and vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data['vocab'] = json.load(f)
        print(f"✓ Loaded vocab: {len(data['vocab'])} words")
    else:
        data['vocab'] = [f"word_{i}" for i in range(data['beta'].shape[1])]
        print(f"⚠ Generated placeholder vocab: {len(data['vocab'])} words")
    
    # Load topic_words from topicwords/ subdirectory
    topic_words_path = model_dir / 'topicwords' / f'topic_words_k{num_topics}.json'
    if not topic_words_path.exists():
        topic_words_path = model_dir / f'topic_words_k{num_topics}.json'
    if topic_words_path.exists():
        with open(topic_words_path, 'r', encoding='utf-8') as f:
            topic_words_raw = json.load(f)
        
        topic_words = []
        if isinstance(topic_words_raw, dict):
            # Format: {"topic_0": ["word1", ...], ...}
            sorted_items = sorted(topic_words_raw.items(), 
                                 key=lambda x: int(x[0].replace('topic_', '')) if 'topic_' in x[0] else int(x[0]))
            for key, words in sorted_items:
                topic_id = int(key.replace('topic_', '')) if 'topic_' in key else int(key)
                if isinstance(words, list) and len(words) > 0 and isinstance(words[0], str):
                    word_weights = []
                    for w in words:
                        idx = data['vocab'].index(w) if w in data['vocab'] else -1
                        # Check if idx is within beta bounds (for BERTopic which may have smaller beta)
                        if idx >= 0 and topic_id < data['beta'].shape[0] and idx < data['beta'].shape[1]:
                            weight = float(data['beta'][topic_id, idx])
                        else:
                            weight = 0.01
                        word_weights.append((w, weight))
                    topic_words.append((topic_id, word_weights))
                else:
                    topic_words.append((topic_id, []))
        else:
            # Generate from beta
            for i in range(data['beta'].shape[0]):
                top_indices = np.argsort(-data['beta'][i])[:20]
                words = [(data['vocab'][idx], float(data['beta'][i, idx])) for idx in top_indices]
                topic_words.append((i, words))
        
        data['topic_words'] = topic_words
        print(f"✓ Loaded topic_words: {len(data['topic_words'])} topics")
    else:
        # Generate from beta
        topic_words = []
        for i in range(data['beta'].shape[0]):
            top_indices = np.argsort(-data['beta'][i])[:20]
            words = [(data['vocab'][idx], float(data['beta'][i, idx])) for idx in top_indices]
            topic_words.append((i, words))
        data['topic_words'] = topic_words
        print(f"⚠ Generated topic_words from beta")
    
    # Load BOW matrix - try multiple locations
    bow_path = None
    # 1. Try data_exp_dir first (new structure)
    if data_exp_dir and (data_exp_dir / 'bow_matrix.npy').exists():
        bow_path = data_exp_dir / 'bow_matrix.npy'
    # 2. Try model_dir / bow / bow_matrix.npy (old structure)
    elif (model_dir / 'bow' / 'bow_matrix.npy').exists():
        bow_path = model_dir / 'bow' / 'bow_matrix.npy'
    # 3. Try dataset_dir / bow_matrix.npy
    elif (dataset_dir / 'bow_matrix.npy').exists():
        bow_path = dataset_dir / 'bow_matrix.npy'
    
    if bow_path and bow_path.exists():
        data['bow_matrix'] = np.load(bow_path)
        print(f"✓ Loaded bow_matrix: {data['bow_matrix'].shape}")
    
    # Load metrics - try multiple locations
    metrics_path = None
    # 1. Try result_dir (experiment directory) first
    if (result_dir / f'metrics_k{num_topics}.json').exists():
        metrics_path = result_dir / f'metrics_k{num_topics}.json'
    # 2. Try model_dir / evaluation / metrics_k{num_topics}.json
    elif (model_dir / 'evaluation' / f'metrics_k{num_topics}.json').exists():
        metrics_path = model_dir / 'evaluation' / f'metrics_k{num_topics}.json'
    # 3. Try model_dir / metrics_k{num_topics}.json
    elif (model_dir / f'metrics_k{num_topics}.json').exists():
        metrics_path = model_dir / f'metrics_k{num_topics}.json'
    
    if metrics_path and metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            data['metrics'] = json.load(f)
        print(f"✓ Loaded metrics from {metrics_path.name}")
    
    # Load timestamps for DTM (time_slices.json and time_indices.npy)
    data['topic_embeddings'] = None
    data['training_history'] = None
    data['timestamps'] = None
    
    # Try to load timestamp data (required for DTM)
    time_slices_path = dataset_dir / 'time_slices.json'
    time_indices_path = dataset_dir / 'time_indices.npy'
    
    if time_slices_path.exists() and time_indices_path.exists():
        with open(time_slices_path, 'r', encoding='utf-8') as f:
            time_slices_info = json.load(f)
        time_indices = np.load(time_indices_path)
        
        from datetime import datetime
        unique_times = time_slices_info.get('unique_times', [])
        
        timestamps = []
        for idx in time_indices:
            if idx < len(unique_times):
                year = unique_times[idx]
                timestamps.append(datetime(year, 1, 1))
            else:
                timestamps.append(datetime(2020, 1, 1))
        
        data['timestamps'] = np.array(timestamps)
        data['time_slices_info'] = time_slices_info
        print(f"✓ Loaded timestamps: {len(data['timestamps'])} dates ({len(unique_times)} unique years)")
    
    training_history_path = model_dir / f'training_history_k{num_topics}.json'
    if training_history_path.exists():
        with open(training_history_path, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"✓ Loaded training_history")

    # Load STM covariate data
    if model == 'stm':
        stm_model_dir = model_dir / 'stm' / 'model'
        if not stm_model_dir.exists():
            stm_model_dir = model_dir / 'model'

        covariate_info_path = stm_model_dir / f'covariate_info_k{num_topics}.json'
        if covariate_info_path.exists():
            with open(covariate_info_path, 'r', encoding='utf-8') as f:
                data['covariate_info'] = json.load(f)

        Gamma_path = stm_model_dir / f'Gamma_k{num_topics}.npy'
        if Gamma_path.exists():
            data['Gamma'] = np.load(Gamma_path)
            print(f"✓ Loaded Gamma: {data['Gamma'].shape}")

        cov_effects_path = stm_model_dir / f'covariate_effects_k{num_topics}.json'
        if cov_effects_path.exists():
            with open(cov_effects_path, 'r', encoding='utf-8') as f:
                data['covariate_effects'] = json.load(f)
            print(f"✓ Loaded covariate_effects")

        cov_saved_path = stm_model_dir / f'covariates_k{num_topics}.npy'
        if cov_saved_path.exists():
            data['covariates'] = np.load(cov_saved_path)
            print(f"✓ Loaded covariates (from model): {data['covariates'].shape}")

        if 'covariates' not in data and data_exp_dir is not None:
            ws_cov_path = data_exp_dir / 'covariates.npy'
            if ws_cov_path.exists():
                data['covariates'] = np.load(ws_cov_path)
                print(f"✓ Loaded covariates (from workspace): {data['covariates'].shape}")
            ws_names_path = data_exp_dir / 'covariate_names.json'
            if ws_names_path.exists():
                with open(ws_names_path, 'r', encoding='utf-8') as f:
                    data['covariate_names'] = json.load(f)
                print(f"✓ Loaded covariate_names: {data['covariate_names']}")

        if 'covariate_names' not in data and 'covariate_info' in data:
            data['covariate_names'] = data['covariate_info'].get('covariate_names', [])

    print(f"{'='*60}\n")
    return data


def _run_stm_specific_visualizations(data, output_dir, language='zh', dpi=300):
    """STM covariate visualizations: platform-topic association charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import warnings
    warnings.filterwarnings('ignore')

    output_dir = Path(output_dir)
    global_dir = output_dir / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)

    zh = (language == 'zh')
    theta = data.get('theta')
    covariates = data.get('covariates')
    covariate_names = data.get('covariate_names', [])
    topic_words = data.get('topic_words', [])
    K = theta.shape[1] if theta is not None else 0

    if theta is None or covariates is None or covariates.shape[0] != theta.shape[0]:
        print("  ⚠ STM covariate viz skipped: theta/covariates unavailable or shape mismatch")
        return

    # Recover original platform labels
    platform_labels = {}
    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        config_path = output_dir.parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            dataset = cfg.get('dataset', '')
            candidate = Path(__file__).parent.parent.parent / 'data' / dataset / f'{dataset}_cleaned.csv'
            if candidate.exists():
                df_raw = pd.read_csv(candidate)
                cov_col = covariate_names[0] if covariate_names else None
                if cov_col and cov_col in df_raw.columns:
                    le = LabelEncoder()
                    le.fit(df_raw[cov_col].fillna('unknown').astype(str))
                    for i, label in enumerate(le.classes_):
                        platform_labels[i] = label
    except Exception:
        pass

    unique_vals = sorted(set(covariates[:, 0].astype(int).tolist()))
    cov_labels = [platform_labels.get(v, f'cat_{v}') for v in unique_vals]

    def topic_label(tid):
        if tid < len(topic_words):
            words = topic_words[tid][1]
            if words:
                return f"T{tid+1}:{words[0][0]}"
        return f"Topic {tid+1}"

    topic_labels = [topic_label(i) for i in range(K)]

    mean_theta = np.zeros((len(unique_vals), K))
    for i, val in enumerate(unique_vals):
        mask = covariates[:, 0].astype(int) == val
        if mask.sum() > 0:
            mean_theta[i] = theta[mask].mean(axis=0)

    # Chart 1: 协变量-主题关联热力图
    try:
        fig, ax = plt.subplots(figsize=(max(10, K * 0.9), max(4, len(unique_vals) * 0.8)))
        im = ax.imshow(mean_theta, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label='平均主题占比' if zh else 'Mean Topic Proportion')
        ax.set_xticks(range(K)); ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(unique_vals))); ax.set_yticklabels(cov_labels, fontsize=9)
        ax.set_title('协变量-主题关联热力图' if zh else 'Covariate-Topic Heatmap', fontsize=12, fontweight='bold')
        for i in range(len(unique_vals)):
            for j in range(K):
                ax.text(j, i, f'{mean_theta[i,j]:.3f}', ha='center', va='center', fontsize=6,
                        color='white' if mean_theta[i,j] > 0.15 else 'black')
        plt.tight_layout()
        plt.savefig(global_dir / ('协变量主题关联热力图.png' if zh else 'covariate_topic_heatmap.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 协变量主题关联热力图.png")
    except Exception as e:
        print(f"  ⚠ heatmap: {e}")

    # Chart 2: 各平台主题分布堆积图
    try:
        colors = cm.tab10(np.linspace(0, 1, K))
        fig, ax = plt.subplots(figsize=(max(8, len(unique_vals) * 1.5), 6))
        bottom = np.zeros(len(unique_vals))
        for k in range(K):
            ax.bar(cov_labels, mean_theta[:, k], bottom=bottom, color=colors[k], label=topic_labels[k], alpha=0.85)
            bottom += mean_theta[:, k]
        ax.set_title('各平台主题分布对比（堆积）' if zh else 'Topic Distribution by Platform', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), fontsize=7)
        plt.xticks(rotation=30, ha='right', fontsize=9); plt.tight_layout()
        plt.savefig(global_dir / ('各平台主题分布堆积图.png' if zh else 'platform_topic_stacked.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 各平台主题分布堆积图.png")
    except Exception as e:
        print(f"  ⚠ stacked bar: {e}")

    # Chart 3: 平台主题偏好偏差图
    try:
        deviation = mean_theta - theta.mean(axis=0)[np.newaxis, :]
        vmax = np.abs(deviation).max()
        fig, ax = plt.subplots(figsize=(max(10, K * 0.9), max(4, len(unique_vals) * 0.8)))
        im = ax.imshow(deviation, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='偏差（相对全局均值）' if zh else 'Deviation from Global Mean')
        ax.set_xticks(range(K)); ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(unique_vals))); ax.set_yticklabels(cov_labels, fontsize=9)
        ax.set_title('平台主题偏好偏差图（红=高于均值，蓝=低于均值）' if zh else 'Platform Topic Deviation', fontsize=11, fontweight='bold')
        for i in range(len(unique_vals)):
            for j in range(K):
                ax.text(j, i, f'{deviation[i,j]:+.3f}', ha='center', va='center', fontsize=6, color='black')
        plt.tight_layout()
        plt.savefig(global_dir / ('平台主题偏好偏差图.png' if zh else 'platform_topic_deviation.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 平台主题偏好偏差图.png")
    except Exception as e:
        print(f"  ⚠ deviation: {e}")

    # Chart 4: 各平台主导主题 Top-5
    try:
        colors_top = cm.tab10(np.linspace(0, 1, K))
        fig, axes = plt.subplots(1, len(unique_vals), figsize=(len(unique_vals) * 3, 5), sharey=False)
        if len(unique_vals) == 1: axes = [axes]
        for ax, i in zip(axes, range(len(unique_vals))):
            top5 = np.argsort(-mean_theta[i])[:5]
            ax.barh(range(5), mean_theta[i, top5][::-1], color=[colors_top[j] for j in top5[::-1]], alpha=0.85)
            ax.set_yticks(range(5)); ax.set_yticklabels([topic_labels[j] for j in top5[::-1]], fontsize=8)
            ax.set_title(cov_labels[i], fontsize=9, fontweight='bold')
        fig.suptitle('各平台 Top-5 主导主题' if zh else 'Top-5 Topics per Platform', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(global_dir / ('各平台主导主题.png' if zh else 'platform_dominant_topics.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 各平台主导主题.png")
    except Exception as e:
        print(f"  ⚠ dominant topics: {e}")

    # Chart 5: Gamma 系数图（若已保存）
    Gamma = data.get('Gamma')
    if Gamma is not None:
        try:
            n_cov = Gamma.shape[0] - 1
            fig, axes = plt.subplots(1, max(1, n_cov), figsize=(max(8, K * 0.7) * n_cov, 5))
            if n_cov == 1: axes = [axes]
            for c in range(n_cov):
                ax = axes[c]
                coefs = Gamma[c + 1, :]
                cname = covariate_names[c] if c < len(covariate_names) else f'cov_{c}'
                ax.bar(range(K - 1), coefs, color=['#d62728' if v > 0 else '#1f77b4' for v in coefs], alpha=0.8)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.set_xticks(range(K - 1)); ax.set_xticklabels([f'T{i+1}' for i in range(K - 1)], rotation=45, fontsize=8)
                ax.set_title(f'协变量效应：{cname}' if zh else f'Covariate Effect: {cname}', fontsize=10, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(global_dir / ('STM协变量Gamma系数图.png' if zh else 'stm_gamma_coefficients.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
            plt.close(); print(f"  ✓ STM协变量Gamma系数图.png")
        except Exception as e:
            print(f"  ⚠ gamma: {e}")

    # Chart 6: ANOVA F 检验显著性
    try:
        from scipy import stats
        f_stats, p_vals = [], []
        for k in range(K):
            groups = [theta[covariates[:, 0].astype(int) == v, k] for v in unique_vals if (covariates[:, 0].astype(int) == v).sum() >= 2]
            if len(groups) >= 2:
                f, p = stats.f_oneway(*groups)
                f_stats.append(f); p_vals.append(p)
            else:
                f_stats.append(0); p_vals.append(1.0)
        f_stats, p_vals = np.array(f_stats), np.array(p_vals)
        fig, ax = plt.subplots(figsize=(max(8, K * 0.8), 5))
        ax.bar(range(K), f_stats, color=['#d62728' if p < 0.05 else '#aec7e8' for p in p_vals], alpha=0.85, edgecolor='white')
        ax.set_xticks(range(K)); ax.set_xticklabels(topic_labels, rotation=45, ha='right', fontsize=8)
        ax.set_title('平台对各主题的协变量效应（ANOVA F检验，红=p<0.05）' if zh else 'ANOVA F-Statistic: Platform Effect on Topics', fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for i, (f, p) in enumerate(zip(f_stats, p_vals)):
            ax.text(i, f + f_stats.max() * 0.02, f'p={p:.3f}', ha='center', va='bottom', fontsize=6)
        plt.tight_layout()
        plt.savefig(global_dir / ('协变量效应显著性检验.png' if zh else 'covariate_effect_anova.png'), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 协变量效应显著性检验.png")
    except Exception as e:
        print(f"  ⚠ anova: {e}")


def _run_dtm_specific_visualizations(data, output_dir, language='en', dpi=300):
    """
    Run DTM-specific visualizations using visualization_generator2.
    
    DTM-specific visualizations:
    - Topic evolution sankey diagram (topic_sankey.png)
    - Topic strength temporal changes (topic_similarity_evolution.png)
    - All topics strength table (all_topics_strength_table.png)
    - High-frequency word evolution (vocab_evolution.png)
    - Topic independence visualization (topic_independence.png)
    - Global word cloud (wordcloud_global.png)
    - Topic proportion pie chart (topic_proportion.png)
    """
    from pathlib import Path
    import numpy as np
    
    output_dir = Path(output_dir)
    global_dir = output_dir / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load DTM-specific data (beta_over_time, topic_evolution)
    dtm_dir = output_dir.parent
    
    # Load beta_over_time from model/ subdirectory
    beta_over_time = None
    beta_over_time_file = dtm_dir / 'model' / f"beta_over_time_k{data['theta'].shape[1]}.npy"
    if not beta_over_time_file.exists():
        beta_over_time_file = dtm_dir / f"beta_over_time_k{data['theta'].shape[1]}.npy"
    if beta_over_time_file.exists():
        beta_over_time = np.load(beta_over_time_file)
        print(f"  Loaded beta_over_time: {beta_over_time.shape}")
    
    # Load topic_evolution from topicwords/ subdirectory
    topic_evolution = None
    topic_evolution_file = dtm_dir / 'topicwords' / f"topic_evolution_k{data['theta'].shape[1]}.json"
    if not topic_evolution_file.exists():
        topic_evolution_file = dtm_dir / f"topic_evolution_k{data['theta'].shape[1]}.json"
    if topic_evolution_file.exists():
        import json
        with open(topic_evolution_file, 'r', encoding='utf-8') as f:
            topic_evolution = json.load(f)
        print(f"  Loaded topic_evolution: {len(topic_evolution)} topics")
    
    try:
        from visualization.visualization_generator2 import VisualizationGenerator as VG2
        
        gen2 = VG2(
            theta=data['theta'],
            beta=data['beta'],
            vocab=data['vocab'],
            topic_words=data['topic_words'],
            topic_embeddings=data.get('topic_embeddings'),
            timestamps=data.get('timestamps'),
            bow_matrix=data.get('bow_matrix'),
            training_history=data.get('training_history'),
            metrics=data.get('metrics'),
            output_dir=str(output_dir),
            language=language,
            dpi=dpi
        )
        
        try:
            gen2.generate_pyldavis()
        except Exception as e:
            print(f"  ⚠ topic_independence.png skipped: {e}")
        
        try:
            gen2.generate_global_wordcloud()
        except Exception as e:
            print(f"  ⚠ wordcloud_global.png skipped: {e}")
        
        try:
            gen2.generate_topic_proportion_pie()
        except Exception as e:
            print(f"  ⚠ topic_proportion.png skipped: {e}")
        
        if data.get('timestamps') is not None:
            try:
                gen2.generate_sankey_diagram()
            except Exception as e:
                print(f"  ⚠ topic_sankey.png skipped: {e}")
            
            try:
                gen2.generate_topic_similarity_evolution()
            except Exception as e:
                print(f"  ⚠ topic_similarity_evolution.png skipped: {e}")
            
            try:
                gen2.generate_all_topics_strength_table()
            except Exception as e:
                print(f"  ⚠ all_topics_strength_table.png skipped: {e}")
        
        if data.get('training_history') is not None:
            try:
                gen2.generate_training_convergence()
            except Exception as e:
                print(f"  ⚠ training_convergence skipped: {e}")
        
    except ImportError as e:
        print(f"  ⚠ visualization_generator2 not available: {e}")
    except Exception as e:
        print(f"  ⚠ DTM visualizations error: {e}")
    
    if topic_evolution is not None:
        try:
            _generate_topic_word_evolution(topic_evolution, global_dir, language, dpi)
        except Exception as e:
            print(f"  ⚠ topic_word_evolution.png skipped: {e}")


def _generate_topic_word_evolution(topic_evolution, output_dir, language='en', dpi=300):
    """Generate DTM topic word evolution visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    output_dir = Path(output_dir)
    n_topics = len(topic_evolution)
    
    n_show = min(6, n_topics)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (topic_id, time_data) in enumerate(list(topic_evolution.items())[:n_show]):
        ax = axes[idx]
        
        times = sorted(time_data.keys())
        
        if len(times) < 2:
            ax.text(0.5, 0.5, f'Topic {topic_id}\n(insufficient data)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
        
        all_words = set()
        for t in times:
            words = time_data[t][:5]  # top 5 words
            for w, _ in words:
                all_words.add(w)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(all_words)))
        
        for i, word in enumerate(list(all_words)[:5]):
            weights = []
            for t in times:
                weight = 0
                for w, wt in time_data[t]:
                    if w == word:
                        weight = wt
                        break
                weights.append(weight)
            
            ax.plot(range(len(times)), weights, 'o-', color=colors[i], 
                   label=word, linewidth=2, markersize=4)
        
        ax.set_title(f'Topic {topic_id}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Weight', fontsize=9)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    
    title = 'DTM Topic Word Evolution'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_word_evolution.png', dpi=dpi, 
               bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ topic_word_evolution.png")


def run_baseline_visualization(
    result_dir,
    dataset,
    model,
    num_topics=20,
    output_dir=None,
    language='both',  # Changed default to 'both' for bilingual output
    dpi=300
):
    """
    Run visualizations for baseline models (LDA, ETM, CTM, DTM).
    
    Args:
        result_dir: Base result directory for baseline models
        dataset: Dataset name
        model: Model name (lda, etm, ctm_zeroshot, dtm)
        num_topics: Number of topics
        output_dir: Output directory (default: result_dir/dataset/model/visualization/)
        language: Language for labels ('en', 'zh', or 'both' for bilingual)
        dpi: DPI for saved figures
    
    Returns:
        Path to output directory
    """
    # Load data
    data = load_baseline_data(result_dir, dataset, model, num_topics)
    
    # Determine output directory (no language suffix in folder name anymore)
    if output_dir is None:
        result_path = Path(result_dir)
        viz_folder = 'visualization'
        
        # Check if result_dir is already an experiment directory (new structure)
        if result_path.name.startswith('exp_') or (result_path / model).exists():
            # New structure: output to result_dir/visualization/
            output_dir = result_path / viz_folder
        else:
            # Old structure: result_dir/dataset/model/visualization
            output_dir = result_path / dataset / model / viz_folder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which languages to generate
    # Map 'cn' to 'zh' for internal processing (visualization uses 'zh' internally)
    lang_map = {'cn': 'zh', 'en': 'en', 'zh': 'zh'}
    if language == 'both':
        languages = ['en', 'zh']
    else:
        languages = [lang_map.get(language, language)]
    
    print(f"\n{'='*60}")
    print(f"Running Visualizations for {model.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Output: {output_dir}")
    print(f"Languages: {', '.join(languages)}")
    print(f"{'='*60}\n")
    
    # Use VisualizationGenerator
    from visualization.visualization_generator import VisualizationGenerator
    
    # Generate visualizations for each language
    for lang in languages:
        print(f"\n{'='*60}")
        print(f"Generating visualizations ({lang.upper()})")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        generator = VisualizationGenerator(
            theta=data['theta'],
            beta=data['beta'],
            vocab=data['vocab'],
            topic_words=data['topic_words'],
            topic_embeddings=data.get('topic_embeddings'),
            timestamps=data.get('timestamps'),
            bow_matrix=data.get('bow_matrix'),
            training_history=data.get('training_history'),
            metrics=data.get('metrics'),
            output_dir=str(output_dir),
            language=lang,
            dpi=dpi
        )
        
        generator.generate_all()
        
        # Additional visualizations using TopicVisualizer
        try:
            from visualization.topic_visualizer import TopicVisualizer
            
            print(f"\n[Additional Visualizations ({lang.upper()})]")
            # Use global directory directly under output_dir (new structure)
            global_dir = output_dir / 'global'
            global_dir.mkdir(parents=True, exist_ok=True)
            viz = TopicVisualizer(output_dir=str(global_dir), dpi=dpi, language=lang)
            
            # Filename mapping based on language
            if lang == 'zh':
                filenames = {
                    'topic_words_bars': '主题词条形图.png',
                    'topic_similarity': '主题相似度图.png',
                    'doc_topic_umap': '文档主题UMAP图.png',
                    'metrics': '评估指标图.png',
                    'topic_wordclouds': '主题词云图.png',
                    'pyldavis_intertopic': '主题间距离图.png',
                    'pyldavis_interactive': '交互式主题可视化.html',
                }
            else:
                filenames = {
                    'topic_words_bars': 'topic_words_bars.png',
                    'topic_similarity': 'topic_similarity.png',
                    'doc_topic_umap': 'doc_topic_umap.png',
                    'metrics': 'metrics.png',
                    'topic_wordclouds': 'topic_wordclouds.png',
                    'pyldavis_intertopic': 'pyldavis_intertopic.png',
                    'pyldavis_interactive': 'pyldavis_interactive.html',
                }
            
            viz.visualize_topic_words(data['topic_words'], num_words=10)
            print(f"  ✓ Per-topic word distribution charts generated")
            
            viz.visualize_topic_similarity(data['beta'], data['topic_words'], filename=filenames['topic_similarity'])
            print(f"  ✓ {filenames['topic_similarity']}")
            
            viz.visualize_document_topics(data['theta'], method='umap', max_docs=5000, filename=filenames['doc_topic_umap'])
            print(f"  ✓ {filenames['doc_topic_umap']}")
            
            
            try:
                viz.visualize_all_wordclouds(data['topic_words'], num_words=30, filename=filenames['topic_wordclouds'])
                print(f"  ✓ {filenames['topic_wordclouds']}")
            except Exception as e:
                print(f"  ⚠ {filenames['topic_wordclouds']} skipped: {e}")
            
            try:
                viz.visualize_intertopic_distance(data['theta'], data['beta'])
                viz.visualize_topic_word_frequency(data['beta'], data['topic_words'], selected_topic=0, n_words=30)
                print(f"  ✓ pyldavis split charts generated")
            except Exception as e:
                print(f"  ⚠ pyldavis charts skipped: {e}")
            
            try:
                from visualization.topic_visualizer import generate_pyldavis_visualization
                html_path = generate_pyldavis_visualization(
                    theta=data['theta'], beta=data['beta'], bow_matrix=data.get('bow_matrix'),
                    vocab=data['vocab'], output_path=str(global_dir / filenames['pyldavis_interactive'])
                )
                if html_path:
                    print(f"  ✓ {filenames['pyldavis_interactive']}")
            except Exception as e:
                print(f"  ⚠ pyldavis_interactive.html skipped: {e}")
                
        except Exception as e:
            print(f"  ⚠ Additional visualizations error: {e}")
        
        if model == 'dtm':
            try:
                print(f"\n[DTM-Specific Visualizations ({lang.upper()})]")
                _run_dtm_specific_visualizations(data, output_dir, lang, dpi)
            except Exception as e:
                print(f"  ⚠ DTM-specific visualizations error: {e}")

        if model == 'stm':
            try:
                print(f"\n[STM Covariate Visualizations ({lang.upper()})]")
                _run_stm_specific_visualizations(data, output_dir, lang, dpi)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  ⚠ STM covariate visualizations error: {e}")
    
    generate_summary_report(data, output_dir)
    
    print(f"\n{'='*60}")
    print(f"✓ Visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def run_all_baseline_visualizations(
    result_dir=None,
    datasets=None,
    models=None,
    num_topics=20,
    language='en',
    dpi=300
):
    """
    Run visualizations for all baseline models.
    
    Args:
        result_dir: Base result directory
        datasets: List of datasets (default: all)
        models: List of models (default: all)
        num_topics: Number of topics
        language: Language for labels
        dpi: DPI for figures
    """
    if datasets is None:
        datasets = ['socialTwitter', 'hatespeech', 'mental_health', 'FCPB', 'germanCoal']
    if models is None:
        models = ['lda', 'etm', 'ctm_zeroshot']
    
    print("="*70)
    print("Running Visualizations for All Baseline Models")
    print("="*70)
    
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for model in models:
            print(f"\n>>> {dataset} / {model}")
            try:
                run_baseline_visualization(result_dir, dataset, model, num_topics, language=language, dpi=dpi)
                results[dataset][model] = 'SUCCESS'
            except Exception as e:
                print(f"  [ERROR] {e}")
                results[dataset][model] = f'FAILED: {e}'
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for dataset, models_result in results.items():
        print(f"\n{dataset}:")
        for model, status in models_result.items():
            print(f"  {model}: {status}")


def main():
    parser = argparse.ArgumentParser(
        description='ETM Unified Visualization Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # THETA model visualization
    python run_visualization.py --result_dir ./result --dataset socialTwitter --mode zero_shot
    
    # Baseline model visualization
    python run_visualization.py --baseline --result_dir ./result/baseline --dataset FCPB --model lda
    
    # All baseline models
    python run_visualization.py --baseline --all
        """
    )
    
    # Baseline mode arguments
    parser.add_argument('--baseline', action='store_true',
                        help='Run visualization for baseline models (LDA, ETM, CTM)')
    parser.add_argument('--all', action='store_true',
                        help='Run for all datasets and models (baseline mode only)')
    parser.add_argument('--model', type=str, default=None,
                        choices=['lda', 'hdp', 'stm', 'btm', 'etm', 'ctm', 'ctm_zeroshot', 'dtm', 'nvdm', 'gsm', 'prodlda', 'bertopic'],
                        help='Model name (baseline mode only)')
    parser.add_argument('--num_topics', type=int, default=20,
                        help='Number of topics (baseline mode only)')
    
    # Common arguments
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Base result directory')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['zero_shot', 'supervised', 'unsupervised'],
                        help='Training mode (THETA mode only)')
    parser.add_argument('--model_size', type=str, default=None,
                        help='Model size subdirectory (e.g., 0.6B)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto)')
    parser.add_argument('--language', type=str, default='en',
                        choices=['en', 'zh'],
                        help='Language for labels')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for saved figures')
    
    args = parser.parse_args()
    
    if args.baseline:
        # Baseline model visualization
        if args.all:
            run_all_baseline_visualizations(
                result_dir=args.result_dir or os.path.join(os.environ.get('RESULT_DIR', 'result'), 'baseline'),
                num_topics=args.num_topics,
                language=args.language,
                dpi=args.dpi
            )
        elif args.dataset and args.model:
            run_baseline_visualization(
                result_dir=args.result_dir or os.path.join(os.environ.get('RESULT_DIR', 'result'), 'baseline'),
                dataset=args.dataset,
                model=args.model,
                num_topics=args.num_topics,
                output_dir=args.output_dir,
                language=args.language,
                dpi=args.dpi
            )
        else:
            parser.error("Baseline mode requires --all or both --dataset and --model")
    else:
        # THETA model visualization
        if not args.result_dir or not args.dataset or not args.mode:
            parser.error("THETA mode requires --result_dir, --dataset, and --mode")
        run_all_visualizations(
            result_dir=args.result_dir,
            dataset=args.dataset,
            mode=args.mode,
            model_size=args.model_size,
            output_dir=args.output_dir,
            language=args.language,
            dpi=args.dpi
        )


if __name__ == '__main__':
    main()
