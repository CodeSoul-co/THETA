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
from visualization.publication import COLORS, save_figure, setup_style, validate_export, export_manifest


def find_latest_file(directory, pattern, fixed_name=None):
    """
    Find the latest file matching pattern in directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern (e.g., "theta_*.npy")
        fixed_name: Optional fixed filename to try first (e.g., "theta.npy")
    
    Returns:
        Path to the file, or None if not found
    """
    from glob import glob
    directory = Path(directory)
    
    # Priority 1: Try fixed filename first (new format without timestamp)
    if fixed_name:
        fixed_path = directory / fixed_name
        if fixed_path.exists():
            return str(fixed_path)
    
    # Priority 2: Try glob pattern (legacy format with timestamp)
    files = glob(str(directory / pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def resolve_theta_data_dir(exp_dir, mode):
    """Resolve the exact preparation experiment recorded by the existing training config."""
    exp_dir = Path(exp_dir)
    for file in [exp_dir / f'config_{mode}.json', exp_dir / 'config.json']:
        if file.is_file():
            config = json.loads(file.read_text(encoding='utf-8'))
            name = config.get('data_exp')
            if name:
                if Path(name).name != name or name in {'.', '..'}:
                    raise ValueError('Invalid preparation experiment identifier')
                return exp_dir.parent / name / 'data'
    return exp_dir / 'data'


def load_visualization_data(
    result_dir, 
    dataset, 
    mode, 
    model_size=None, 
    model_exp=None,
    model_type='theta',
    num_topics=None
):
    """
    Load all data needed for visualization from result directory.
    
    Directory structures:
    - THETA:    result/{dataset}/{model_size}/theta/exp_{timestamp}/theta/
    - Baseline: result/baseline/{dataset}/vocab_{size}/{model_name}/
    
    Args:
        result_dir: Base result directory (e.g., ./result)
        dataset: Dataset name (e.g., socialTwitter)
        mode: Training mode (zero_shot, supervised, unsupervised)
        model_size: Model size for THETA (e.g., 0.6B)
        model_exp: Specific experiment ID (e.g., exp_20260326_141527)
        model_type: 'theta' | 'lda' | 'etm' | 'ctm_combined' | 'ctm_zeroshot'
        num_topics: Number of topics (required for baseline models)
    
    Returns:
        dict with all visualization data
    """
    from scipy import sparse
    
    result_dir = Path(result_dir)
    is_theta = model_type == 'theta'
    
    # Auto-discover directories based on model type
    model_dir = None
    topic_words_dir = None
    bow_dir = None
    evaluation_dir = None
    exp_dir = None
    
    # ==========================================================================
    # Directory Discovery - Strict path alignment with main.py / baseline_trainer.py
    # ==========================================================================
    
    if is_theta:
        # THETA structure:
        # result/{dataset}/{model_size}/theta/exp_{timestamp}/theta/{mode}/
        # Matches main.py config.model_dir
        if model_size:
            theta_base = result_dir / dataset / model_size / 'theta'
        else:
            # Fallback: try to find model_size from directory structure
            theta_base = None
            for ms in ['0.6B', '1.5B', '3B', '7B', '14B', '32B', '72B']:
                candidate = result_dir / dataset / ms / 'theta'
                if candidate.exists():
                    theta_base = candidate
                    model_size = ms
                    break
            if theta_base is None:
                theta_base = result_dir / dataset / 'theta'
        
        # Find experiment directory
        if model_exp:
            exp_dir = theta_base / model_exp
        else:
            # Find latest exp_* directory
            if theta_base.exists():
                exp_dirs = sorted(theta_base.glob('exp_*'), key=os.path.getmtime, reverse=True)
                if exp_dirs:
                    exp_dir = exp_dirs[0]
        
        if exp_dir and exp_dir.exists():
            mode_dir = exp_dir / 'theta' / mode
            # Current runs separate artifacts by mode. Keep the parent-level
            # layout readable for already-produced archives and integrations
            # that implement the older fixed-file contract.
            model_dir = mode_dir if mode_dir.exists() else exp_dir / 'theta'
            bow_dir = resolve_theta_data_dir(exp_dir, mode) / 'bow'  # result/{dataset}/{model_size}/theta/exp_*/data/bow/
            evaluation_dir = exp_dir
            topic_words_dir = model_dir  # topic_words.json is in model_dir
    
    else:
        # Baseline structure: result/baseline/{dataset}/vocab_{size}/{model_name}/
        # Matches baseline_trainer.py self.output_dir
        baseline_base = result_dir / 'baseline' / dataset
        
        # Find vocab_* directory (latest or specified)
        vocab_dirs = sorted(baseline_base.glob('vocab_*'), key=os.path.getmtime, reverse=True)
        if vocab_dirs:
            vocab_dir = vocab_dirs[0]
            model_dir = vocab_dir / model_type  # e.g., vocab_5000/lda/
            bow_dir = vocab_dir  # BOW is directly in vocab_* dir
            topic_words_dir = model_dir
    
    # ==========================================================================
    # Fallback for legacy structures
    # ==========================================================================
    
    if model_dir is None or not model_dir.exists():
        # Try legacy THETA paths
        legacy_paths = [
            result_dir / dataset / mode / 'model',
            result_dir / dataset / mode / 'theta',
            result_dir / model_size / dataset / 'theta' if model_size else None,
        ]
        for p in legacy_paths:
            if p and p.exists():
                model_dir = p
                bow_dir = p.parent / 'bow'
                evaluation_dir = p.parent / 'evaluation'
                topic_words_dir = p.parent / 'topic_words'
                break
    
    if model_dir is None or not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find model directory.\n"
            f"  Model type: {model_type}\n"
            f"  Dataset: {dataset}\n"
            f"  Model size: {model_size}\n"
            f"  Searched in: {result_dir}"
        )
    
    # ==========================================================================
    # Generate file names based on model type
    # ==========================================================================
    
    if is_theta:
        # THETA uses fixed filenames (no K suffix)
        theta_fixed = "theta.npy"
        beta_fixed = "beta.npy"
        topic_emb_fixed = "topic_embeddings.npy"
        topic_words_fixed = "topic_words.json"
        history_fixed = "training_history.json"
        theta_pattern = "theta_*.npy"
        beta_pattern = "beta_*.npy"
        topic_emb_pattern = "topic_embeddings_*.npy"
        topic_words_pattern = "topic_words_*.json"
        history_pattern = "training_history_*.json"
    else:
        # Baseline uses K-suffixed filenames
        if num_topics is None:
            # Auto-detect num_topics from existing files
            theta_files = list(model_dir.glob('theta_k*.npy'))
            if theta_files:
                # Extract K from filename like theta_k20.npy
                import re
                match = re.search(r'theta_k(\d+)\.npy', theta_files[0].name)
                if match:
                    num_topics = int(match.group(1))
        
        if num_topics:
            theta_fixed = f"theta_k{num_topics}.npy"
            beta_fixed = f"beta_k{num_topics}.npy"
            topic_words_fixed = f"topic_words_k{num_topics}.json"
            history_fixed = f"training_history_k{num_topics}.json"
        else:
            theta_fixed = None
            beta_fixed = None
            topic_words_fixed = None
            history_fixed = None
        
        topic_emb_fixed = None  # Baseline doesn't have topic embeddings
        theta_pattern = "theta_k*.npy"
        beta_pattern = "beta_k*.npy"
        topic_emb_pattern = None
        topic_words_pattern = "topic_words_k*.json"
        history_pattern = "training_history_k*.json"
    
    print(f"\n{'='*60}")
    print(f"Loading visualization data")
    print(f"{'='*60}")
    print(f"Model type: {model_type}")
    print(f"Model directory: {model_dir}")
    if topic_words_dir:
        print(f"Topic words directory: {topic_words_dir}")
    if bow_dir:
        print(f"BOW directory: {bow_dir}")
    if evaluation_dir:
        print(f"Evaluation directory: {evaluation_dir}")
    
    data = {}
    data['model_type'] = model_type
    data['is_theta'] = is_theta
    
    # Load theta (document-topic distribution)
    # Use dynamically generated filenames based on model type
    theta_file = find_latest_file(model_dir, theta_pattern, fixed_name=theta_fixed)
    if theta_file:
        data['theta'] = np.load(theta_file)
        print(f"✓ Loaded theta: {data['theta'].shape} from {Path(theta_file).name}")
    else:
        raise FileNotFoundError(f"theta not found in {model_dir}, expected: {theta_fixed or theta_pattern}")
    
    # Load beta (topic-word distribution)
    beta_file = find_latest_file(model_dir, beta_pattern, fixed_name=beta_fixed)
    if beta_file:
        data['beta'] = np.load(beta_file)
        print(f"✓ Loaded beta: {data['beta'].shape} from {Path(beta_file).name}")
    else:
        raise FileNotFoundError(f"beta not found in {model_dir}, expected: {beta_fixed or beta_pattern}")
    
    # Load topic embeddings (THETA only)
    if topic_emb_pattern:
        emb_file = find_latest_file(model_dir, topic_emb_pattern, fixed_name=topic_emb_fixed)
        if emb_file:
            data['topic_embeddings'] = np.load(emb_file)
            print(f"✓ Loaded topic_embeddings: {data['topic_embeddings'].shape}")
    
    # Load topic words - use dynamically generated filenames
    words_file = find_latest_file(topic_words_dir, topic_words_pattern, fixed_name=topic_words_fixed)
    if not words_file:
        words_file = find_latest_file(model_dir, topic_words_pattern, fixed_name=topic_words_fixed)
    
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
    
    # Load training history - use dynamically generated filenames
    history_file = find_latest_file(model_dir, history_pattern, fixed_name=history_fixed)
    if history_file:
        with open(history_file, 'r', encoding='utf-8') as f:
            data['training_history'] = json.load(f)
        print(f"✓ Loaded training_history: {len(data['training_history'].get('train_loss', []))} epochs")
    
    # Load evaluation metrics (THETA only has metrics.json in exp_dir)
    if is_theta and evaluation_dir:
        metrics_file = find_latest_file(evaluation_dir, "metrics_*.json", fixed_name="metrics.json")
        if metrics_file:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                data['metrics'] = json.load(f)
            print(f"✓ Loaded metrics")
    elif not is_theta:
        # Baseline has info_k{K}.json instead of metrics.json
        info_fixed = f"info_k{num_topics}.json" if num_topics else None
        info_file = find_latest_file(model_dir, "info_k*.json", fixed_name=info_fixed)
        if info_file:
            with open(info_file, 'r', encoding='utf-8') as f:
                data['metrics'] = json.load(f)
            print(f"✓ Loaded model info as metrics")
    
    # Use only persisted vocabulary; fabricated word_N labels cannot support interpretation.
    vocab_paths = [model_dir / 'vocab.json'] + ([bow_dir / 'vocab.txt', bow_dir / 'vocab.json'] if bow_dir else [])
    vocab_path = next((file for file in vocab_paths if file.is_file()), None)
    if vocab_path is None: raise ValueError('Training vocabulary is unavailable')
    data['vocab'] = json.loads(vocab_path.read_text(encoding='utf-8')) if vocab_path.suffix == '.json' else vocab_path.read_text(encoding='utf-8').splitlines()
    from artifact_utils import validate_topic_matrices
    validate_topic_matrices(data['theta'], data['beta'], data['vocab'], model_type)
    data['topic_words'] = [(i, [(data['vocab'][j], float(data['beta'][i,j])) for j in np.argsort(-data['beta'][i])[:20]]) for i in range(data['beta'].shape[0])]

    # Load BOW matrix (optional)
    if bow_dir:
        bow_file = bow_dir / 'bow_matrix.npy'
        if (bow_dir / 'bow_matrix.npz').is_file():
            data['bow_matrix'] = sparse.load_npz(bow_dir / 'bow_matrix.npz')
        elif bow_file.exists():
            data['bow_matrix'] = np.load(bow_file)
            print(f"✓ Loaded bow_matrix: {data['bow_matrix'].shape}")
    
    # Load timestamps (optional) - check in model_dir parent or evaluation_dir
    ts_file = model_dir.parent / 'timestamps.npy' if model_dir else None
    if ts_file and ts_file.exists():
        data['timestamps'] = np.load(ts_file, allow_pickle=False).astype('datetime64[ms]').astype(object)
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
    dpi=300,
    model_type='theta',
    num_topics=None,
    model_exp=None,
    data=None,
    formats=('png', 'pdf', 'svg')
):
    """
    Run all visualizations for ETM/Baseline results.
    
    Args:
        result_dir: Base result directory
        dataset: Dataset name
        mode: Training mode
        model_size: Model size for THETA (e.g., 0.6B)
        output_dir: Output directory for visualizations
        language: Language for labels ('en' or 'zh')
        dpi: DPI for saved figures
        model_type: 'theta' | 'lda' | 'etm' | 'ctm_combined' | 'ctm_zeroshot'
        num_topics: Number of topics (required for baseline models)
        model_exp: Specific experiment ID
    
    Returns:
        Path to output directory
    """
    formats = validate_export(dpi, formats)
    if data is None:
        # Load data with model type awareness
        data = load_visualization_data(
            result_dir, dataset, mode,
            model_size=model_size,
            model_exp=model_exp,
            model_type=model_type,
            num_topics=num_topics
        )
    
    is_theta = model_type == 'theta'
    
    # Determine output directory based on model type
    if output_dir is None:
        result_dir = Path(result_dir)
        
        if is_theta:
            # THETA: result/{dataset}/{model_size}/theta/visualization/
            if model_size:
                output_dir = result_dir / dataset / model_size / 'theta' / 'visualization' / language
            else:
                output_dir = result_dir / dataset / 'theta' / 'visualization' / language
        else:
            # Baseline: result/baseline/{dataset}/visualization/{model_type}/
            output_dir = result_dir / 'baseline' / dataset / 'visualization' / model_type / language
    
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
        dimension_values=data.get('dimension_values'),
        bow_matrix=data.get('bow_matrix'),
        training_history=data.get('training_history'),
        beta_over_time=data.get('beta_over_time'), time_slices_info=data.get('time_slices_info'),
        metrics=data.get('metrics'),
        output_dir=str(output_dir),
        language=language,
        dpi=dpi, formats=formats
    )
    
    # Generate all visualizations
    generator.generate_all()
    
    _run_additional_visualizations(data, output_dir, language, dpi, formats, model_type)

    # Generate summary report
    generate_summary_report(data, output_dir)
    export_manifest(output_dir, dpi, formats, data=data)
    if data.get('source_metadata'):
        (Path(output_dir) / 'source-metadata.json').write_text(json.dumps(data['source_metadata'], ensure_ascii=False, indent=2))
    
    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    return output_dir


def _run_additional_visualizations(data, output_dir, language, dpi, formats, model):
    """One native rendering path, with a durable outcome for every attempted chart."""
    from contextlib import redirect_stdout
    from io import StringIO
    from visualization.topic_visualizer import TopicVisualizer, generate_pyldavis_visualization
    root = Path(output_dir)
    global_dir = root / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)
    _export_chart_source_data(data, root)
    viz = TopicVisualizer(output_dir=str(global_dir), dpi=dpi, formats=formats, language=language)
    zh = language == 'zh'
    statuses = []

    def render(name, function, reason=None):
        from visualization.publication import selected_chart
        if not selected_chart(name):
            return
        if reason:
            statuses.append({'chart': name, 'status': 'skipped', 'detail': reason, 'files': []})
            return
        before = {p: p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
        capture = StringIO()
        error = None
        try:
            with redirect_stdout(capture): function()
            if '⚠' in capture.getvalue():
                error = capture.getvalue().strip()
        except Exception as exc:
            error = str(exc)
        files = [str(p.relative_to(root)) for p in root.rglob('*')
                 if p.is_file() and p.stat().st_mtime_ns != before.get(p)]
        statuses.append({'chart': name, 'status': 'failed' if error else ('generated' if files else 'skipped'),
                         'detail': error or capture.getvalue().strip() or ('No artifacts returned' if not files else ''),
                         'files': sorted(files)})
        print(f"  [{statuses[-1]['status']}] {name}" + (f': {error}' if error else ''))

    render('topic_words', lambda: viz.visualize_topic_words(data['topic_words'], num_words=10))
    render('topic_similarity', lambda: viz.visualize_topic_similarity(data['beta'], data['topic_words'],
        filename='主题相似度图.png' if zh else 'topic_similarity.png'))
    render('document_projection', lambda: viz.visualize_document_topics(data['theta'], labels=data.get('document_topics'),
        max_docs=5000, filename='文档主题UMAP图.png' if zh else 'doc_topic_umap.png'))
    render('wordclouds', lambda: viz.visualize_all_wordclouds(data['topic_words'], num_words=30))
    grid_words=[(i,[(data['vocab'][j],float(row[j])) for j in np.argsort(-row)[:80]
                     if np.isfinite(row[j]) and row[j]>0]) for i,row in enumerate(data['beta'])]
    render('wordcloud_grid', lambda: viz.visualize_wordcloud_grid(grid_words))
    generative = model not in {'nvdm', 'bertopic', 'dtm'}
    reason = None if generative else f'{model}: latent coordinates, c-TF-IDF or time-varying beta do not support this probability view'
    render('intertopic_distance', lambda: viz.visualize_intertopic_distance(data['theta'], data['beta']), reason)
    render('word_weights', lambda: viz.visualize_topic_word_frequency(data['beta'], data['topic_words']), reason)
    render('pyldavis', lambda: generate_pyldavis_visualization(theta=data['theta'], beta=data['beta'],
        bow_matrix=data.get('bow_matrix'), vocab=data['vocab'],
        output_path=str(global_dir / ('交互式主题可视化.html' if zh else 'pyldavis_interactive.html'))),
        reason or ('Interactive topic comparison requires at least two topics' if data['beta'].shape[0] < 2 else
                   'No observed BOW counts available' if data.get('bow_matrix') is None else None))
    if model == 'stm':
        render('stm_covariates', lambda: _run_stm_specific_visualizations(data, root, language, dpi, formats),
               'Covariates not exported' if data.get('covariates') is None else None)
    if model == 'dtm':
        render('dtm_word_evolution', lambda: _run_dtm_specific_visualizations(data, root, language, dpi, formats),
               'Time-specific beta not exported' if data.get('beta_over_time') is None else None)
    (root / 'additional-chart-status.json').write_text(json.dumps(statuses, ensure_ascii=False, indent=2))
    if any(item['status'] == 'failed' for item in statuses):
        export_manifest(root, dpi, formats, data)
        raise ValueError(f'Native chart export failed; inspect {root / "additional-chart-status.json"}')


def _export_chart_source_data(data, output_dir):
    """Persist the numeric inputs used by additional charts for non-visual interpretation."""
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    root = Path(output_dir)
    global_dir = root / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)
    word_rows = []
    for topic_id, words in data['topic_words']:
        rows = [
            {'rank': rank, 'word': word, 'weight': float(weight)}
            for rank, (word, weight) in enumerate(words, 1)
        ]
        word_rows.extend({'topic_id': int(topic_id) + 1, **row} for row in rows)
        topic_dir = root / 'topic' / f'topic_{int(topic_id) + 1}'
        topic_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(topic_dir / 'word_importance.csv', index=False)
    pd.DataFrame(word_rows).to_csv(global_dir / 'topic_word_weights.csv', index=False)

    beta = np.asarray(data['beta'], dtype=float)
    topic_ids = np.arange(1, beta.shape[0] + 1)
    pd.DataFrame(cosine_similarity(beta), index=topic_ids, columns=topic_ids).to_csv(
        global_dir / 'topic_beta_similarity.csv', index_label='topic_id')
    coordinates = PCA(n_components=2).fit_transform(beta) if min(beta.shape) >= 2 else np.zeros((beta.shape[0], 2))
    pd.DataFrame({
        'topic_id': topic_ids,
        'pc1': coordinates[:, 0],
        'pc2': coordinates[:, 1],
        'mean_topic_weight': np.asarray(data['theta'], dtype=float).mean(axis=0),
    }).to_csv(global_dir / 'intertopic_distance.csv', index=False)


def generate_summary_report(data, output_dir):
    """Generate a summary report of the visualization."""
    output_dir = Path(output_dir)
    
    report = []
    report.append("# THETA Topic Model Visualization Report")
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
        label = "Mean latent coordinate (not probability)" if data.get("theta_semantics") == "latent_coordinates" else "Strength"
        report.append(f"- **{label}**: {strength:.6f}")
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
    topics_dir = output_dir / 'topic'
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


def load_baseline_data(result_dir, dataset, model, num_topics=20, workspace_dir=None):
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
    if result_dir.name.startswith('exp_') or (result_dir / 'config.json').exists() or any(result_dir.glob('theta*.npy')):
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
    
    from artifact_utils import find_topic_matrix_pair, validate_topic_matrices
    theta_path, beta_path = find_topic_matrix_pair(model_dir, num_topics)
    data['theta'] = np.load(theta_path, allow_pickle=False)
    data['beta'] = np.load(beta_path, allow_pickle=False)
    num_topics = data['theta'].shape[1]
    matrix_dir = theta_path.parent
    data['matrix_dir'] = matrix_dir
    data['model_type'] = model
    print(f"✓ Loaded original theta {data['theta'].shape}, beta {data['beta'].shape}")

    # The worker uses an explicit preprocessing workspace outside the result tree.
    if workspace_dir is not None:
        data_exp_dir = Path(workspace_dir)
    vocab_paths = [matrix_dir / 'vocab.json', *([data_exp_dir / 'vocab.json'] if data_exp_dir else []), model_dir / 'bow/vocab.json', dataset_dir / 'vocab.json']
    if model == 'bertopic': vocab_paths = [matrix_dir / 'vocab.json']
    vocab_path = next((file for file in vocab_paths if file.is_file()), None)
    if vocab_path is None:
        raise ValueError('Actual model vocabulary unavailable; refusing placeholder or mismatched words')
    data['vocab'] = json.loads(vocab_path.read_text(encoding='utf-8'))
    validate_topic_matrices(data['theta'], data['beta'], data['vocab'], model)
    def optional_file(name):
        return next((directory / name for directory in [matrix_dir, model_dir / 'topicwords', model_dir, dataset_dir, data_exp_dir] if directory is not None and (directory / name).is_file()), matrix_dir / name)
    data['topic_words'] = [
        (i, [(data['vocab'][idx], float(data['beta'][i, idx])) for idx in np.argsort(-data['beta'][i])[:20]])
        for i in range(data['beta'].shape[0])]

    # Keep BOW aligned with this model's own vocabulary (BERTopic exports a different one).
    bow_paths = [matrix_dir] if model == 'bertopic' else [matrix_dir, data_exp_dir, model_dir / 'bow', dataset_dir]
    for directory in bow_paths:
        if directory is None: continue
        if (directory / 'bow_matrix.npz').is_file(): data['bow_matrix'] = sparse.load_npz(directory / 'bow_matrix.npz'); break
        if (directory / 'bow_matrix.npy').is_file(): data['bow_matrix'] = np.load(directory / 'bow_matrix.npy', allow_pickle=False); break
    if data.get('bow_matrix') is not None and data['bow_matrix'].shape != (data['theta'].shape[0], len(data['vocab'])):
        raise ValueError('BOW rows/vocabulary are not aligned to the result matrices')
    metrics_path = optional_file(f'metrics_k{num_topics}.json')
    if not metrics_path.exists(): metrics_path = optional_file(f'evaluation/metrics_k{num_topics}.json')
    if metrics_path.is_file(): data['metrics'] = json.loads(metrics_path.read_text())
    history_path = optional_file(f'training_history_k{num_topics}.json')
    if not history_path.exists(): history_path = optional_file('training_history.json')
    data['training_history'] = json.loads(history_path.read_text()) if history_path.is_file() else None
    data['timestamps'] = None
    data['topic_embeddings'] = None
    for name in ['source_rows', 'document_topics']:
        file = optional_file(name + '.npy')
        if file.is_file(): data[name] = np.load(file, allow_pickle=False)
    time_file, index_file = optional_file('time_slices.json'), optional_file('time_indices.npy')
    if time_file.is_file() and index_file.is_file():
        info = json.loads(time_file.read_text()); indices = np.load(index_file, allow_pickle=False)
        years = info.get('unique_times', [])
        if len(indices) != len(data['theta']) or (indices < 0).any() or (indices >= len(years)).any(): raise ValueError('Time indices do not align to theta')
        data['timestamps'] = np.array([datetime(int(years[int(index)]), 1, 1) for index in indices])
        data['time_slices_info'] = info
    evolution_file = optional_file(f'beta_over_time_k{num_topics}.npy')
    if evolution_file.is_file():
        data['beta_over_time'] = np.load(evolution_file, allow_pickle=False)
        if data['beta_over_time'].ndim != 3 or data['beta_over_time'].shape[1:] != data['beta'].shape or not np.isfinite(data['beta_over_time']).all(): raise ValueError('Invalid temporal beta axes or values')
        if data.get('time_slices_info') and len(data['beta_over_time']) != len(data['time_slices_info']['unique_times']): raise ValueError('Temporal beta/time axes do not match')
        data['plot_scope'] = 'DTM theta covers all modeled rows; static beta/keyword plots describe the last time slice. Temporal word plots use beta_over_time and the recorded periods.'
    document_topics = data.get('document_topics')
    if model == 'bertopic':
        if document_topics is None or len(document_topics) != len(data['theta']):
            raise ValueError('BERTopic 导出不完整：缺少或不匹配 document_topics.npy，无法确定离群点与文档主题分配；不能从软权重猜测真实聚类标签')
        if any(not str(word).strip() for word in data['vocab']):
            raise ValueError('BERTopic 词表含空白占位词，不能作为真实主题词生成图表')
        data['theta_semantics'] = 'membership_mass_excluding_outliers'
        data['beta_semantics'] = 'normalized_selected_ctfidf_weights'

    # Load STM covariate data
    if model == 'stm':
        # Sidecars belong to the exact resolved theta/beta pair, including direct leaf paths.
        stm_model_dir = matrix_dir

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


def _run_stm_specific_visualizations(data, output_dir, language='zh', dpi=300, formats=('png', 'pdf', 'svg')):
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

    setup_style(language)
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
        source_frame = data.get('source_frame')
        source_column = data.get('source_covariate_column')
        if source_frame is not None and source_column in source_frame.columns:
            le = LabelEncoder()
            le.fit(source_frame[source_column].fillna('unknown').astype(str))
            platform_labels = dict(enumerate(le.classes_))
        config_path = output_dir.parent / 'config.json'
        if not platform_labels and config_path.exists():
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

    platform_labels.update(data.get('covariate_value_labels', {}))
    unique_vals = sorted(set(covariates[:, 0].astype(int).tolist()))
    cov_labels = [platform_labels.get(v, f'分组{v}' if zh else f'cat_{v}') for v in unique_vals]

    def topic_label(tid):
        if tid < len(topic_words):
            words = topic_words[tid][1]
            if words:
                return f"T{tid+1}:{words[0][0]}"
        return f"主题 {tid+1}" if zh else f"Topic {tid+1}"

    topic_labels = [topic_label(i) for i in range(K)]

    mean_theta = np.zeros((len(unique_vals), K))
    for i, val in enumerate(unique_vals):
        mask = covariates[:, 0].astype(int) == val
        if mask.sum() > 0:
            mean_theta[i] = theta[mask].mean(axis=0)

    # Keep every group in the downloadable evidence, but bound static previews.
    # Thousands of source/category levels otherwise produce metre-high figures.
    counts = [int((covariates[:, 0].astype(int) == value).sum()) for value in unique_vals]
    group_table = pd.DataFrame(mean_theta, columns=[f'T{k+1}' for k in range(K)])
    group_table.insert(0, 'group', cov_labels)
    group_table.insert(1, 'n_documents', counts)
    group_table.to_csv(global_dir / 'stm_group_topic_means.csv', index=False)
    total_groups = len(unique_vals)
    preview_ids = sorted(range(total_groups), key=lambda i: (-counts[i], i))[:24]
    if total_groups > 24:
        unique_vals = [unique_vals[i] for i in preview_ids]
        cov_labels = [cov_labels[i] for i in preview_ids]
        mean_theta = mean_theta[preview_ids]
    coverage = (f'（展示文档数最多的24组，共{total_groups}组；完整数据见CSV）' if zh else
                f'(24 largest groups of {total_groups}; complete data in CSV)') if total_groups > 24 else ''
    def figure_title(title):
        return title + ('\n' + coverage if coverage else '')

    # Chart 1: 协变量-主题关联热力图
    try:
        fig, ax = plt.subplots(figsize=(max(6.8, K * .55), max(2.6, len(unique_vals) * .45)))
        im = ax.imshow(mean_theta, aspect='auto', cmap='Blues',vmin=0)
        plt.colorbar(im, ax=ax, label='平均主题占比' if zh else 'Mean Topic Proportion')
        ax.set_xticks(range(K)); ax.set_xticklabels([f'T{i+1}' for i in range(K)], rotation=0, fontsize=8)
        ax.set_yticks(range(len(unique_vals))); ax.set_yticklabels(cov_labels, fontsize=9)
        ax.set_title(figure_title('协变量-主题关联热力图' if zh else 'Covariate-Topic Heatmap'), fontsize=12, fontweight='bold')
        for i in range(len(unique_vals)):
            for j in range(K):
                ax.text(j, i, f'{mean_theta[i,j]:.1%}', ha='center', va='center', fontsize=6,
                        color='white' if im.norm(mean_theta[i,j]) > .6 else '#202020')
        plt.tight_layout()
        save_figure(plt.gcf(), global_dir / ('协变量主题关联热力图.png' if zh else 'covariate_topic_heatmap.png'), dpi=dpi, formats=formats, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 协变量主题关联热力图.png")
    except Exception as e:
        print(f"  ⚠ heatmap: {e}")

    # Chart 2: 各平台主题分布堆积图
    try:
        colors = [COLORS[i % len(COLORS)] for i in range(K)]
        fig, ax = plt.subplots(figsize=(8, max(2.5,len(unique_vals)*.5,K*.17)))
        bottom = np.zeros(len(unique_vals))
        for k in range(K):
            ax.barh(cov_labels, mean_theta[:, k], left=bottom, color=colors[k], label=topic_labels[k], height=.55, edgecolor='white',linewidth=.5)
            for i,value in enumerate(mean_theta[:,k]):
                if value>.08:ax.text(bottom[i]+value/2,i,f'{value:.0%}',ha='center',va='center',color='white',fontsize=8)
            bottom += mean_theta[:, k]
        ax.set_title(figure_title('分组主题构成' if zh else 'Topic composition by group'), fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=7)
        from matplotlib.ticker import PercentFormatter
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set_xlim(0,1)
        plt.xticks(rotation=0, fontsize=8); plt.tight_layout()
        save_figure(plt.gcf(), global_dir / ('各平台主题分布堆积图.png' if zh else 'platform_topic_stacked.png'), dpi=dpi, formats=formats, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 各平台主题分布堆积图.png")
    except Exception as e:
        print(f"  ⚠ stacked bar: {e}")

    # Chart 3: 平台主题偏好偏差图
    try:
        deviation = mean_theta - theta.mean(axis=0)[np.newaxis, :]
        vmax = np.abs(deviation).max()
        fig, ax = plt.subplots(figsize=(max(6.8, K * .65), max(2.6, len(unique_vals) * .45)))
        im = ax.imshow(deviation, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='偏差（相对全局均值）' if zh else 'Deviation from Global Mean')
        ax.set_xticks(range(K)); ax.set_xticklabels([f'T{i+1}' for i in range(K)], rotation=0, fontsize=8)
        ax.set_yticks(range(len(unique_vals))); ax.set_yticklabels(cov_labels, fontsize=9)
        ax.set_title(figure_title('分组主题偏差（红=高于均值，蓝=低于均值）' if zh else 'Topic deviation by group'), fontsize=11, fontweight='bold')
        for i in range(len(unique_vals)):
            for j in range(K):
                ax.text(j, i, f'{deviation[i,j]:+.1%}', ha='center', va='center', fontsize=8, color='white' if vmax and abs(deviation[i,j])/vmax>.6 else '#202020')
        plt.tight_layout()
        save_figure(plt.gcf(), global_dir / ('平台主题偏好偏差图.png' if zh else 'platform_topic_deviation.png'), dpi=dpi, formats=formats, bbox_inches='tight', facecolor='white')
        plt.close(); print(f"  ✓ 平台主题偏好偏差图.png")
    except Exception as e:
        print(f"  ⚠ deviation: {e}")

    # Chart 4: paginate groups instead of compressing an arbitrary number into one row.
    try:
        from matplotlib.ticker import PercentFormatter
        for start in range(0,len(unique_vals),6):
            group_ids=list(range(start,min(start+6,len(unique_vals))))
            cols=min(3,len(group_ids));rows=(len(group_ids)+cols-1)//cols
            fig,axes=plt.subplots(rows,cols,figsize=(7.15,rows*2.3+.4),sharex=True,squeeze=False,layout='constrained')
            for ax,i in zip(axes.flat,group_ids):
                top5=np.argsort(-mean_theta[i])[:5][::-1]
                ax.barh(range(len(top5)),mean_theta[i,top5],color=[COLORS[j%len(COLORS)] for j in top5],height=.55)
                ax.set_xlim(0,max(mean_theta.max()*1.12,.01));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
                ax.set_yticks(range(len(top5)),[topic_labels[j] for j in top5],fontsize=7)
                ax.set_title(cov_labels[i],fontsize=9);ax.set_xlabel('平均权重' if zh else 'Mean weight')
            for ax in list(axes.flat)[len(group_ids):]:ax.set_visible(False)
            fig.suptitle(figure_title(('分组 Top-5 主题' if zh else 'Top-5 topics by group')+
                         f' · {start+1}–{start+len(group_ids)}'),fontsize=10)
            stem='各平台主导主题' if zh else 'platform_dominant_topics'
            filename=stem+(f'_{start//6+1}' if start else '')+'.png'
            save_figure(fig,global_dir/filename,dpi=dpi,formats=formats);plt.close(fig)
    except Exception as e:
        print(f'  ⚠ dominant topics: {e}')

    # Chart 5: Gamma 系数图（若已保存）
    Gamma = data.get('Gamma')
    if Gamma is not None:
        try:
            n_cov = Gamma.shape[0] - 1
            fig, axes = plt.subplots(1, max(1,n_cov),figsize=(7.1,max(2.5,(K-1)*.28)),squeeze=False,layout='constrained')
            axes=axes.ravel()
            for c in range(n_cov):
                ax = axes[c]
                coefs = Gamma[c + 1, :]
                cname = covariate_names[c] if c < len(covariate_names) else f'cov_{c}'
                ax.barh(range(K-1),coefs,color=[COLORS[1] if v>0 else COLORS[0] for v in coefs],height=.55)
                ax.axvline(0,color='#727B82',linewidth=.8)
                ax.set_yticks(range(K-1),[f'T{i+1}' for i in range(K-1)]);ax.invert_yaxis()
                ax.set_xlabel(('系数；参照主题 ' if zh else 'Coefficient; reference topic ')+f'T{K}')
                ax.set_title(f'协变量效应：{cname}' if zh else f'Covariate Effect: {cname}', fontsize=10, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            save_figure(plt.gcf(), global_dir / ('STM协变量Gamma系数图.png' if zh else 'stm_gamma_coefficients.png'), dpi=dpi, formats=formats, bbox_inches='tight', facecolor='white')
            plt.close(); print(f"  ✓ STM协变量Gamma系数图.png")
        except Exception as e:
            print(f"  ⚠ gamma: {e}")

    # theta was fitted using these same covariates; a post-fit one-way ANOVA
    # treats estimated dependent quantities as independently observed outcomes.
    # Keep descriptive evidence and leave inferential uncertainty to a valid STM procedure.
    print('[SKIP] STM post-fit ANOVA: these covariates already entered topic estimation; no independent effect test or model uncertainty is available. Descriptive group means and Gamma estimates retained.')


def _run_dtm_specific_visualizations(data, output_dir, language='en', dpi=300, formats=('png', 'pdf', 'svg')):
    """Render actual beta_over_time word trajectories; common plots use the shared generator."""
    from pathlib import Path
    import numpy as np
    
    output_dir = Path(output_dir)
    global_dir = output_dir / 'global'
    global_dir.mkdir(parents=True, exist_ok=True)
    
    beta = data.get('beta_over_time')
    years = data.get('time_slices_info', {}).get('unique_times', [])
    if beta is None or len(years) != len(beta):
        print('[Skip] DTM word evolution requires aligned beta_over_time and actual time slices')
        return
    # Adapt the actual exported time/topic/word values to the existing renderer's input.
    evolution = {}
    for k in range(beta.shape[1]):
        selected_words = np.argsort(-beta[:, k].mean(axis=0))[:10]
        evolution[str(k+1)] = {str(year): [(data['vocab'][i], float(beta[t, k, i])) for i in selected_words]
                              for t, year in enumerate(years)}
    _generate_topic_word_evolution(evolution, global_dir, language, dpi, formats)


def _generate_topic_word_evolution(topic_evolution, output_dir, language='en', dpi=300, formats=('png', 'pdf', 'svg')):
    """Generate DTM topic word evolution visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    output_dir = Path(output_dir)
    n_topics = len(topic_evolution)
    
    import pandas as pd
    records = []
    items = list(topic_evolution.items())
    for start in range(0, n_topics, 6):
        page = items[start:start+6]
        cols = min(2, len(page)); rows = (len(page)+cols-1)//cols
        fig, axes = plt.subplots(rows, cols, figsize=(7.2, rows*2.1), squeeze=False, layout='constrained')
        for ax, (topic_id, time_data) in zip(axes.flat, page):
            times = list(time_data)  # Preserve the model's actual ordered time slices.
            # Ordered selection is deterministic, not a set/hash iteration.
            words = list(dict.fromkeys(word for t in times for word, _ in time_data[t]))[:5]
            for i, word in enumerate(words):
                values = [dict(time_data[t]).get(word, np.nan) for t in times]
                ax.plot(range(len(times)), values, marker=['o','s','^','D','v'][i],
                        color=COLORS[i], label=word, linewidth=1, markersize=2.5)
                records.extend({'topic': topic_id, 'time': t, 'word': word, 'weight': value}
                               for t, value in zip(times, values))
            ax.set_title(f'T{topic_id}', loc='left', weight='bold')
            ax.set_xticks(range(len(times)), times, rotation=45, ha='right')
            ax.set_xlabel('时间' if language == 'zh' else 'Time')
            ax.set_ylabel('词权重' if language == 'zh' else 'Word weight')
            ax.legend(loc='upper center',bbox_to_anchor=(.5,-.36),ncol=3,fontsize=7,handlelength=1.2,columnspacing=.8)
        for ax in list(axes.flat)[len(page):]: ax.axis('off')
        filename = 'topic_word_evolution' + (f'_{start//6+1}' if start else '') + '.png'
        save_figure(fig, output_dir / filename, dpi=dpi, formats=formats)
        plt.close(fig)
    pd.DataFrame(records).to_csv(output_dir/'dtm_word_evolution.csv', index=False)



def run_baseline_visualization(
    result_dir,
    dataset,
    model,
    num_topics=20,
    output_dir=None,
    language='both',  # Changed default to 'both' for bilingual output
    dpi=300,
    workspace_dir=None,
    data=None,
    formats=('png', 'pdf', 'svg')
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
    formats = validate_export(dpi, formats)
    # Load data
    if data is None:
        data = load_baseline_data(result_dir, dataset, model, num_topics, workspace_dir=workspace_dir)

    latent = model == 'nvdm'
    plot_data = data
    if model == 'bertopic':
        # Plot conditional membership among assigned clusters; preserve original matrices and noise labels.
        labels = np.asarray(data['document_topics'])
        mass = data['theta'].sum(axis=1)
        keep = (labels >= 0) & (mass > 0)
        if not keep.any(): raise ValueError('BERTopic has no assigned documents to visualize')
        plot_data = dict(data, theta=data['theta'][keep] / mass[keep, None])
        for key in ['timestamps', 'dimension_values', 'bow_matrix']:
            if data.get(key) is not None: plot_data[key] = np.asarray(data[key])[keep] if key != 'bow_matrix' else data[key][keep]
        data['plot_scope'] = f"Conditional membership of {int(keep.sum())}/{len(keep)} assigned documents; {int((~keep).sum())} outlier/zero-mass rows excluded from share charts. Original theta and assignments are unchanged. Beta contains normalized selected c-TF-IDF weights, not generative word probabilities."
        print('[Scope] ' + data['plot_scope'])
    if latent:
        data['theta_semantics'] = 'latent_coordinates'
        print('[Skip] NVDM theta contains latent coordinates, not probabilities; probability-share and pyLDAvis charts are not applicable.')

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
        lang_output = output_dir / lang if len(languages) > 1 else output_dir
        setup_style(lang)
        print(f"\n{'='*60}")
        print(f"Generating visualizations ({lang.upper()})")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")
        
        generator = VisualizationGenerator(
            theta=plot_data['theta'],
            beta=data['beta'],
            vocab=data['vocab'],
            topic_words=data['topic_words'],
            topic_embeddings=data.get('topic_embeddings'),
            timestamps=plot_data.get('timestamps'),
            dimension_values=plot_data.get('dimension_values'),
            bow_matrix=plot_data.get('bow_matrix'),
            training_history=data.get('training_history'),
            beta_over_time=data.get('beta_over_time'), time_slices_info=data.get('time_slices_info'),
            metrics=data.get('metrics'),
            output_dir=str(lang_output),
            language=lang,
            dpi=dpi, formats=formats
        )
        
        if latent:
            (lang_output/'chart-status.json').write_text(json.dumps([{'chart':'sankey_diagram','status':'skipped',
                'detail':'NVDM 输出有正负值的潜在坐标，不是非负主题权重；不适用权重分配桑基图。'}],ensure_ascii=False,indent=2))
            generator.generate_topic_table(strength_label='mean_latent_coordinate')
            # Reuse only existing charts whose inputs don't require probability theta.
            for topic_idx in range(data['beta'].shape[0]):
                generator.generate_topic_word_importance(topic_idx)
            generator.generate_training_convergence()
        else:
            generator.generate_all()
        
        if model == 'bertopic':
            generator.theta = data['theta']
            generator.generate_topic_table(strength_label='mean_membership_mass')

        _run_additional_visualizations(data, lang_output, lang, dpi, formats, model)

    generate_summary_report(data, output_dir)
    export_manifest(output_dir, dpi, formats, data=data)
    if data.get('source_metadata'):
        (Path(output_dir) / 'source-metadata.json').write_text(json.dumps(data['source_metadata'], ensure_ascii=False, indent=2))
    
    print(f"\n{'='*60}")
    print(f"✓ Visualizations saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    if data.get('plot_scope'):
        with (output_dir / 'README.md').open('a', encoding='utf-8') as file:
            file.write('\n\n## Plot scope\n' + data['plot_scope'] + '\n')
            file.write('pyLDAvis requires generative probabilities with a single aligned beta; it is not applicable to c-TF-IDF or a time-varying beta.\n')
    return output_dir


def run_all_baseline_visualizations(
    result_dir=None,
    datasets=None,
    models=None,
    num_topics=20,
    language='en',
    dpi=300,
    formats=('png', 'pdf', 'svg')
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
        datasets = sorted(p.name for p in Path(result_dir or 'result/baseline').iterdir() if p.is_dir())
    if models is None:
        models = ['lda', 'hdp', 'btm', 'stm', 'dtm', 'etm', 'nvdm', 'gsm', 'prodlda', 'ctm', 'ctm_zeroshot', 'ctm_combined', 'bertopic']
    
    print("="*70)
    print("Running Visualizations for All Baseline Models")
    print("="*70)
    
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for model in models:
            print(f"\n>>> {dataset} / {model}")
            try:
                run_baseline_visualization(result_dir, dataset, model, num_topics, language=language, dpi=dpi, formats=formats)
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


def attach_source_metadata(data, source_file, training_data, text_column, time_column=None, group_column=None):
    """Prove row alignment against training input before attaching research labels."""
    import pandas as pd
    from artifact_utils import sha256_file
    if not training_data or not text_column:
        raise ValueError('Source metadata requires --training_data and --text_column; row counts alone do not prove alignment')
    source = Path(source_file)
    frame = pd.read_excel(source) if source.suffix.lower() in {'.xlsx', '.xls'} else pd.read_csv(source)
    recorded = pd.read_csv(training_data, keep_default_na=False)
    raw = frame[text_column].fillna('').astype(str).to_numpy()
    expected = recorded['text'].astype(str).to_numpy()
    rows = data.get('source_rows')
    if rows is not None:
        rows = np.asarray(rows)
        if not np.issubdtype(rows.dtype, np.integer) or (rows < 0).any() or (rows >= len(raw)).any():
            raise ValueError('Invalid source row mapping')
        raw = raw[rows]
        if len(expected) != len(raw): expected = expected[rows]
        frame = frame.iloc[rows].reset_index(drop=True)
    if len(raw) != len(data['theta']) or not np.array_equal(raw, expected):
        raise ValueError('Original texts do not match the recorded training input row by row')
    if time_column:
        parsed = pd.to_datetime(frame[time_column], errors='coerce', format='mixed')
        data['timestamps'] = np.array(parsed.dt.to_pydatetime())
    if group_column:
        values = frame[group_column].fillna('(missing)').astype(str)
        # Top categories selected by document count, with the rest explicitly grouped.
        retained = values.value_counts().head(8).index
        data['dimension_values'] = np.array(values.where(values.isin(retained), '(other sources)'))
    data['source_metadata'] = {'source': str(source.resolve()), 'sha256': sha256_file(source),
        'trainingDataSha256': sha256_file(training_data), 'textColumn': text_column,
        'timeColumn': time_column, 'groupColumn': group_column, 'rows': len(raw),
        'grouping': '8 largest groups by document count; all remaining groups combined',
        'missingTimeRows': int(pd.isna(data.get('timestamps', [])).sum())}
    return data


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
                        choices=['lda', 'hdp', 'stm', 'btm', 'etm', 'ctm', 'ctm_zeroshot', 'ctm_combined', 'dtm', 'nvdm', 'gsm', 'prodlda', 'bertopic'],
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
    
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf', 'svg'],
                        help='Export formats; PNG uses --dpi, PDF/SVG retain vector lines/text')
    parser.add_argument('--workspace_dir', help='Exact preprocessing workspace for this experiment')
    parser.add_argument('--source_file', help='Original CSV/XLSX for verified metadata alignment')
    parser.add_argument('--training_data', help='Normalized CSV used in training, with its text column named text')
    parser.add_argument('--text_column', help='Original source text column')
    parser.add_argument('--time_column', help='Original source timestamp column')
    parser.add_argument('--group_column', help='Original source categorical column')
    args = parser.parse_args()
    validate_export(args.dpi, args.formats)
    if args.all and args.source_file:
        parser.error('--source_file requires selecting a single experiment')

    
    if args.baseline:
        # Baseline model visualization
        if args.all:
            run_all_baseline_visualizations(
                result_dir=args.result_dir or os.path.join(os.environ.get('RESULT_DIR', 'result'), 'baseline'),
                num_topics=args.num_topics,
                language=args.language,
                dpi=args.dpi, formats=args.formats
            )
        elif args.dataset and args.model:
            data = load_baseline_data(args.result_dir, args.dataset, args.model, args.num_topics, workspace_dir=args.workspace_dir)
            if args.source_file:
                attach_source_metadata(data, args.source_file, args.training_data, args.text_column, args.time_column, args.group_column)
            run_baseline_visualization(
                result_dir=args.result_dir or os.path.join(os.environ.get('RESULT_DIR', 'result'), 'baseline'),
                dataset=args.dataset,
                model=args.model,
                data=data,
                workspace_dir=args.workspace_dir,
                num_topics=args.num_topics,
                output_dir=args.output_dir,
                language=args.language,
                dpi=args.dpi, formats=args.formats
            )
        else:
            parser.error("Baseline mode requires --all or both --dataset and --model")
    else:
        # THETA model visualization
        if not args.result_dir or not args.dataset or not args.mode:
            parser.error("THETA mode requires --result_dir, --dataset, and --mode")
        data = load_visualization_data(args.result_dir, args.dataset, args.mode, model_size=args.model_size)
        if args.source_file:
            attach_source_metadata(data, args.source_file, args.training_data, args.text_column, args.time_column, args.group_column)
        run_all_visualizations(
            data=data,
            result_dir=args.result_dir,
            dataset=args.dataset,
            mode=args.mode,
            model_size=args.model_size,
            output_dir=args.output_dir,
            language=args.language,
            dpi=args.dpi, formats=args.formats,
            model_type='theta',
            num_topics=args.num_topics
        )


if __name__ == '__main__':
    main()
