"""
ETM Configuration Management

Centralized configuration for all ETM pipeline components.

Multi-level Priority Parameter System:
    P1 (Highest): Command-line arguments
    P2: Environment variables / .env file
    P3: Script default values
    P4 (Lowest): Code hardcoded defaults

Environment Variables:
    PROJECT_ROOT: Project root directory (auto-detected if not set)
    ETM_DIR: ETM module directory
    DATA_DIR: Data directory
    RESULT_DIR: Result directory
    EMBEDDING_MODELS_DIR: Embedding models directory
    WORKSPACE_DIR: Workspace directory
    
Hyperparameter Environment Variables (optional):
    NUM_TOPICS: Number of topics (default: 20)
    EPOCHS: Training epochs (default: 200)
    BATCH_SIZE: Batch size (default: 64)
    VOCAB_SIZE: Vocabulary size (default: 5000)
    LEARNING_RATE: Learning rate (default: 0.002)
    HIDDEN_DIM: Hidden dimension (default: 1024)
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# =============================================================================
# .env Auto-Loading - Ensures Python can sense .env even when run directly
# =============================================================================

def _load_dotenv_safe():
    """
    Safely load .env file. Works even if python-dotenv is not installed.
    Priority: Already set env vars > .env file values
    """
    # First, auto-detect project root
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        return
    
    try:
        # Try using python-dotenv if available
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)  # Don't override existing env vars
    except ImportError:
        # Fallback: Manual .env parsing (same logic as env_setup.sh)
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    # Parse key=value
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        # Only set if not already defined (priority: existing > .env)
                        if key and key not in os.environ:
                            os.environ[key] = value
        except Exception:
            pass  # Silently ignore .env parsing errors

# Load .env at module import time
_load_dotenv_safe()


# =============================================================================
# Project Root Detection
# =============================================================================

def _get_project_root() -> Path:
    """
    Get PROJECT_ROOT from environment or auto-detect from this file's location.
    This file is at PROJECT_ROOT/src/models/config.py
    """
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"])
    # Auto-detect: this file is at src/models/config.py, so go up three levels
    return Path(__file__).resolve().parent.parent.parent


# Project root (auto-detected or from environment)
PROJECT_ROOT = _get_project_root()


# =============================================================================
# Global Path Management - Solves "path not found" issues once and for all
# =============================================================================

def get_absolute_path(
    env_var: str, 
    default_suffix: str, 
    base: Path = None,
    must_exist: bool = False
) -> Path:
    """
    Get absolute path with multi-level priority:
    P1: Environment variable (if set and non-empty)
    P2: Base path + default_suffix
    
    Args:
        env_var: Environment variable name (e.g., 'RESULT_DIR')
        default_suffix: Default path suffix relative to base (e.g., 'result')
        base: Base path for default (default: PROJECT_ROOT)
        must_exist: If True, raise error if path doesn't exist
        
    Returns:
        Absolute Path object
        
    Example:
        >>> get_absolute_path('RESULT_DIR', 'result')
        PosixPath('./result')
        
        >>> os.environ['RESULT_DIR'] = '/custom/path'
        >>> get_absolute_path('RESULT_DIR', 'result')
        PosixPath('/custom/path')
    """
    if base is None:
        base = PROJECT_ROOT
    
    # Priority 1: Environment variable
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value)
        # Convert relative to absolute based on PROJECT_ROOT
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        # Priority 2: Default suffix
        path = base / default_suffix
    
    # Resolve to absolute path
    path = path.resolve()
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path} (env_var={env_var})")
    
    return path


# =============================================================================
# Multi-level Priority Parameter Resolution
# =============================================================================

# =============================================================================
# Physical Safety Limits - Prevent GPU OOM and system crashes
# =============================================================================
HARD_LIMIT_TOKEN = 8192  # Maximum token length to prevent GPU memory explosion
MIN_DOCUMENT_COUNT = 5   # Minimum documents required for statistical significance

# Heuristic defaults for beginners - "just works" out of the box
HEURISTIC_DEFAULTS = {
    "num_topics": 20,
    "epochs": 100,
    "batch_size": 64,
    "vocab_size": 5000,
    "hidden_dim": 1024,
    "learning_rate": 0.002,
    "min_df": 2,
    "max_df_ratio": 0.7,
    "kl_start": 0.0,
    "kl_end": 1.0,
    "kl_warmup": 50,
    "patience": 10,
    "max_embed_length": 512,  # Default embedding max length
}

# Environment variable name mapping (param_name -> ENV_VAR_NAME)
PARAM_ENV_MAPPING = {
    "num_topics": "NUM_TOPICS",
    "epochs": "EPOCHS",
    "batch_size": "BATCH_SIZE",
    "vocab_size": "VOCAB_SIZE",
    "hidden_dim": "HIDDEN_DIM",
    "learning_rate": "LEARNING_RATE",
    "min_df": "MIN_DF",
    "max_df_ratio": "MAX_DF_RATIO",
    "kl_start": "KL_START",
    "kl_end": "KL_END",
    "kl_warmup": "KL_WARMUP",
    "patience": "PATIENCE",
    "model_size": "MODEL_SIZE",
    "mode": "MODE",
    "language": "LANGUAGE",
}


def resolve_param(
    param_name: str,
    cli_value: Any = None,
    default_value: Any = None,
    param_type: type = None
) -> Any:
    """
    Resolve parameter value with multi-level priority:
    P1 (Highest): CLI value (if explicitly provided, not None)
    P2: Environment variable
    P3: Provided default value
    P4 (Lowest): Heuristic default
    
    Args:
        param_name: Parameter name (e.g., 'num_topics')
        cli_value: Value from command line (None if not provided)
        default_value: Script/code default value
        param_type: Expected type for conversion (int, float, str, bool)
        
    Returns:
        Resolved parameter value
        
    Example:
        >>> resolve_param('num_topics', cli_value=30)  # CLI wins
        30
        >>> os.environ['NUM_TOPICS'] = '50'
        >>> resolve_param('num_topics', cli_value=None)  # ENV wins
        50
        >>> resolve_param('num_topics', cli_value=None, default_value=25)  # default wins
        25
    """
    # P1: CLI value (highest priority)
    if cli_value is not None:
        return cli_value
    
    # P2: Environment variable
    env_var = PARAM_ENV_MAPPING.get(param_name, param_name.upper())
    env_value = os.environ.get(env_var)
    if env_value is not None and env_value != "":
        # Type conversion
        if param_type is None:
            # Infer type from default or heuristic
            if default_value is not None:
                param_type = type(default_value)
            elif param_name in HEURISTIC_DEFAULTS:
                param_type = type(HEURISTIC_DEFAULTS[param_name])
            else:
                param_type = str
        
        try:
            if param_type == bool:
                return env_value.lower() in ('true', '1', 'yes', 'on')
            elif param_type == float:
                return float(env_value)
            elif param_type == int:
                return int(env_value)
            else:
                return env_value
        except (ValueError, TypeError):
            pass  # Fall through to defaults
    
    # P3: Provided default value
    if default_value is not None:
        return default_value
    
    # P4: Heuristic default (lowest priority)
    return HEURISTIC_DEFAULTS.get(param_name)


def get_env_param(param_name: str, default: Any = None, param_type: type = None) -> Any:
    """
    Shorthand for getting a parameter from environment with type conversion.
    
    Args:
        param_name: Parameter name (will be converted to uppercase for env lookup)
        default: Default value if not found
        param_type: Type to convert to (inferred from default if not provided)
        
    Returns:
        Parameter value
    """
    return resolve_param(param_name, cli_value=None, default_value=default, param_type=param_type)


class PriorityArgParser:
    """
    Argument parser wrapper that respects multi-level priority:
    CLI > ENV > argparse default > heuristic default
    
    Usage:
        parser = PriorityArgParser(description='My script')
        parser.add_argument('--num_topics', type=int, help='Number of topics')
        parser.add_argument('--epochs', type=int, help='Training epochs')
        args = parser.parse_args()
        
        # args.num_topics will be resolved with priority:
        # 1. --num_topics 30 (CLI)
        # 2. NUM_TOPICS=50 (ENV)
        # 3. HEURISTIC_DEFAULTS['num_topics'] = 20
    """
    
    def __init__(self, *args, **kwargs):
        self._parser = argparse.ArgumentParser(*args, **kwargs)
        self._priority_params = {}  # param_name -> (type, default)
    
    def add_argument(self, *args, **kwargs):
        """Add argument with priority resolution support."""
        # Extract param name from args
        param_name = None
        for arg in args:
            if arg.startswith('--'):
                param_name = arg[2:].replace('-', '_')
                break
        
        # Store type and default for priority resolution
        if param_name:
            param_type = kwargs.get('type', str)
            # Don't set argparse default - we'll resolve it ourselves
            argparse_default = kwargs.pop('default', None)
            self._priority_params[param_name] = (param_type, argparse_default)
        
        return self._parser.add_argument(*args, **kwargs)
    
    def parse_args(self, args=None):
        """Parse arguments with priority resolution."""
        parsed = self._parser.parse_args(args)
        
        # Apply priority resolution to each registered param
        for param_name, (param_type, argparse_default) in self._priority_params.items():
            cli_value = getattr(parsed, param_name, None)
            resolved = resolve_param(
                param_name,
                cli_value=cli_value,
                default_value=argparse_default,
                param_type=param_type
            )
            setattr(parsed, param_name, resolved)
        
        return parsed
    
    def __getattr__(self, name):
        """Delegate to underlying parser."""
        return getattr(self._parser, name)

# =============================================================================
# Path Constants - Using get_absolute_path for robust path resolution
# =============================================================================

# Base paths - all derived from PROJECT_ROOT or environment variables
BASE_DIR = get_absolute_path("PROJECT_ROOT", "")

# Source directory
SRC_DIR = get_absolute_path("SRC_DIR", "src")

# Core directories
MODELS_DIR = get_absolute_path("MODELS_DIR", "src/models")
EMBEDDING_DIR = get_absolute_path("EMBEDDING_DIR", "src/embedding")

# ETM_DIR alias for backward compatibility
ETM_DIR = MODELS_DIR

# Data directories
DATA_DIR = get_absolute_path("DATA_DIR", "data")

# =============================================================================
# Unified Path Structure (NO default_user redundancy)
# =============================================================================
# New unified structure: result/{model_size}/{dataset_name}/
# This removes the unnecessary default_user layer for cleaner organization

# Base workspace directory (for shared matrices: BOW, embeddings, covariates)
BASE_WORKSPACE = get_absolute_path("WORKSPACE_DIR", "data/workspace")
WORKSPACE_DIR = BASE_WORKSPACE  # Alias for backward compatibility

# Base result directory (for model-specific outputs: theta, beta, visualizations)
BASE_RESULT = get_absolute_path("RESULT_DIR", "result")
RESULT_DIR = BASE_RESULT  # Alias for backward compatibility

# Logs directory
LOGS_DIR = BASE_RESULT / "logs"


def get_workspace_path(dataset_name: str = "", model_size: str = "0.6B") -> Path:
    """
    Get workspace path for shared matrices.
    
    Unified Structure: workspace/{model_size}/{dataset_name}/
    (Removed default_user redundancy)
    
    Contains:
        - bow_matrix.npy / bow_matrix.npz
        - vocab.json
        - word2vec_embeddings.npy
        - sbert_embeddings.npy
        - time_slices.json
        - time_indices.npy
        - covariates.npy
        - covariate_names.json
        - config.json
    """
    path = BASE_WORKSPACE / model_size
    if dataset_name:
        path = path / dataset_name
    return path


def get_result_path(
    dataset_name: str = "",
    model_size: str = "0.6B",
    model_name: str = "theta",
    task_name: str = "",
    lang: str = ""
) -> Path:
    """
    Get result path for model-specific outputs.
    
    New Structure: result/{dataset}/{model_size}/theta/exp_{timestamp}/
    
    Args:
        dataset_name: Dataset name
        model_size: Model size (0.6B, 4B, 8B)
        model_name: Model name (theta, dtm, stm, lda, ctm, etc.)
        task_name: Task/experiment name (default: exp_YYYYMMDD_HHMMSS)
        lang: Language for visualization output ('zh' or 'en')
    
    Directory Structure:
        result/{dataset}/{model_size}/theta/exp_{timestamp}/
        ├── config.json         # Experiment config
        ├── data/               # Preprocessed data
        │   ├── bow/
        │   └── embeddings/
        ├── theta/              # Model outputs
        │   ├── theta.npy
        │   ├── beta.npy
        │   ├── topic_words.json
        │   └── etm_model.pt
        ├── metrics.json        # Evaluation metrics
        └── {lang}/             # Visualization (zh or en)
            ├── global/
            └── topic/
    """
    path = BASE_RESULT
    if dataset_name:
        path = path / dataset_name
    if model_size:
        path = path / model_size
    if model_name:
        path = path / model_name
    if task_name:
        path = path / task_name
    if lang:
        path = path / lang
    return path


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path

# Model directories - using get_absolute_path for robust resolution
EMBEDDING_MODELS_DIR = get_absolute_path("EMBEDDING_MODELS_DIR", "embedding_models")


def _get_qwen_model_path(model_size: str, env_var: str, default_name: str) -> Path:
    """Get Qwen model path with priority: ENV > default."""
    env_value = os.environ.get(env_var)
    if env_value:
        path = Path(env_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return EMBEDDING_MODELS_DIR / default_name


# Embedding model paths and dimension mapping
# Priority: QWEN_MODEL_0_6B > QWEN_MODEL_PATH > default
QWEN_MODEL_PATHS = {
    '0.6B': str(_get_qwen_model_path('0.6B', 'QWEN_MODEL_0_6B', 'qwen3_embedding_0.6B')),
    '4B': str(_get_qwen_model_path('4B', 'QWEN_MODEL_4B', 'qwen3_embedding_4B')),
    '8B': str(_get_qwen_model_path('8B', 'QWEN_MODEL_8B', 'qwen3_embedding_8B')),
}

# Legacy compatibility: QWEN_MODEL_PATH now derives from QWEN_MODEL_PATHS['0.6B']
QWEN_MODEL_PATH = Path(QWEN_MODEL_PATHS['0.6B'])

EMBEDDING_DIMS = {
    '0.6B': 1024,
    '4B': 2560,
    '8B': 4096,
}


def get_qwen_model_path(model_size: str) -> str:
    """Get Qwen model path based on model_size"""
    if model_size not in QWEN_MODEL_PATHS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(QWEN_MODEL_PATHS.keys())}")
    return QWEN_MODEL_PATHS[model_size]


def get_embedding_dim(model_size: str) -> int:
    """Get embedding dimension based on model_size"""
    return EMBEDDING_DIMS.get(model_size, 1024)

# Dataset-specific configurations
# vocab_size should be proportional to dataset size but not too large
# Rule of thumb: vocab_size ~ min(sqrt(n_docs) * 50, 10000)
DATASET_CONFIGS = {
    "socialTwitter": {
        "vocab_size": 5000,      # ~40K docs
        "num_topics": 20,
        "min_doc_freq": 5,
        "language": "multi",     # Spanish + some English
    },
    "hatespeech": {
        "vocab_size": 8000,      # ~437K docs
        "num_topics": 20,
        "min_doc_freq": 10,
        "language": "english",
    },
    "mental_health": {
        "vocab_size": 10000,     # ~1M docs
        "num_topics": 30,
        "min_doc_freq": 20,
        "language": "english",
    },
    "FCPB": {
        "vocab_size": 8000,      # ~854K docs
        "num_topics": 25,
        "min_doc_freq": 15,
        "language": "english",
    },
    "germanCoal": {
        "vocab_size": 3000,      # ~9K docs (smaller dataset)
        "num_topics": 15,
        "min_doc_freq": 3,
        "language": "german",
    },
    "edu_data": {
        "vocab_size": 5000,      # ~857 docs (education policy documents)
        "num_topics": 20,
        "min_doc_freq": 3,
        "language": "chinese",
        "has_timestamp": True,   # DTM specific
    },
    "edu_data_enhanced": {
        "vocab_size": 5000,      # ~734 docs with province/year metadata
        "num_topics": 20,
        "min_doc_freq": 3,
        "language": "chinese",
        "has_timestamp": True,   # DTM specific - year column
        "timestamp_column": "year",
        "covariate_columns": ["province_id"],  # STM specific - province as covariate
    },
    "test_data": {
        "vocab_size": 3000,      # ~858 docs (test dataset)
        "num_topics": 10,
        "min_doc_freq": 2,
        "language": "chinese",
    },
}


@dataclass
class DataConfig:
    """Data configuration"""
    dataset: str = "socialTwitter"
    data_dir: str = str(DATA_DIR)
    text_column: str = "clean_text"
    label_column: str = "label"
    timestamp_column: Optional[str] = None
    
    @property
    def raw_data_path(self) -> str:
        """Get the raw data path, handling different naming conventions"""
        dataset_dir = os.path.join(self.data_dir, self.dataset)
        # Try different file naming patterns
        patterns = [
            f"{self.dataset}_text_only.csv",
            f"{self.dataset}_cleaned.csv",
            "complaints_text_only.csv",      # FCPB
            "german_coal_text_only.csv",     # germanCoal
        ]
        for pattern in patterns:
            path = os.path.join(dataset_dir, pattern)
            if os.path.exists(path):
                return path
        # Default fallback
        return os.path.join(dataset_dir, f"{self.dataset}_text_only.csv")
    
    @property
    def cleaned_data_path(self) -> str:
        return os.path.join(self.data_dir, self.dataset, f"{self.dataset}_cleaned.csv")


@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    mode: str = "zero_shot"  # zero_shot, supervised, unsupervised
    embedding_dim: int = 1024
    model_path: str = str(QWEN_MODEL_PATH)
    output_dir: str = str(EMBEDDING_DIR / "outputs")
    batch_size: int = 64
    max_length: int = 512
    
    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.output_dir, self.mode, f"{{dataset}}_{self.mode}_embeddings.npy")
    
    @property
    def labels_path(self) -> str:
        return os.path.join(self.output_dir, self.mode, f"{{dataset}}_{self.mode}_labels.npy")


@dataclass
class BOWConfig:
    """BOW generation configuration"""
    vocab_size: int = 8000
    min_doc_freq: int = 10
    max_doc_freq_ratio: float = 0.5
    use_tfidf: bool = False
    language: str = "english"  # english, chinese, german
    
    # Tokenization
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    remove_numbers: bool = True
    lowercase: bool = True
    min_word_length: int = 3  # Minimum word length to filter short noise


@dataclass
class ModelConfig:
    """ETM model configuration"""
    # Architecture
    num_topics: int = 20
    hidden_dim: int = 1024
    doc_embedding_dim: int = 1024
    word_embedding_dim: int = 1024
    encoder_dropout: float = 0.2
    encoder_activation: str = "relu"
    train_word_embeddings: bool = True  # Default: train word embeddings from scratch
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.002
    weight_decay: float = 1e-4
    
    # KL Annealing - gradual warmup for stable training
    kl_start: float = 0.0
    kl_end: float = 1.0
    kl_warmup_epochs: int = 30
    
    # Early stopping - need enough epochs for KL warmup
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 0.001
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    
    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    top_k_coherence: int = 10
    top_k_diversity: int = 25
    compute_stability: bool = False
    stability_runs: int = 5
    
    # pyLDAvis
    use_pyldavis: bool = True
    
    # External coherence (using gensim)
    use_external_coherence: bool = True
    coherence_measures: List[str] = field(default_factory=lambda: ["c_npmi", "c_v", "u_mass"])


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    output_dir: str = ""
    figsize: tuple = (12, 8)
    dpi: int = 150
    language: str = "en"  # en, zh
    
    # Word cloud
    use_wordcloud: bool = True
    wordcloud_max_words: int = 50
    
    # Topic visualization
    num_topics_to_show: int = 20
    num_words_per_topic: int = 10
    
    # Temporal analysis
    enable_temporal: bool = False
    time_bins: int = 10


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration with three-level path decoupling.
    
    Path Structure:
        - Shared matrices (workspace): workspace/{user_id}/{dataset_name}/
        - Model outputs (result):      result/{user_id}/{dataset_name}/{model_name}/{timestamp}/
    """
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bow: BOWConfig = field(default_factory=BOWConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    device: str = "cuda"
    gpu_id: int = 1  # Use GPU 1, avoid GPU 0
    seed: int = 42
    dev_mode: bool = False
    
    # ==========================================================================
    # Three-level path decoupling: user_id -> dataset_name -> model_name
    # ==========================================================================
    user_id: str = "default_user"
    model_name: str = ""  # e.g., "lda", "hdp", "theta", "ctm"
    
    # Model size for THETA (e.g., '0.6B', '4B', '8B')
    model_size: str = "0.6B"
    
    # Experiment timestamp (auto-generated if empty)
    timestamp: str = ""
    
    # Legacy experiment management (for backward compatibility)
    data_exp: str = ""
    train_exp: str = ""
    
    # Force overwrite existing matrices
    force: bool = False
    
    # ==========================================================================
    # Workspace paths (shared matrices)
    # ==========================================================================
    
    @property
    def workspace_dir(self) -> str:
        """
        Workspace directory for shared matrices.
        Structure: workspace/{model_size}/{dataset_name}/
        """
        return str(get_workspace_path(self.data.dataset, self.model_size))
    
    @property
    def bow_matrix_path(self) -> str:
        """Path to BOW matrix (sparse npz or dense npy)"""
        # Prefer sparse format
        npz_path = os.path.join(self.workspace_dir, "bow_matrix.npz")
        if os.path.exists(npz_path):
            return npz_path
        return os.path.join(self.workspace_dir, "bow_matrix.npy")
    
    @property
    def vocab_path(self) -> str:
        """Path to vocabulary JSON"""
        return os.path.join(self.workspace_dir, "vocab.json")
    
    @property
    def word2vec_path(self) -> str:
        """Path to Word2Vec embeddings"""
        return os.path.join(self.workspace_dir, "word2vec_embeddings.npy")
    
    @property
    def sbert_path(self) -> str:
        """Path to SBERT embeddings"""
        return os.path.join(self.workspace_dir, "sbert_embeddings.npy")
    
    @property
    def time_slices_path(self) -> str:
        """Path to time slices JSON"""
        return os.path.join(self.workspace_dir, "time_slices.json")
    
    @property
    def time_indices_path(self) -> str:
        """Path to time indices"""
        return os.path.join(self.workspace_dir, "time_indices.npy")
    
    @property
    def covariates_path(self) -> str:
        """Path to covariates matrix"""
        return os.path.join(self.workspace_dir, "covariates.npy")
    
    @property
    def covariate_names_path(self) -> str:
        """Path to covariate names JSON"""
        return os.path.join(self.workspace_dir, "covariate_names.json")
    
    @property
    def workspace_config_path(self) -> str:
        """Path to workspace config JSON"""
        return os.path.join(self.workspace_dir, "config.json")
    
    # ==========================================================================
    # Result paths (model-specific outputs)
    # ==========================================================================
    
    @property
    def result_base_dir(self) -> str:
        """
        Result directory for model outputs.
        New structure: result/{dataset}/{model_size}/theta/exp_{timestamp}/
        """
        ts = self.timestamp or self.train_exp or ""
        return str(get_result_path(self.data.dataset, self.model_size, "theta", ts))
    
    @property
    def model_dir(self) -> str:
        """Directory for trained model and matrices (theta, beta): exp_*/theta/"""
        return os.path.join(self.result_base_dir, "theta")
    
    @property
    def evaluation_dir(self) -> str:
        """Directory for evaluation results (deprecated, use exp_dir/metrics.json)"""
        return self.result_base_dir
    
    @property
    def visualization_dir(self) -> str:
        """Directory for visualizations: exp_*/{lang}/"""
        lang = self.visualization.language if hasattr(self, 'visualization') else 'zh'
        return os.path.join(self.result_base_dir, lang)
    
    @property
    def log_dir(self) -> str:
        """Directory for training logs"""
        return str(LOGS_DIR)
    
    # ==========================================================================
    # Legacy properties (for backward compatibility)
    # ==========================================================================
    
    output_base_dir: str = str(RESULT_DIR)
    
    @property
    def dataset_base_dir(self) -> str:
        """Base directory for this dataset
        
        New structure: result/{dataset}/{model_size}/theta/
        """
        if self.model_size:
            return os.path.join(self.output_base_dir, self.data.dataset, self.model_size, "theta")
        return os.path.join(self.output_base_dir, self.data.dataset, "theta")
    
    @property
    def exp_dir(self) -> str:
        """Experiment directory: result/{dataset}/{model_size}/theta/exp_{timestamp}/"""
        if self.train_exp:
            return os.path.join(self.dataset_base_dir, self.train_exp)
        return self.dataset_base_dir
    
    @property
    def data_exp_dir(self) -> str:
        """Data directory within experiment: exp_*/data/"""
        return os.path.join(self.exp_dir, "data")
    
    @property
    def result_dir(self) -> str:
        """Model output directory: exp_*/theta/"""
        return os.path.join(self.exp_dir, "theta")
    
    @property
    def embeddings_dir(self) -> str:
        """Directory for document embeddings: exp_*/data/embeddings/"""
        return os.path.join(self.data_exp_dir, "embeddings")
    
    @property
    def bow_dir(self) -> str:
        """Directory for BOW matrices and vocabulary: exp_*/data/bow/"""
        return os.path.join(self.data_exp_dir, "bow")
    
    # Legacy properties for backward compatibility
    @property
    def output_dir(self) -> str:
        return self.model_dir
    
    @property
    def analysis_dir(self) -> str:
        return self.visualization_dir
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "data": asdict(self.data),
            "embedding": asdict(self.embedding),
            "bow": asdict(self.bow),
            "model": asdict(self.model),
            "evaluation": asdict(self.evaluation),
            "visualization": asdict(self.visualization),
            "device": self.device,
            "gpu_id": self.gpu_id,
            "seed": self.seed,
            "dev_mode": self.dev_mode,
            "output_base_dir": self.output_base_dir
        }
    
    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "PipelineConfig":
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        config.data = DataConfig(**data.get("data", {}))
        config.embedding = EmbeddingConfig(**data.get("embedding", {}))
        config.bow = BOWConfig(**data.get("bow", {}))
        config.model = ModelConfig(**data.get("model", {}))
        config.evaluation = EvaluationConfig(**data.get("evaluation", {}))
        config.visualization = VisualizationConfig(**data.get("visualization", {}))
        config.device = data.get("device", "cuda")
        config.gpu_id = data.get("gpu_id", 1)
        config.seed = data.get("seed", 42)
        config.dev_mode = data.get("dev_mode", False)
        config.output_base_dir = data.get("output_base_dir", str(ETM_DIR / "outputs"))
        
        return config


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for ETM pipeline"""
    parser = argparse.ArgumentParser(
        description="ETM Topic Model Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train ETM model")
    _add_common_args(train_parser)
    _add_training_args(train_parser)
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--timestamp", type=str, default=None, help="Model timestamp to evaluate")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    _add_common_args(viz_parser)
    viz_parser.add_argument("--timestamp", type=str, default=None, help="Model timestamp to visualize")
    viz_parser.add_argument("--no_wordcloud", action="store_true", help="Disable word clouds")
    
    # Pipeline command (full pipeline)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    _add_common_args(pipeline_parser)
    _add_training_args(pipeline_parser)
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean data")
    clean_parser.add_argument("--input", type=str, required=True, help="Input file/directory")
    clean_parser.add_argument("--output", type=str, required=True, help="Output file/directory")
    clean_parser.add_argument("--language", type=str, default="english", choices=["english", "chinese", "german"])
    
    return parser


def _add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments"""
    parser.add_argument("--dataset", type=str, default="socialTwitter",
                        help="Dataset name")
    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "supervised", "unsupervised"],
                        help="Embedding mode")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--dev", action="store_true",
                        help="Enable development mode with extra logging")


def _add_training_args(parser: argparse.ArgumentParser):
    """Add training arguments"""
    # Model architecture
    parser.add_argument("--num_topics", type=int, default=20,
                        help="Number of topics (5-100)")
    parser.add_argument("--vocab_size", type=int, default=5000,
                        help="Vocabulary size (1000-20000)")
    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension (256-1024)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="Learning rate")
    
    # KL annealing
    parser.add_argument("--kl_start", type=float, default=0.0,
                        help="KL weight start value")
    parser.add_argument("--kl_end", type=float, default=1.0,
                        help="KL weight end value")
    parser.add_argument("--kl_warmup", type=int, default=50,
                        help="KL warmup epochs")
    
    # Early stopping
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # Word embeddings
    parser.add_argument("--train_word_embeddings", action="store_true", default=True,
                        help="Train word embeddings from scratch (default: True)")
    parser.add_argument("--no_train_word_embeddings", action="store_true",
                        help="Use pretrained Qwen word embeddings (frozen)")
    
    # Temporal analysis
    parser.add_argument("--enable_temporal", action="store_true",
                        help="Enable temporal analysis (requires timestamp column)")
    parser.add_argument("--timestamp_column", type=str, default=None,
                        help="Column name for timestamps in data")
    
    # Model size
    parser.add_argument("--model_size", type=str, default="0.6B",
                        choices=["0.6B", "4B", "8B"],
                        help="Qwen model size (default: 0.6B)")
    
    # Pipeline control
    parser.add_argument("--skip_viz", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation step")
    
    # Experiment management
    parser.add_argument("--data_exp", type=str, default="",
                        help="Data experiment ID (for THETA exp management)")
    parser.add_argument("--train_exp", type=str, default="",
                        help="Training experiment ID (for THETA exp management)")
    parser.add_argument("--output_base_dir", type=str, default="",
                        help="Override base output directory (e.g. result/default_user)")
    
    # Visualization language
    parser.add_argument("--language", type=str, default="en",
                        choices=["en", "zh"],
                        help="Visualization language (chart titles, labels)")
    
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Disable pin_memory for DataLoader")
    parser.add_argument("--no_persistent_workers", action="store_true",
                        help="Disable persistent workers")


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    """
    Create config from parsed arguments with multi-level priority:
    P1: CLI args (highest)
    P2: Environment variables
    P3: Dataset-specific defaults (DATASET_CONFIGS)
    P4: Heuristic defaults (HEURISTIC_DEFAULTS)
    """
    # Load base config if provided
    if hasattr(args, 'config') and args.config:
        config = PipelineConfig.load(args.config)
    else:
        config = PipelineConfig()
    
    # Get dataset-specific defaults (P3)
    ds_defaults = {}
    if hasattr(args, "dataset") and args.dataset:
        config.data.dataset = args.dataset
        if args.dataset in DATASET_CONFIGS:
            ds_defaults = DATASET_CONFIGS[args.dataset]
    
    # Helper to resolve with full priority chain
    def _resolve(param_name: str, cli_value: Any, ds_key: str = None, param_type: type = None) -> Any:
        """Resolve param: CLI > ENV > dataset_config > heuristic"""
        if cli_value is not None:
            return cli_value
        
        # Check environment variable
        env_var = PARAM_ENV_MAPPING.get(param_name, param_name.upper())
        env_value = os.environ.get(env_var)
        if env_value is not None and env_value != "":
            try:
                if param_type == int:
                    return int(env_value)
                elif param_type == float:
                    return float(env_value)
                elif param_type == bool:
                    return env_value.lower() in ('true', '1', 'yes')
                return env_value
            except (ValueError, TypeError):
                pass
        
        # Check dataset-specific defaults
        key = ds_key or param_name
        if key in ds_defaults:
            return ds_defaults[key]
        
        # Fall back to heuristic defaults
        return HEURISTIC_DEFAULTS.get(param_name)
    
    # Apply mode
    if hasattr(args, "mode"):
        config.embedding.mode = _resolve("mode", args.mode, param_type=str) or "zero_shot"
    if hasattr(args, "dev"):
        config.dev_mode = args.dev
    
    # Training args with full priority resolution
    config.model.num_topics = _resolve("num_topics", getattr(args, "num_topics", None), param_type=int) or 20
    config.bow.vocab_size = _resolve("vocab_size", getattr(args, "vocab_size", None), param_type=int) or 5000
    config.model.hidden_dim = _resolve("hidden_dim", getattr(args, "hidden_dim", None), param_type=int) or 1024
    config.model.epochs = _resolve("epochs", getattr(args, "epochs", None), param_type=int) or 100
    config.model.batch_size = _resolve("batch_size", getattr(args, "batch_size", None), param_type=int) or 64
    config.model.learning_rate = _resolve("learning_rate", getattr(args, "learning_rate", None), param_type=float) or 0.002
    config.model.kl_start = _resolve("kl_start", getattr(args, "kl_start", None), param_type=float) or 0.0
    config.model.kl_end = _resolve("kl_end", getattr(args, "kl_end", None), param_type=float) or 1.0
    config.model.kl_warmup_epochs = _resolve("kl_warmup", getattr(args, "kl_warmup", None), param_type=int) or 50
    config.model.patience = _resolve("patience", getattr(args, "patience", None), param_type=int) or 10
    config.bow.min_doc_freq = _resolve("min_df", getattr(args, "min_df", None), "min_doc_freq", param_type=int) or 2
    
    # Language from dataset config or default
    if "language" in ds_defaults:
        config.bow.language = ds_defaults["language"]
    
    if hasattr(args, "no_early_stopping"):
        config.model.early_stopping = not args.no_early_stopping
    
    # Model size with priority
    model_size = _resolve("model_size", getattr(args, "model_size", None), param_type=str)
    if model_size:
        config.model_size = model_size
    
    # Word embeddings - default is True (train from scratch)
    if hasattr(args, "no_train_word_embeddings") and args.no_train_word_embeddings:
        config.model.train_word_embeddings = False
    else:
        config.model.train_word_embeddings = True
    
    # Temporal analysis
    if hasattr(args, "enable_temporal"):
        config.visualization.enable_temporal = args.enable_temporal
    if hasattr(args, "timestamp_column") and args.timestamp_column:
        config.data.timestamp_column = args.timestamp_column
    
    if hasattr(args, "num_workers"):
        config.model.num_workers = args.num_workers
    if hasattr(args, "no_pin_memory") and args.no_pin_memory:
        config.model.pin_memory = False
    if hasattr(args, "no_persistent_workers") and args.no_persistent_workers:
        config.model.persistent_workers = False
    
    # Visualization language with priority
    viz_lang = _resolve("language", getattr(args, "language", None), param_type=str)
    if viz_lang:
        config.visualization.language = viz_lang
    
    # Experiment management
    if hasattr(args, "data_exp") and args.data_exp:
        config.data_exp = args.data_exp
    if hasattr(args, "train_exp") and args.train_exp:
        config.train_exp = args.train_exp
    if hasattr(args, "output_base_dir") and args.output_base_dir:
        config.output_base_dir = args.output_base_dir
    
    return config


# ============================================================================
# Parameter Constraints - Parameter constraint definitions for validating frontend inputs
# ============================================================================

PARAM_CONSTRAINTS = {
    "num_topics": {
        "type": "int",
        "min": 5,
        "max": 100,
        "default": 20,
        "options": [5, 10, 15, 20, 25, 30, 40, 50, 75, 100],
        "description": "Number of topics, recommended to choose based on dataset size"
    },
    "vocab_size": {
        "type": "int",
        "min": 1000,
        "max": 20000,
        "default": 5000,
        "options": [1000, 2000, 3000, 5000, 8000, 10000, 15000, 20000],
        "description": "Vocabulary size, recommended: sqrt(num_docs) * 50"
    },
    "hidden_dim": {
        "type": "int",
        "min": 128,
        "max": 1024,
        "default": 512,
        "options": [256, 512, 768, 1024],
        "description": "Encoder hidden layer dimension"
    },
    "epochs": {
        "type": "int",
        "min": 10,
        "max": 500,
        "default": 50,
        "options": [20, 30, 50, 100, 150, 200],
        "description": "Number of training epochs"
    },
    "batch_size": {
        "type": "int",
        "min": 8,
        "max": 512,
        "default": 64,
        "options": [16, 32, 64, 128, 256],
        "description": "Batch size"
    },
    "learning_rate": {
        "type": "float",
        "min": 0.00001,
        "max": 0.1,
        "default": 0.002,
        "options": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
        "description": "Learning rate"
    },
}


def validate_params(params: dict) -> tuple:
    """
    Validate if parameters are valid
    
    Args:
        params: Parameter dictionary, e.g., {"num_topics": 20, "vocab_size": 5000}
        
    Returns:
        (is_valid: bool, error_message: str)
        
    Example:
        is_valid, msg = validate_params({"num_topics": 200})
        # is_valid = False, msg = "num_topics=200 exceeds maximum 100"
    """
    for param_name, value in params.items():
        if param_name not in PARAM_CONSTRAINTS:
            continue
        
        constraints = PARAM_CONSTRAINTS[param_name]
        
        # Type check
        expected_type = constraints.get("type", "int")
        if expected_type == "int" and not isinstance(value, int):
            return False, f"{param_name} must be an integer, got {type(value).__name__}"
        if expected_type == "float" and not isinstance(value, (int, float)):
            return False, f"{param_name} must be a number, got {type(value).__name__}"
        
        # Range check
        if "min" in constraints and value < constraints["min"]:
            return False, f"{param_name}={value} is below minimum {constraints['min']}"
        if "max" in constraints and value > constraints["max"]:
            return False, f"{param_name}={value} exceeds maximum {constraints['max']}"
    
    return True, "OK"


def get_param_options() -> dict:
    """
    Get all parameter options - for frontend dropdown menus
    
    Returns:
        {
            "num_topics": {"options": [5, 10, ...], "default": 20, "description": "..."},
            ...
        }
    """
    return PARAM_CONSTRAINTS


# ============================================================================
# Predefined configurations for common use cases
# ============================================================================

PRESET_CONFIGS = {
    "small": {
        "num_topics": 10,
        "vocab_size": 2000,
        "epochs": 30,
        "hidden_dim": 256
    },
    "medium": {
        "num_topics": 20,
        "vocab_size": 5000,
        "epochs": 50,
        "hidden_dim": 1024
    },
    "large": {
        "num_topics": 50,
        "vocab_size": 10000,
        "epochs": 100,
        "hidden_dim": 1024
    }
}


def get_preset_config(preset: str) -> PipelineConfig:
    """Get a preset configuration"""
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESET_CONFIGS.keys())}")
    
    config = PipelineConfig()
    preset_values = PRESET_CONFIGS[preset]
    
    config.model.num_topics = preset_values["num_topics"]
    config.bow.vocab_size = preset_values["vocab_size"]
    config.model.epochs = preset_values["epochs"]
    config.model.hidden_dim = preset_values["hidden_dim"]
    
    return config


if __name__ == "__main__":
    # Test configuration
    config = PipelineConfig()
    config.data.dataset = "socialTwitter"
    config.embedding.mode = "zero_shot"
    
    print("Default configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save and load test
    config.save("/tmp/test_config.json")
    loaded_config = PipelineConfig.load("/tmp/test_config.json")
    print("\nLoaded configuration matches:", config.to_dict() == loaded_config.to_dict())
