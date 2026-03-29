"""
THETA Configuration Loader
===========================

Implements three-tier configuration priority:
1. CLI arguments (highest priority) - User command line inputs
2. YAML config (second priority) - config/default.yaml
3. .env file (lowest priority) - Physical paths only

Usage:
    from config_loader import ConfigLoader
    
    # Load config with CLI args override
    config = ConfigLoader.load(model_name='theta', cli_args=args)
    
    # Get specific model config
    theta_config = ConfigLoader.get_model_config('theta')
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dotenv import load_dotenv

# =============================================================================
# Path Resolution
# =============================================================================

def get_project_root() -> Path:
    """Get project root directory."""
    # Try to find project root by looking for .env or config directory
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Max 5 levels up
        if (current / '.env').exists() or (current / 'config').exists():
            return current
        current = current.parent
    # Fallback to parent of src/models
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / 'config'
DEFAULT_YAML = CONFIG_DIR / 'default.yaml'
ENV_FILE = PROJECT_ROOT / '.env'


# =============================================================================
# Environment Validation
# =============================================================================

class EnvironmentError(Exception):
    """Raised when required environment configuration is missing or invalid."""
    pass


def validate_path(path: str, name: str, must_exist: bool = True) -> Path:
    """
    Validate a path from environment variable.
    
    Args:
        path: Path string to validate
        name: Name of the environment variable (for error messages)
        must_exist: If True, raise error if path doesn't exist
        
    Returns:
        Resolved Path object
        
    Raises:
        EnvironmentError: If path is invalid or doesn't exist
    """
    if not path:
        raise EnvironmentError(
            f"Environment variable '{name}' is not set.\n"
            f"Please check your .env file at: {ENV_FILE}\n"
            f"You can copy from .env.example: cp .env.example .env"
        )
    
    resolved = Path(path).resolve()
    
    if must_exist and not resolved.exists():
        raise EnvironmentError(
            f"Path specified by '{name}' does not exist: {resolved}\n"
            f"Please check your .env file and ensure the path is correct.\n"
            f"Current value: {path}"
        )
    
    return resolved


# =============================================================================
# YAML Configuration Loader
# =============================================================================

class YAMLConfig:
    """Loads and caches YAML configuration."""
    
    _cache: Optional[Dict[str, Any]] = None
    
    @classmethod
    def load(cls, yaml_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            yaml_path: Path to YAML file (default: config/default.yaml)
            
        Returns:
            Dictionary with configuration values
            
        Raises:
            EnvironmentError: If YAML file is missing or invalid
        """
        if cls._cache is not None and yaml_path is None:
            return cls._cache
        
        yaml_path = yaml_path or DEFAULT_YAML
        
        if not yaml_path.exists():
            raise EnvironmentError(
                f"Configuration file not found: {yaml_path}\n"
                f"Please ensure config/default.yaml exists in your project root.\n"
                f"Project root: {PROJECT_ROOT}"
            )
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise EnvironmentError(
                f"Invalid YAML syntax in {yaml_path}:\n{e}"
            )
        
        if yaml_path == DEFAULT_YAML:
            cls._cache = config
        
        return config
    
    @classmethod
    def get_model_defaults(cls, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model.
        
        Args:
            model_name: Name of the model (theta, lda, etm, etc.)
            
        Returns:
            Dictionary with model-specific defaults
        """
        config = cls.load()
        model_name = model_name.lower()
        
        # Get model-specific config
        model_config = config.get(model_name, {})
        
        # Merge with global settings
        global_config = config.get('global', {})
        
        # Model config overrides global
        merged = {**global_config, **model_config}
        
        return merged
    
    @classmethod
    def get_visualization_defaults(cls) -> Dict[str, Any]:
        """Get default visualization settings."""
        config = cls.load()
        return config.get('visualization', {})
    
    @classmethod
    def clear_cache(cls):
        """Clear the configuration cache."""
        cls._cache = None


# =============================================================================
# Environment Configuration Loader
# =============================================================================

class EnvConfig:
    """Loads environment configuration from .env file."""
    
    _loaded: bool = False
    
    @classmethod
    def load(cls, env_path: Optional[Path] = None):
        """
        Load environment variables from .env file.
        
        Args:
            env_path: Path to .env file (default: project_root/.env)
        """
        if cls._loaded and env_path is None:
            return
        
        env_path = env_path or ENV_FILE
        
        if env_path.exists():
            load_dotenv(env_path, override=True)
            cls._loaded = True
        else:
            # Warning but don't fail - some paths may have defaults
            print(f"Warning: .env file not found at {env_path}", file=sys.stderr)
    
    @classmethod
    def get_path(cls, name: str, default: Optional[str] = None, 
                 must_exist: bool = False) -> Optional[Path]:
        """
        Get a path from environment variable.
        
        Args:
            name: Environment variable name
            default: Default value if not set
            must_exist: If True, validate path exists
            
        Returns:
            Path object or None
        """
        cls.load()
        value = os.environ.get(name, default)
        
        if value is None:
            return None
        
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        
        if must_exist:
            return validate_path(str(path), name, must_exist=True)
        
        return path.resolve()
    
    @classmethod
    def get(cls, name: str, default: Optional[str] = None) -> Optional[str]:
        """Get an environment variable value."""
        cls.load()
        return os.environ.get(name, default)
    
    @classmethod
    def get_qwen_model_path(cls, model_size: str = "0.6B") -> Path:
        """
        Get Qwen model path for specified size.
        
        Args:
            model_size: Model size (0.6B, 4B, 8B)
            
        Returns:
            Path to Qwen model directory
        """
        # Try size-specific env var first
        size_var = f"QWEN_MODEL_{model_size.replace('.', '_').upper()}"
        path = cls.get_path(size_var)
        
        if path and path.exists():
            return path
        
        # Fallback to generic QWEN_MODEL_PATH
        path = cls.get_path('QWEN_MODEL_PATH')
        if path and path.exists():
            return path
        
        # Fallback to default location
        default_path = PROJECT_ROOT / 'embedding_models' / f'qwen3_embedding_{model_size}'
        if default_path.exists():
            return default_path
        
        raise EnvironmentError(
            f"Qwen model not found for size {model_size}.\n"
            f"Please set one of the following in your .env file:\n"
            f"  - {size_var}=/path/to/qwen_model\n"
            f"  - QWEN_MODEL_PATH=/path/to/qwen_model\n"
            f"Or place the model at: {default_path}"
        )
    
    @classmethod
    def get_sbert_model_path(cls) -> Path:
        """
        Get SBERT model path.
        
        Returns:
            Path to SBERT model directory
        """
        path = cls.get_path('SBERT_MODEL_PATH')
        
        if path and path.exists():
            return path
        
        # Try common default locations
        default_paths = [
            PROJECT_ROOT / 'models' / 'sbert' / 'sentence-transformers' / 'all-MiniLM-L6-v2',
            PROJECT_ROOT / 'embedding_models' / 'sbert' / 'all-MiniLM-L6-v2',
        ]
        
        for default_path in default_paths:
            if default_path.exists():
                return default_path
        
        raise EnvironmentError(
            "SBERT model not found.\n"
            "Please set SBERT_MODEL_PATH in your .env file:\n"
            "  SBERT_MODEL_PATH=/path/to/sbert_model\n"
            f"Or place the model at one of:\n"
            + "\n".join(f"  - {p}" for p in default_paths)
        )
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get data directory path."""
        path = cls.get_path('DATA_DIR')
        if path:
            return path
        return PROJECT_ROOT / 'data'
    
    @classmethod
    def get_result_dir(cls) -> Path:
        """Get result directory path."""
        path = cls.get_path('RESULT_DIR')
        if path:
            return path
        return PROJECT_ROOT / 'result'
    
    @classmethod
    def get_workspace_dir(cls) -> Path:
        """Get workspace directory path."""
        path = cls.get_path('WORKSPACE_DIR')
        if path:
            return path
        return PROJECT_ROOT / 'data' / 'workspace'


# =============================================================================
# Main Configuration Loader
# =============================================================================

class ConfigLoader:
    """
    Main configuration loader implementing three-tier priority:
    1. CLI arguments (highest)
    2. YAML config (second)
    3. .env paths (lowest, for physical paths only)
    """
    
    @classmethod
    def load(cls, model_name: str, cli_args: Optional[Dict[str, Any]] = None,
             yaml_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load configuration with proper priority.
        
        Args:
            model_name: Name of the model (theta, lda, etc.)
            cli_args: Command line arguments (highest priority)
            yaml_path: Optional custom YAML config path
            
        Returns:
            Merged configuration dictionary
        """
        # Load environment first (for paths)
        EnvConfig.load()
        
        # Get YAML defaults
        yaml_config = YAMLConfig.get_model_defaults(model_name)
        
        # Get visualization defaults
        viz_config = YAMLConfig.get_visualization_defaults()
        
        # Start with YAML config
        config = {**yaml_config}
        
        # Add visualization settings
        config['visualization'] = viz_config
        
        # Override with CLI args (filter out None values)
        if cli_args:
            for key, value in cli_args.items():
                if value is not None:
                    config[key] = value
        
        # Add path configurations from .env
        config['paths'] = cls._get_paths(model_name, config.get('model_size', '0.6B'))
        
        return config
    
    @classmethod
    def _get_paths(cls, model_name: str, model_size: str = "0.6B") -> Dict[str, Path]:
        """Get all required paths from environment."""
        paths = {
            'project_root': PROJECT_ROOT,
            'config_dir': CONFIG_DIR,
            'data_dir': EnvConfig.get_data_dir(),
            'result_dir': EnvConfig.get_result_dir(),
            'workspace_dir': EnvConfig.get_workspace_dir(),
        }
        
        # Model-specific paths
        if model_name.lower() == 'theta':
            try:
                paths['qwen_model'] = EnvConfig.get_qwen_model_path(model_size)
            except EnvironmentError:
                pass  # Will be caught later if needed
        
        if model_name.lower() in ('ctm', 'bertopic'):
            try:
                paths['sbert_model'] = EnvConfig.get_sbert_model_path()
            except EnvironmentError:
                pass  # Will be caught later if needed
        
        return paths
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model (YAML defaults only).
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary
        """
        return YAMLConfig.get_model_defaults(model_name)
    
    @classmethod
    def get_language(cls, cli_lang: Optional[str] = None) -> str:
        """
        Get visualization language with proper priority.
        
        Args:
            cli_lang: Language from CLI (highest priority)
            
        Returns:
            Language code ('zh' or 'en')
        """
        # CLI has highest priority
        if cli_lang:
            return cli_lang
        
        # Check YAML config
        viz_config = YAMLConfig.get_visualization_defaults()
        yaml_lang = viz_config.get('language')
        if yaml_lang:
            return yaml_lang
        
        # Default to Chinese
        return 'zh'
    
    @classmethod
    def validate_environment(cls, model_name: str, model_size: str = "0.6B"):
        """
        Validate that all required environment configuration exists.
        
        Args:
            model_name: Name of the model to validate for
            model_size: Model size (for Qwen models)
            
        Raises:
            EnvironmentError: If required configuration is missing
        """
        errors = []
        
        # Check .env file exists
        if not ENV_FILE.exists():
            errors.append(
                f".env file not found at {ENV_FILE}\n"
                f"Please create it: cp .env.example .env"
            )
        
        # Check YAML config exists
        if not DEFAULT_YAML.exists():
            errors.append(
                f"Configuration file not found: {DEFAULT_YAML}\n"
                f"Please ensure config/default.yaml exists"
            )
        
        # Check model-specific requirements
        if model_name.lower() == 'theta':
            try:
                EnvConfig.get_qwen_model_path(model_size)
            except EnvironmentError as e:
                errors.append(str(e))
        
        if model_name.lower() in ('ctm', 'bertopic'):
            try:
                EnvConfig.get_sbert_model_path()
            except EnvironmentError as e:
                errors.append(str(e))
        
        if errors:
            raise EnvironmentError(
                "Environment validation failed:\n\n" + 
                "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(errors))
            )


# =============================================================================
# Convenience Functions
# =============================================================================

def load_config(model_name: str, **cli_args) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        model_name: Name of the model
        **cli_args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    return ConfigLoader.load(model_name, cli_args)


def get_default_params(model_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a model from YAML config.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Default parameters dictionary
    """
    return ConfigLoader.get_model_config(model_name)


def get_language(cli_lang: Optional[str] = None) -> str:
    """
    Get visualization language.
    
    Args:
        cli_lang: Language from CLI
        
    Returns:
        Language code
    """
    return ConfigLoader.get_language(cli_lang)


# =============================================================================
# Module Initialization
# =============================================================================

# Load environment on import
EnvConfig.load()
