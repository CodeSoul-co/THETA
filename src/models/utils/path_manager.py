#!/usr/bin/env python3
"""
Path Manager - Centralized path validation and management for THETA

Provides:
- user_id and dataset_name validation
- Standardized path generation
- DLC-compatible path normalization
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple


# Validation regex: alphanumeric, underscores, hyphens only
VALID_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


class PathValidationError(ValueError):
    """Raised when path component validation fails."""
    pass


def validate_name(name: str, field_name: str = "name") -> str:
    """
    Validate that a name contains only allowed characters.
    
    Args:
        name: The name to validate (user_id, dataset_name, task_name, etc.)
        field_name: Field name for error messages
    
    Returns:
        The validated name (unchanged if valid)
    
    Raises:
        PathValidationError: If name contains invalid characters
    """
    if not name:
        raise PathValidationError(f"Invalid {field_name}: cannot be empty.")
    
    if not VALID_NAME_PATTERN.match(name):
        raise PathValidationError(
            f"Invalid {field_name} '{name}'. "
            f"Only alphanumeric characters, underscores (_), and hyphens (-) are allowed. "
            f"No spaces, special characters, or non-ASCII characters."
        )
    
    return name


def validate_user_id(user_id: str) -> str:
    """Validate user_id format."""
    return validate_name(user_id, "user_id")


def validate_dataset_name(dataset_name: str) -> str:
    """Validate dataset_name format."""
    return validate_name(dataset_name, "dataset_name")


def validate_task_name(task_name: str) -> str:
    """Validate task_name format."""
    return validate_name(task_name, "task_name")


def validate_model_name(model_name: str) -> str:
    """Validate model_name format."""
    return validate_name(model_name, "model_name")


def normalize_path_component(name: str, lowercase: bool = False) -> str:
    """
    Normalize a path component for DLC compatibility.
    
    Args:
        name: The name to normalize
        lowercase: If True, convert to lowercase (recommended for DLC)
    
    Returns:
        Normalized name
    """
    if lowercase:
        return name.lower()
    return name


def validate_and_normalize(
    user_id: str,
    dataset_name: str,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
    lowercase: bool = False
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Validate and optionally normalize all path components.
    
    Args:
        user_id: User identifier
        dataset_name: Dataset name
        model_name: Model name (optional)
        task_name: Task/experiment name (optional)
        lowercase: Convert to lowercase for DLC compatibility
    
    Returns:
        Tuple of (user_id, dataset_name, model_name, task_name)
    
    Raises:
        PathValidationError: If any component is invalid
    """
    user_id = validate_user_id(user_id)
    dataset_name = validate_dataset_name(dataset_name)
    
    if model_name:
        model_name = validate_model_name(model_name)
    
    if task_name:
        task_name = validate_task_name(task_name)
    
    if lowercase:
        user_id = normalize_path_component(user_id, True)
        dataset_name = normalize_path_component(dataset_name, True)
        if model_name:
            model_name = normalize_path_component(model_name, True)
        if task_name:
            task_name = normalize_path_component(task_name, True)
    
    return user_id, dataset_name, model_name, task_name


def build_result_path(
    base_dir: Path,
    user_id: str,
    dataset_name: str,
    model_name: str,
    task_name: str,
    validate: bool = True
) -> Path:
    """
    Build a standardized result path.
    
    Structure: {base_dir}/{user_id}/{dataset_name}/{model_name}/{task_name}/
    
    Args:
        base_dir: Base result directory
        user_id: User identifier
        dataset_name: Dataset name
        model_name: Model name
        task_name: Task/experiment name
        validate: Whether to validate path components
    
    Returns:
        Path object for the result directory
    """
    if validate:
        user_id, dataset_name, model_name, task_name = validate_and_normalize(
            user_id, dataset_name, model_name, task_name
        )
    
    return Path(base_dir) / user_id / dataset_name / model_name / task_name


def build_workspace_path(
    base_dir: Path,
    user_id: str,
    dataset_name: str,
    validate: bool = True
) -> Path:
    """
    Build a standardized workspace path.
    
    Structure: {base_dir}/{user_id}/{dataset_name}/
    
    Args:
        base_dir: Base workspace directory
        user_id: User identifier
        dataset_name: Dataset name
        validate: Whether to validate path components
    
    Returns:
        Path object for the workspace directory
    """
    if validate:
        user_id = validate_user_id(user_id)
        dataset_name = validate_dataset_name(dataset_name)
    
    return Path(base_dir) / user_id / dataset_name


# CLI validation helper for argparse
def add_path_validation_to_args(args):
    """
    Validate path-related arguments in an argparse Namespace.
    
    Call this after parsing arguments to validate user_id, dataset, etc.
    
    Args:
        args: argparse.Namespace object
    
    Raises:
        PathValidationError: If validation fails
    """
    if hasattr(args, 'user_id') and args.user_id:
        validate_user_id(args.user_id)
    
    if hasattr(args, 'dataset') and args.dataset:
        validate_dataset_name(args.dataset)
    
    if hasattr(args, 'task_name') and args.task_name:
        validate_task_name(args.task_name)


if __name__ == '__main__':
    # Test validation
    import argparse
    
    parser = argparse.ArgumentParser(description='Test path validation')
    parser.add_argument('--user_id', type=str, help='User ID to validate')
    parser.add_argument('--dataset', type=str, help='Dataset name to validate')
    parser.add_argument('--task_name', type=str, help='Task name to validate')
    
    args = parser.parse_args()
    
    try:
        add_path_validation_to_args(args)
        print(f"✓ All validations passed")
        if args.user_id:
            print(f"  user_id: {args.user_id}")
        if args.dataset:
            print(f"  dataset: {args.dataset}")
        if args.task_name:
            print(f"  task_name: {args.task_name}")
    except PathValidationError as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)
