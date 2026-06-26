#!/usr/bin/env python
"""
ETM Topic Model Pipeline - Main Entry Point

Usage:
    python main.py train --dataset socialTwitter --mode zero_shot --num_topics 20
    python main.py evaluate --dataset socialTwitter --mode zero_shot
    python main.py visualize --dataset socialTwitter --mode zero_shot
    python main.py pipeline --dataset socialTwitter --mode zero_shot --num_topics 20
"""

import os
import sys
import json
import logging
import tempfile
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PipelineConfig, create_parser, config_from_args,
    DATA_DIR, EMBEDDING_DIR, ETM_DIR, get_qwen_model_path
)
from gpu_utils import configure_cuda_visible_devices
from artifact_utils import (
    BOW_MANIFEST_NAME,
    adapter_signature,
    commit_bow_artifacts,
    file_set_signature,
    has_valid_bow_manifest,
    validate_bow_manifest,
)


def setup_logging(
    config: PipelineConfig,
    local_rank: int = -1,
    use_ddp: bool = False
) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("ETM")
    logger.setLevel(logging.DEBUG if config.dev_mode else logging.INFO)

    for handler in list(logger.handlers):
        if getattr(handler, "_theta_owned", False):
            logger.removeHandler(handler)
            handler.close()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch._theta_owned = True

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    def prepare_log_file() -> str:
        os.makedirs(config.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(
            config.log_dir,
            f"{config.data.dataset}_{config.embedding.mode}_{timestamp}.log"
        )
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh._theta_owned = True
        logger.addHandler(fh)
        return log_file

    log_file = run_rank_zero(
        prepare_log_file,
        use_ddp,
        local_rank,
        "prepare log file",
    )
    if is_main_process(local_rank):
        logger.info(f"Log file: {log_file}")
    return logger


def setup_device(config: PipelineConfig, local_rank: int = -1) -> torch.device:
    """Setup compute device with optional DDP support

    Args:
        config: Pipeline configuration
        local_rank: Local rank for DDP (-1 for single GPU)

    Returns:
        torch.device: The device to use
    """
    if config.device == "cuda" and torch.cuda.is_available():
        if local_rank >= 0:
            # DDP mode: use the assigned GPU
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
        else:
            # Single GPU mode
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_global_rank(local_rank: int = -1) -> int:
    """Return the global DDP rank, falling back to the local rank."""
    return int(os.environ.get("RANK", local_rank))


def setup_ddp(local_rank: int, world_size: int):
    """Initialize Distributed Data Parallel

    Args:
        local_rank: Local rank of the current process
        world_size: Total number of processes
    """
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=get_global_rank(local_rank)
    )


def cleanup_ddp():
    """Clean up DDP resources"""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_rank_zero(
    operation: Callable[[], Any],
    use_ddp: bool,
    local_rank: int = -1,
    operation_name: str = "rank-zero operation",
) -> Any:
    """Run an operation on global rank zero and broadcast its small result."""
    if not (use_ddp and dist.is_initialized()):
        return operation()

    payload = [True, None, ""]
    if is_main_process(local_rank):
        try:
            payload[1] = operation()
        except Exception as exc:
            payload = [
                False,
                None,
                (
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                ),
            ]

    dist.broadcast_object_list(payload, src=0)
    if not payload[0]:
        raise RuntimeError(f"{operation_name} failed on rank 0: {payload[2]}")
    return payload[1]


def run_all_ranks(
    operation: Callable[[], Any],
    use_ddp: bool,
    operation_name: str = "all-ranks operation",
    signature_getter: Optional[Callable[[Any], Any]] = None,
    expected_signature: Any = None,
) -> Any:
    """Run a local operation on every rank and collectively report failures."""
    if not (use_ddp and dist.is_initialized()):
        return operation()

    local_result = None
    local_status = [get_global_rank(), True, "", None]
    try:
        local_result = operation()
        if signature_getter is not None:
            local_status[3] = signature_getter(local_result)
    except Exception as exc:
        local_status = [
            get_global_rank(),
            False,
            (
                f"{type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}"
            ),
            None,
        ]

    statuses = [None] * dist.get_world_size()
    dist.all_gather_object(statuses, local_status)
    failures = [
        f"rank {status[0]}: {status[2]}"
        for status in statuses
        if not status[1]
    ]
    if failures:
        raise RuntimeError(f"{operation_name} failed: {'; '.join(failures)}")
    if signature_getter is not None:
        signatures = [status[3] for status in statuses]
        reference = (
            expected_signature
            if expected_signature is not None
            else signatures[0]
        )
        mismatches = [
            f"rank {status[0]}"
            for status in statuses
            if status[3] != reference
        ]
        if mismatches:
            raise RuntimeError(
                f"{operation_name} signature mismatch: "
                f"{', '.join(mismatches)}"
            )
    return local_result


def is_main_process(local_rank: int = -1) -> bool:
    """Check if current process is the main process"""
    return local_rank < 0 or get_global_rank(local_rank) == 0


def load_texts(config: PipelineConfig, logger: logging.Logger) -> Tuple[List[str], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load texts from dataset, optionally with timestamps"""
    import pandas as pd

    csv_path = config.data.raw_data_path
    logger.info(f"Loading texts from {csv_path}")

    df = pd.read_csv(csv_path)

    # Find text column - support various naming conventions across datasets
    text_col = None
    for col in ['cleaned_content', 'clean_text', 'cleaned_text', 'text', 'content', 'Text', 'Consumer complaint narrative']:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")

    texts = df[text_col].fillna('').astype(str).tolist()

    # Find label column
    labels = None
    for col in ['label', 'Label', 'labels', 'category']:
        if col in df.columns:
            labels = df[col].values
            break

    # Load timestamps if enabled and column exists
    timestamps = None
    if config.visualization.enable_temporal and config.data.timestamp_column:
        ts_col = config.data.timestamp_column
        if ts_col in df.columns:
            try:
                timestamps = pd.to_datetime(df[ts_col], errors='coerce').values
                valid_count = pd.notna(timestamps).sum()
                logger.info(f"Loaded {valid_count}/{len(timestamps)} valid timestamps from column '{ts_col}'")
            except Exception as e:
                logger.warning(f"Failed to parse timestamps from column '{ts_col}': {e}")
        else:
            logger.warning(f"Timestamp column '{ts_col}' not found in data. Available: {df.columns.tolist()}")

    logger.info(f"Loaded {len(texts)} documents, text_col={text_col}, has_labels={labels is not None}")
    return texts, labels, timestamps


def generate_bow(
    texts: List[str],
    config: PipelineConfig,
    logger: logging.Logger
) -> Tuple[sparse.csr_matrix, List[str]]:
    """Generate BOW matrix"""
    from bow.vocab_builder import VocabBuilder, VocabConfig
    from bow.bow_generator import BOWGenerator

    logger.info(f"Generating BOW: vocab_size={config.bow.vocab_size}")

    # Build vocabulary
    vocab_config = VocabConfig(
        max_vocab_size=config.bow.vocab_size,
        min_df=config.bow.min_doc_freq,
        max_df_ratio=config.bow.max_doc_freq_ratio
    )
    vocab_builder = VocabBuilder(config=vocab_config, dev_mode=config.dev_mode)

    vocab_builder.add_documents(texts, dataset_name=config.data.dataset)
    vocab_builder.build_vocab()

    # Generate BOW
    bow_generator = BOWGenerator(vocab_builder, dev_mode=config.dev_mode)
    bow_output = bow_generator.generate_bow(texts, dataset_name=config.data.dataset)

    vocab = vocab_builder.get_vocab_list()

    logger.info(f"BOW shape: {bow_output.bow_matrix.shape}, vocab_size: {len(vocab)}")
    return bow_output.bow_matrix, vocab


def generate_vocab_embeddings(
    vocab: List[str],
    config: PipelineConfig,
    logger: logging.Logger
) -> np.ndarray:
    """Generate vocabulary embeddings using configured embedding provider."""
    from model.embedding_providers import create_cloud_embedding_provider, resolve_embedding_settings

    settings = resolve_embedding_settings(config=config)
    if config.embedding.mode == 'zero_shot' and settings.is_cloud:
        logger.info(
            "Generating vocab embeddings with cloud provider=%s model=%s",
            settings.cloud_provider,
            settings.model,
        )
        provider = create_cloud_embedding_provider(config=config)
        embeddings = provider.embed(
            vocab,
            batch_size=config.embedding.batch_size,
            show_progress=True,
            desc="Embedding vocabulary",
        )
        logger.info(f"Vocab embeddings shape: {embeddings.shape}")
        return embeddings

    if config.embedding.mode != 'zero_shot' and settings.is_cloud:
        logger.warning(
            "Embedding mode %s requires a local model for fine-tuning; "
            "ignoring cloud provider %s and using local Qwen.",
            config.embedding.mode,
            settings.cloud_provider,
        )

    from model.vocab_embedder import VocabEmbedder

    logger.info(f"Generating vocab embeddings for {len(vocab)} words with local Qwen")

    embedder = VocabEmbedder(
        model_path=config.embedding.model_path,
        device="cuda" if config.device == "cuda" else "cpu",
        batch_size=64,
        dev_mode=config.dev_mode
    )

    embeddings = embedder.embed_vocab(vocab)
    logger.info(f"Vocab embeddings shape: {embeddings.shape}")

    return embeddings


def load_doc_embeddings(
    config: PipelineConfig,
    logger: logging.Logger
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load document embeddings from result/{dataset}/{mode}/embeddings/"""
    embedding_dir = config.embeddings_dir
    # Support multiple embedding file naming conventions (prefer simple name first)
    emb_candidates = [
        os.path.join(embedding_dir, "embeddings.npy"),
        os.path.join(embedding_dir, f"{config.data.dataset}_{config.embedding.mode}_embeddings.npy"),
    ]
    emb_path = emb_candidates[0]  # default for error message
    for candidate in emb_candidates:
        if os.path.exists(candidate):
            emb_path = candidate
            break

    label_candidates = [
        os.path.join(embedding_dir, f"{config.data.dataset}_{config.embedding.mode}_labels.npy"),
        os.path.join(embedding_dir, "labels.npy"),
    ]
    label_path = label_candidates[0]
    for candidate in label_candidates:
        if os.path.exists(candidate):
            label_path = candidate
            break

    logger.info(f"Loading doc embeddings from {emb_path}")
    embeddings = np.load(emb_path)

    labels = None
    if os.path.exists(label_path):
        try:
            labels = np.load(label_path, allow_pickle=True)
        except Exception as e:
            logger.warning(f"Failed to load labels: {e}")

    logger.info(f"Doc embeddings: {embeddings.shape}, labels: {labels.shape if labels is not None else 'None'}")
    return embeddings, labels


def load_labels_for_supervised(
    config: PipelineConfig,
    logger: logging.Logger,
    num_docs: int
) -> Tuple[Optional[np.ndarray], int, Optional[LabelEncoder]]:
    """
    Load labels for supervised learning mode.

    IMPORTANT: Labels must be aligned with embeddings.npy (same row count).
    Priority order:
    1. labels.npy from embeddings directory (saved during preprocessing, guaranteed aligned)
    2. CSV file (only if row count matches num_docs)

    Args:
        config: Pipeline configuration
        logger: Logger instance
        num_docs: Number of documents in embeddings.npy (for alignment check)

    Returns:
        Tuple of (encoded_labels, num_classes, label_encoder)
        Returns (None, 0, None) if not in supervised mode or labels not available

    Note:
        The label_encoder should be saved to label_mapping.json for inference.
    """
    if config.embedding.mode != 'supervised':
        return None, 0, None

    # Priority 1: Try to load labels.npy from embeddings directory (aligned with embeddings)
    embedding_dir = config.embeddings_dir
    label_npy_candidates = [
        os.path.join(embedding_dir, "labels.npy"),
        os.path.join(embedding_dir, f"{config.data.dataset}_{config.embedding.mode}_labels.npy"),
    ]

    for label_path in label_npy_candidates:
        if os.path.exists(label_path):
            logger.info(f"Loading labels from preprocessed file: {label_path}")
            raw_labels = np.load(label_path, allow_pickle=True)

            # Verify alignment
            if len(raw_labels) != num_docs:
                logger.warning(
                    f"Labels count mismatch: labels.npy has {len(raw_labels)} rows, "
                    f"but embeddings has {num_docs} rows. Skipping this file."
                )
                continue

            # Encode labels
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(raw_labels)
            num_classes = len(label_encoder.classes_)

            logger.info(f"Loaded {len(encoded_labels)} labels, {num_classes} classes")
            logger.info(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

            return encoded_labels.astype(np.int64), num_classes, label_encoder

    # Priority 2: Fallback to CSV (with strict alignment check)
    csv_path = config.data.raw_data_path
    if not os.path.exists(csv_path):
        csv_path = config.data.cleaned_data_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Cannot find labels for supervised learning. "
            f"No labels.npy in {embedding_dir} and no CSV file found. "
            f"Tried: {config.data.raw_data_path}, {config.data.cleaned_data_path}"
        )

    logger.info(f"Loading labels from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # CRITICAL: Check alignment with embeddings
    if len(df) != num_docs:
        raise ValueError(
            f"ALIGNMENT ERROR: CSV has {len(df)} rows but embeddings has {num_docs} rows. "
            f"This can happen if preprocessing dropped some rows (e.g., invalid timestamps in DTM mode). "
            f"Please ensure labels.npy is saved during preprocessing, or use the same CSV that was preprocessed."
        )

    # Get label column name from config
    label_col = getattr(config, 'label_col', 'label')

    if label_col not in df.columns:
        available_cols = ', '.join(df.columns.tolist())
        raise ValueError(
            f"Label column '{label_col}' not found in dataset. "
            f"Available columns: [{available_cols}]. "
            f"Please specify the correct column name using --label_col argument."
        )

    raw_labels = df[label_col].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(raw_labels)
    num_classes = len(label_encoder.classes_)

    logger.info(f"Label column: '{label_col}'")
    logger.info(f"Unique classes: {num_classes}")
    logger.info(f"Class mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")
    logger.info(f"Labels shape: {encoded_labels.shape}")

    return encoded_labels.astype(np.int64), num_classes, label_encoder


def train_two_stage(
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    local_rank: int = -1,
    world_size: int = 1
) -> Dict[str, Any]:
    """
    Two-Stage Training Pipeline for THETA

    Stage 1: Embedding-LoRA Fine-tuning
    - Fine-tune Qwen embedding model with LoRA
    - Loss: Contrastive (unsupervised) or CE (supervised)
    - Output: adapter_config.json + adapter_model.safetensors

    Stage 2: ETM-KL Only Training
    - Load LoRA weights from Stage 1
    - Freeze embedding layer
    - Train ETM with KL divergence only

    Args:
        doc_embeddings: Document embeddings
        bow_matrix: Bag-of-words matrix
        vocab_embeddings: Vocabulary embeddings
        config: Pipeline configuration
        logger: Logger instance
        device: Torch device
        local_rank: Local rank for DDP
        world_size: Total number of GPUs for DDP

    Returns:
        Dict containing model, history, and other results
    """
    from model.theta.etm import ETM
    from data.dataloader import ETMDataset, create_dataloader
    from torch.utils.data import random_split
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    use_ddp = local_rank >= 0 and world_size > 1
    is_main = is_main_process(local_rank)

    # Load labels for supervised mode
    num_docs = doc_embeddings.shape[0]
    labels, num_classes, label_encoder = load_labels_for_supervised(config, logger, num_docs)

    # Determine loss type for Stage 1
    if config.embedding.mode == 'supervised':
        if labels is None:
            raise ValueError("Supervised mode requires labels")
        stage1_loss_type = 'ce'
        if is_main:
            logger.info(f"Stage 1: Using CE loss (supervised mode, num_classes={num_classes})")
    else:
        stage1_loss_type = 'contrastive'
        if is_main:
            logger.info(f"Stage 1: Using Contrastive loss (unsupervised mode)")

    # Create dataset
    use_sparse = doc_embeddings.shape[0] > 200000
    dataset = ETMDataset(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix,
        labels=labels,
        normalize_bow=True,
        dev_mode=config.dev_mode,
        keep_sparse=use_sparse
    )

    # Split data
    n_total = len(dataset)
    n_train = int(n_total * config.model.train_ratio)
    n_val = int(n_total * config.model.val_ratio)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Create dataloaders
    dl_num_workers = config.model.num_workers if n_total <= 200000 else min(config.model.num_workers, 2)
    dl_persistent = config.model.persistent_workers if n_total <= 200000 else False

    train_sampler = None
    if use_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size,
            rank=get_global_rank(local_rank), shuffle=True
        )

    train_loader = create_dataloader(
        train_dataset, batch_size=config.model.batch_size,
        shuffle=(train_sampler is None), num_workers=dl_num_workers,
        pin_memory=config.model.pin_memory, persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor, sampler=train_sampler
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=config.model.batch_size, shuffle=False,
        num_workers=dl_num_workers, pin_memory=config.model.pin_memory,
        persistent_workers=dl_persistent, prefetch_factor=config.model.prefetch_factor
    )

    if is_main:
        logger.info(f"Data splits: train={n_train}, val={n_val}, test={n_test}")

    # ========================================
    # STAGE 1: Embedding-LoRA Fine-tuning
    # ========================================
    if is_main:
        logger.info("=" * 60)
        logger.info("STAGE 1: Embedding-LoRA Fine-tuning")
        logger.info("=" * 60)

    # Load Qwen model
    model_path = get_qwen_model_path(config.model_size)
    if is_main:
        logger.info(f"Loading Qwen model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32  # Use float32 to match embeddings dtype
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=["q_proj", "v_proj"],  # Target attention layers
        bias="none"
    )

    # Apply LoRA to model
    model_stage1 = get_peft_model(base_model, lora_config)
    model_stage1 = model_stage1.to(device)

    # Only LoRA parameters are trainable
    trainable_params = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_stage1.parameters())
    if is_main:
        logger.info(f"LoRA params: trainable={trainable_params:,} / total={total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Wrap with DDP if needed
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model_stage1 = DDP(model_stage1, device_ids=[local_rank], output_device=local_rank)

    # Optimizer for Stage 1
    optimizer_stage1 = torch.optim.AdamW(
        model_stage1.parameters(),
        lr=config.model.stage1_lr,
        weight_decay=config.model.weight_decay
    )

    # Training loop for Stage 1
    best_val_loss_stage1 = float('inf')
    expected_adapter_signature = None
    history_stage1 = {'train_loss': [], 'val_loss': []}

    if is_main:
        logger.info(f"Starting Stage 1 training ({config.model.stage1_epochs} epochs)...")

    for epoch in range(config.model.stage1_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model_stage1.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}", disable=not is_main):
            doc_emb = batch['doc_embedding'].to(device)
            batch_labels = batch.get('label', None)
            if batch_labels is not None:
                batch_labels = batch_labels.to(device)

            optimizer_stage1.zero_grad()

            # Forward pass through LoRA model
            # Note: We need to reconstruct embeddings from doc_emb
            # For simplicity, we'll use the existing embeddings as targets
            outputs = model_stage1(inputs_embeds=doc_emb.unsqueeze(1))
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            # Compute loss based on mode
            if stage1_loss_type == 'ce':
                # Supervised: CE loss
                # Add a classification head
                if not hasattr(model_stage1, 'classifier'):
                    classifier = nn.Linear(embeddings.size(-1), num_classes).to(device)
                    if use_ddp:
                        model_stage1.module.classifier = classifier
                    else:
                        model_stage1.classifier = classifier

                logits = (model_stage1.module.classifier if use_ddp else model_stage1.classifier)(embeddings)
                loss = F.cross_entropy(logits, batch_labels)
            else:
                # Unsupervised: Contrastive loss
                # Normalize embeddings
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / config.model.contrastive_temp

                # Create positive pairs (same document)
                batch_size = embeddings.size(0)
                labels_contrastive = torch.arange(batch_size).to(device)

                # InfoNCE loss
                loss = F.cross_entropy(sim_matrix, labels_contrastive)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_stage1.parameters(), max_norm=1.0)
            optimizer_stage1.step()

            train_loss += loss.item() * doc_emb.size(0)

        train_loss /= n_train
        history_stage1['train_loss'].append(train_loss)

        # Validation
        model_stage1.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                doc_emb = batch['doc_embedding'].to(device)
                batch_labels = batch.get('label', None)
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)

                outputs = model_stage1(inputs_embeds=doc_emb.unsqueeze(1))
                embeddings = outputs.last_hidden_state[:, 0, :]

                if stage1_loss_type == 'ce':
                    logits = (model_stage1.module.classifier if use_ddp else model_stage1.classifier)(embeddings)
                    loss = F.cross_entropy(logits, batch_labels)
                else:
                    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                    sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / config.model.contrastive_temp
                    labels_contrastive = torch.arange(embeddings.size(0)).to(device)
                    loss = F.cross_entropy(sim_matrix, labels_contrastive)

                val_loss += loss.item() * doc_emb.size(0)

        val_loss /= n_val
        history_stage1['val_loss'].append(val_loss)

        if is_main:
            logger.info(f"Stage1 Epoch {epoch+1}/{config.model.stage1_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        improved_stage1 = val_loss < best_val_loss_stage1
        if improved_stage1:
            best_val_loss_stage1 = val_loss

        def save_lora_adapter():
            if improved_stage1:
                lora_save_dir = Path(config.model_dir) / "lora_adapter"
                lora_save_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = model_stage1.module if use_ddp else model_stage1
                model_to_save.save_pretrained(lora_save_dir)

                logger.info(f"✓ Stage 1 best model saved to {lora_save_dir}")
                logger.info(f"  - adapter_config.json")
                logger.info(f"  - adapter_model.safetensors")
                return adapter_signature(lora_save_dir)
            return None

        saved_signature = run_rank_zero(
            save_lora_adapter,
            use_ddp,
            local_rank,
            "save LoRA adapter",
        )
        if saved_signature is not None:
            expected_adapter_signature = saved_signature

    if is_main:
        logger.info("=" * 60)
        logger.info("Stage 1 Complete!")
        logger.info(f"Best val_loss: {best_val_loss_stage1:.4f}")
        logger.info("=" * 60)

    # Clean up Stage 1 model
    del model_stage1, optimizer_stage1
    torch.cuda.empty_cache()

    # ========================================
    # STAGE 2: ETM-KL Only Training
    # ========================================
    if is_main:
        logger.info("=" * 60)
        logger.info("STAGE 2: ETM-KL Only Training")
        logger.info("=" * 60)

    # Load LoRA weights from Stage 1
    lora_save_dir = Path(config.model_dir) / "lora_adapter"
    if is_main:
        logger.info(f"Loading LoRA weights from {lora_save_dir}")

    from peft import PeftModel

    def load_lora_adapter():
        local_signature = adapter_signature(lora_save_dir)
        base_model_stage2 = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        model = PeftModel.from_pretrained(base_model_stage2, lora_save_dir)
        return model, local_signature

    embedding_model, _ = run_all_ranks(
        load_lora_adapter,
        use_ddp,
        "load LoRA adapter",
        signature_getter=lambda result: result[1],
        expected_signature=expected_adapter_signature,
    )
    embedding_model = embedding_model.to(device)

    # Freeze embedding model completely
    for param in embedding_model.parameters():
        param.requires_grad = False
    embedding_model.eval()

    if is_main:
        logger.info("✓ Embedding model loaded and frozen")

    # Generate new embeddings using fine-tuned model for FULL dataset
    if is_main:
        logger.info("Generating embeddings with fine-tuned model for FULL dataset...")
        logger.info(f"Full dataset size: {len(doc_embeddings)} samples")

    # CRITICAL: Create a temporary DataLoader for the FULL dataset with shuffle=False
    # to ensure the generated embeddings align with the original bow_matrix row-by-row
    full_dataset = ETMDataset(
        doc_embeddings=doc_embeddings,  # Use ORIGINAL full embeddings
        bow_matrix=bow_matrix,
        labels=labels,
        normalize_bow=True,
        dev_mode=config.dev_mode,
        keep_sparse=use_sparse
    )

    full_loader = create_dataloader(
        full_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,  # CRITICAL: Must be False to maintain alignment with bow_matrix
        num_workers=dl_num_workers,
        pin_memory=config.model.pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor,
        drop_last=False  # CRITICAL: Must be False to process all samples
    )

    new_doc_embeddings = []
    embedding_model.eval()
    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Generating embeddings", disable=not is_main):
            doc_emb = batch['doc_embedding'].to(device)
            outputs = embedding_model(inputs_embeds=doc_emb.unsqueeze(1))
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            new_doc_embeddings.append(embeddings)

    new_doc_embeddings = np.vstack(new_doc_embeddings)

    if is_main:
        logger.info(f"✓ New embeddings generated: shape={new_doc_embeddings.shape}")
        logger.info(f"✓ Alignment verified: {new_doc_embeddings.shape[0]} embeddings == {bow_matrix.shape[0]} BOW rows")

    # Create NEW dataset and dataloaders for Stage 2 using new_doc_embeddings
    dataset_stage2 = ETMDataset(
        doc_embeddings=new_doc_embeddings,  # Use new embeddings from Stage 1
        bow_matrix=bow_matrix,
        labels=labels,
        normalize_bow=True,
        dev_mode=config.dev_mode,
        keep_sparse=use_sparse
    )

    # Split data (use same split as Stage 1)
    train_dataset_stage2, val_dataset_stage2, test_dataset_stage2 = random_split(
        dataset_stage2, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Create new samplers for Stage 2
    train_sampler_stage2 = None
    if use_ddp:
        train_sampler_stage2 = DistributedSampler(
            train_dataset_stage2, num_replicas=world_size,
            rank=get_global_rank(local_rank), shuffle=True
        )

    # Create new dataloaders for Stage 2
    train_loader_stage2 = create_dataloader(
        train_dataset_stage2, batch_size=config.model.batch_size,
        shuffle=(train_sampler_stage2 is None), num_workers=dl_num_workers,
        pin_memory=config.model.pin_memory, persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor, sampler=train_sampler_stage2
    )
    val_loader_stage2 = create_dataloader(
        val_dataset_stage2, batch_size=config.model.batch_size, shuffle=False,
        num_workers=dl_num_workers, pin_memory=config.model.pin_memory,
        persistent_workers=dl_persistent, prefetch_factor=config.model.prefetch_factor
    )

    # Create ETM model
    vocab_size = bow_matrix.shape[1]
    word_emb_tensor = torch.tensor(vocab_embeddings, dtype=torch.float32) if not config.model.train_word_embeddings else None

    model_stage2 = ETM(
        vocab_size=vocab_size,
        num_topics=config.model.num_topics,
        doc_embedding_dim=new_doc_embeddings.shape[1],
        word_embedding_dim=config.model.word_embedding_dim,
        hidden_dim=config.model.hidden_dim,
        encoder_dropout=config.model.encoder_dropout,
        word_embeddings=word_emb_tensor,
        train_word_embeddings=config.model.train_word_embeddings,
        num_classes=0,  # No classification in Stage 2
        dev_mode=config.dev_mode
    ).to(device)

    if use_ddp:
        model_stage2 = DDP(model_stage2, device_ids=[local_rank], output_device=local_rank)

    # Optimizer for Stage 2
    optimizer_stage2 = torch.optim.Adam(
        model_stage2.parameters(),
        lr=config.model.stage2_lr,
        weight_decay=config.model.weight_decay
    )

    # Training loop for Stage 2 (KL only)
    best_val_loss_stage2 = float('inf')
    history_stage2 = {'train_loss': [], 'val_loss': [], 'kl_loss': []}

    if is_main:
        logger.info(f"Starting Stage 2 training ({config.model.stage2_epochs} epochs, KL only)...")

    for epoch in range(config.model.stage2_epochs):
        if train_sampler_stage2 is not None:
            train_sampler_stage2.set_epoch(epoch)

        # KL annealing
        warmup_progress = min(1.0, (epoch + 1) / config.model.kl_warmup_epochs)
        kl_weight = config.model.kl_start + (config.model.kl_end - config.model.kl_start) * warmup_progress

        model_stage2.train()
        train_loss = 0.0
        train_kl = 0.0

        for batch in tqdm(train_loader_stage2, desc=f"Stage2 Epoch {epoch+1}", disable=not is_main):
            doc_emb = batch['doc_embedding'].to(device)
            bow = batch['bow'].to(device)

            optimizer_stage2.zero_grad()
            output = model_stage2(doc_emb, bow, mode='unsupervised', kl_weight=kl_weight)

            # Stage 2: KL divergence only
            loss = output['kl_loss']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_stage2.parameters(), max_norm=1.0)
            optimizer_stage2.step()

            train_loss += loss.item() * doc_emb.size(0)
            train_kl += output['kl_loss'].item() * doc_emb.size(0)

        train_loss /= n_train
        train_kl /= n_train
        history_stage2['train_loss'].append(train_loss)
        history_stage2['kl_loss'].append(train_kl)

        # Validation
        model_stage2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_stage2:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)

                output = model_stage2(doc_emb, bow, mode='unsupervised', kl_weight=kl_weight)
                loss = output['kl_loss']

                val_loss += loss.item() * doc_emb.size(0)

        val_loss /= n_val
        history_stage2['val_loss'].append(val_loss)

        if is_main:
            logger.info(f"Stage2 Epoch {epoch+1}/{config.model.stage2_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, kl={train_kl:.4f}")

        # Save best model
        if val_loss < best_val_loss_stage2:
            best_val_loss_stage2 = val_loss
            best_model_state = model_stage2.state_dict()

    if is_main:
        logger.info("=" * 60)
        logger.info("Stage 2 Complete!")
        logger.info(f"Best val_loss: {best_val_loss_stage2:.4f}")
        logger.info("=" * 60)

    # Return results
    return {
        'model': model_stage2,
        'history': {
            'stage1': history_stage1,
            'stage2': history_stage2
        },
        'best_model_state': best_model_state,
        'embedding_model': embedding_model
    }


def train_etm(
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    local_rank: int = -1,
    world_size: int = 1
) -> Dict[str, Any]:
    """Train ETM model with optional DDP support

    Args:
        doc_embeddings: Document embeddings
        bow_matrix: Bag-of-words matrix
        vocab_embeddings: Vocabulary embeddings
        config: Pipeline configuration
        logger: Logger instance
        device: Torch device
        local_rank: Local rank for DDP (-1 for single GPU)
        world_size: Total number of GPUs for DDP

    Returns:
        Dict containing model, history, and other results
    """
    from model.theta.etm import ETM
    from data.dataloader import ETMDataset, create_dataloader
    from torch.utils.data import random_split

    use_ddp = local_rank >= 0 and world_size > 1

    if is_main_process(local_rank):
        logger.info(f"Device: {device}")
        logger.info(f"DDP enabled: {use_ddp}, world_size: {world_size}")
        logger.info(f"Config: num_topics={config.model.num_topics}, epochs={config.model.epochs}, batch_size={config.model.batch_size}")
        logger.info(f"Mode: {config.embedding.mode}")

    # Auto-enable two-stage training for supervised/unsupervised modes
    # zero_shot: No fine-tuning, use pretrained embeddings directly
    # supervised/unsupervised: Auto fine-tune embeddings with LoRA
    enable_two_stage = config.embedding.mode in ['supervised', 'unsupervised']

    if enable_two_stage:
        if is_main_process(local_rank):
            logger.info("=" * 60)
            logger.info("TWO-STAGE TRAINING (Auto-enabled for supervised/unsupervised mode)")
            logger.info(f"Stage 1: Embedding-LoRA fine-tuning ({config.model.stage1_epochs} epochs)")
            logger.info(f"Stage 2: ETM-KL only training ({config.model.stage2_epochs} epochs)")
            logger.info("=" * 60)

        # Execute two-stage training
        return train_two_stage(
            doc_embeddings, bow_matrix, vocab_embeddings,
            config, logger, device, local_rank, world_size
        )
    else:
        if is_main_process(local_rank):
            logger.info("=" * 60)
            logger.info("ZERO-SHOT MODE: Using pretrained embeddings without fine-tuning")
            logger.info("=" * 60)

    # Load labels for supervised mode (pass num_docs for alignment check)
    num_docs = doc_embeddings.shape[0]
    labels, num_classes, label_encoder = load_labels_for_supervised(config, logger, num_docs)

    if config.embedding.mode == 'supervised':
        if labels is None:
            raise ValueError(
                "Supervised mode requires labels. Please ensure labels.npy exists in embeddings directory, "
                "or your CSV file contains a label column (specify with --label_col if needed)."
            )
        if is_main_process(local_rank):
            logger.info(f"Supervised mode enabled: num_classes={num_classes}")

    # Create dataset
    # For large datasets, keep BOW in sparse format to avoid massive memory usage
    use_sparse = doc_embeddings.shape[0] > 200000
    if use_sparse and is_main_process(local_rank):
        logger.info(f"Large dataset ({doc_embeddings.shape[0]:,} docs): keeping BOW sparse to save memory")
    dataset = ETMDataset(
        doc_embeddings=doc_embeddings,
        bow_matrix=bow_matrix,
        labels=labels,  # Pass labels for supervised mode
        normalize_bow=True,
        dev_mode=config.dev_mode,
        keep_sparse=use_sparse
    )

    # Split data
    n_total = len(dataset)
    n_train = int(n_total * config.model.train_ratio)
    n_val = int(n_total * config.model.val_ratio)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(config.seed)
    )

    # Scale DataLoader workers for large datasets
    # With keep_sparse=True, BOW is CSR (~few hundred MB) so workers are safe.
    # But limit workers to avoid excessive memory from forked doc_embeddings tensor.
    dl_num_workers = config.model.num_workers
    dl_persistent = config.model.persistent_workers
    dl_pin_memory = config.model.pin_memory
    if n_total > 200000:
        dl_num_workers = min(dl_num_workers, 2)
        dl_persistent = False
        if is_main_process(local_rank):
            logger.info(f"Large dataset ({n_total:,} docs): using num_workers={dl_num_workers}, persistent=False")

    # Create samplers for DDP
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=get_global_rank(local_rank),
            shuffle=True
        )

    train_loader = create_dataloader(
        train_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor,
        sampler=train_sampler
    )
    val_loader = create_dataloader(
        val_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor
    )
    test_loader = create_dataloader(
        test_dataset, 
        batch_size=config.model.batch_size, 
        shuffle=False,
        num_workers=dl_num_workers,
        pin_memory=dl_pin_memory,
        persistent_workers=dl_persistent,
        prefetch_factor=config.model.prefetch_factor
    )

    if is_main_process(local_rank):
        logger.info(f"Data splits: train={n_train}, val={n_val}, test={n_test}")

    # Create model
    vocab_size = bow_matrix.shape[1]

    # Use pretrained embeddings only if train_word_embeddings is False
    # Otherwise use random initialization for better learning
    if config.model.train_word_embeddings:
        word_emb_tensor = None  # Use random init
        logger.info("Using random word embeddings (trainable)")
    else:
        word_emb_tensor = torch.tensor(vocab_embeddings, dtype=torch.float32)
        logger.info("Using pretrained word embeddings (frozen)")

    model = ETM(
        vocab_size=vocab_size,
        num_topics=config.model.num_topics,
        doc_embedding_dim=config.model.doc_embedding_dim,
        word_embedding_dim=config.model.word_embedding_dim,
        hidden_dim=config.model.hidden_dim,
        encoder_dropout=config.model.encoder_dropout,
        word_embeddings=word_emb_tensor,
        train_word_embeddings=config.model.train_word_embeddings,
        num_classes=num_classes,  # For supervised mode
        dev_mode=config.dev_mode
    ).to(device)

    # Wrap model with DDP if using multi-GPU
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process(local_rank):
            logger.info(f"Model wrapped with DistributedDataParallel on {world_size} GPUs")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process(local_rank):
        logger.info(f"Model params: total={total_params:,}, trainable={trainable_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay
    )

    scheduler = None
    if config.model.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.model.scheduler_factor,
            patience=config.model.scheduler_patience
        )

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'recon_loss': [], 'kl_loss': [],
        'perplexity': []  # Track perplexity during training
    }

    # Initialize AdaptiveLossWeighter for supervised mode
    loss_weighter = None
    if config.embedding.mode == 'supervised':
        from model.theta.loss_weighter import AdaptiveLossWeighter
        loss_weighter = AdaptiveLossWeighter(
            warmup_steps=100,
            ema_decay=0.99,
            target_ratio={'ce': 0.7, 'recon': 0.3},  # Prioritize classification
            warmup_ce_weight=10.0,  # High initial CE weight to prevent classifier drift
            min_weight=0.1,
            max_weight=50.0
        )
        if is_main_process(local_rank):
            logger.info("AdaptiveLossWeighter initialized for supervised mode")
            logger.info(f"  Target ratio: CE=70%, Recon=30%")
            logger.info(f"  Warmup CE weight: 10.0 (for first 100 steps)")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    if is_main_process(local_rank):
        logger.info("=" * 60)
        logger.info("Starting training...")

    for epoch in range(config.model.epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # KL annealing: linear warmup from kl_start to kl_end
        # Free bits in loss function handles posterior collapse, so no minimum here
        warmup_progress = min(1.0, (epoch + 1) / config.model.kl_warmup_epochs)
        kl_weight = config.model.kl_start + (config.model.kl_end - config.model.kl_start) * warmup_progress

        # Training
        model.train()
        train_loss = 0.0

        # Get current mode
        current_mode = config.embedding.mode

        for batch in train_loader:
            doc_emb = batch['doc_embedding'].to(device)
            bow = batch['bow'].to(device)
            batch_labels = batch.get('label', None)
            if batch_labels is not None:
                batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            output = model(doc_emb, bow, labels=batch_labels, mode=current_mode, kl_weight=kl_weight)

            # Compute loss with adaptive weighting for supervised mode
            if current_mode == 'supervised' and loss_weighter is not None:
                # Get adaptive weights (only for CE and Recon, NOT KL)
                weights = loss_weighter.update({
                    'ce': output['ce_loss'],
                    'recon': output['recon_loss']
                })
                # Compute weighted loss: w_ce * CE + w_recon * Recon + KL (KL uses fixed weight)
                loss = (weights['ce'] * output['ce_loss'] + 
                        weights['recon'] * output['recon_loss'] + 
                        output['kl_loss'])  # kl_loss already includes kl_weight
            else:
                loss = output['total_loss']

            # Only backward if not zero_shot mode
            if current_mode != 'zero_shot':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * doc_emb.size(0)

        train_loss /= n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_ce = 0.0  # Track CE loss for supervised mode
        val_contrastive = 0.0  # Track contrastive loss for unsupervised mode

        with torch.no_grad():
            for batch in val_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)
                batch_labels = batch.get('label', None)
                if batch_labels is not None:
                    batch_labels = batch_labels.to(device)

                output = model(doc_emb, bow, labels=batch_labels, mode=current_mode, kl_weight=kl_weight)

                # Use same weighting logic as training for consistent val_loss
                if current_mode == 'supervised' and loss_weighter is not None:
                    weights = loss_weighter.get_weights()  # Use frozen weights, don't update
                    batch_loss = (weights['ce'] * output['ce_loss'].item() + 
                                  weights['recon'] * output['recon_loss'].item() + 
                                  output['kl_loss'].item())
                    val_ce += output['ce_loss'].item() * doc_emb.size(0)
                else:
                    batch_loss = output['total_loss'].item()
                    # Track contrastive loss for unsupervised mode
                    if 'contrastive_loss' in output:
                        val_contrastive += output['contrastive_loss'].item() * doc_emb.size(0)

                val_loss += batch_loss * doc_emb.size(0)
                val_recon += output['recon_loss'].item() * doc_emb.size(0)
                val_kl += output['kl_loss'].item() * doc_emb.size(0)

        val_loss /= n_val
        val_recon /= n_val
        val_kl /= n_val
        if current_mode == 'supervised':
            val_ce /= n_val
        if current_mode == 'unsupervised':
            val_contrastive /= n_val

        # Compute perplexity on validation set
        # Perplexity = exp(negative log-likelihood per word)
        # We compute it properly using the model's reconstruction
        with torch.no_grad():
            total_nll = 0.0
            total_words = 0
            for batch in val_loader:
                doc_emb = batch['doc_embedding'].to(device)
                bow = batch['bow'].to(device)

                # Get topic distribution and word distribution
                # Handle DDP wrapped model
                base_model = model.module if use_ddp else model
                theta_batch, _, _ = base_model.encoder(doc_emb)
                beta = base_model.decoder.get_beta()

                # Compute word probabilities: p(w|d) = sum_k theta_dk * beta_kw
                word_probs = torch.matmul(theta_batch, beta)  # (batch, vocab)
                word_probs = torch.clamp(word_probs, min=1e-10)

                # Negative log-likelihood per document
                log_probs = torch.log(word_probs)
                nll = -torch.sum(bow * log_probs, dim=1)  # (batch,)
                doc_lengths = bow.sum(dim=1)  # (batch,)

                total_nll += nll.sum().item()
                total_words += doc_lengths.sum().item()

            # Perplexity = exp(NLL / total_words)
            val_perplexity = np.exp(total_nll / total_words) if total_words > 0 else float('inf')

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['recon_loss'].append(val_recon)
        history['kl_loss'].append(val_kl)
        history['perplexity'].append(val_perplexity)

        # Check for improvement
        improved = val_loss < best_val_loss - config.model.min_delta
        if improved:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        # Learning rate scheduler
        if scheduler:
            scheduler.step(val_loss)

        if is_main_process(local_rank):
            if current_mode == 'supervised' and loss_weighter is not None:
                weights = loss_weighter.get_weights()
                logger.info(
                    f"Epoch {epoch+1:3d}/{config.model.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"CE: {val_ce:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f} | "
                    f"w_ce: {weights['ce']:.2f} | PPL: {val_perplexity:.1f} {marker}"
                )
            elif current_mode == 'unsupervised':
                logger.info(
                    f"Epoch {epoch+1:3d}/{config.model.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"NLL: {val_recon:.4f} | Contr: {val_contrastive:.4f} | KL: {val_kl:.4f} | "
                    f"PPL: {val_perplexity:.1f} | KL_w: {kl_weight:.3f} {marker}"
                )
            else:  # zero_shot
                logger.info(
                    f"Epoch {epoch+1:3d}/{config.model.epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"PPL: {val_perplexity:.1f} {marker}"
                )

        # Periodic topic quality check (every 10 epochs or at end) - only on main process
        if is_main_process(local_rank) and ((epoch + 1) % 10 == 0 or epoch == config.model.epochs - 1 or improved):
            with torch.no_grad():
                base_model = model.module if use_ddp else model
                beta = base_model.decoder.get_beta().cpu().numpy()
                # Check topic diversity: how different are topics from each other
                top_k = 10
                topic_top_words = []
                for k in range(beta.shape[0]):
                    top_indices = beta[k].argsort()[-top_k:][::-1]
                    topic_top_words.append(set(top_indices))

                # Compute pairwise Jaccard similarity
                total_overlap = 0
                n_pairs = 0
                for i in range(len(topic_top_words)):
                    for j in range(i+1, len(topic_top_words)):
                        intersection = len(topic_top_words[i] & topic_top_words[j])
                        union = len(topic_top_words[i] | topic_top_words[j])
                        total_overlap += intersection / union if union > 0 else 0
                        n_pairs += 1
                avg_overlap = total_overlap / n_pairs if n_pairs > 0 else 0
                diversity = 1 - avg_overlap

                # Check beta sharpness (how peaked are topic distributions)
                beta_max = beta.max(axis=1).mean()
                beta_entropy = -np.sum(beta * np.log(beta + 1e-10), axis=1).mean()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"  Topic Quality: diversity={diversity:.3f}, beta_max={beta_max:.4f}, entropy={beta_entropy:.2f}")

        # Early stopping
        if config.model.early_stopping and patience_counter >= config.model.patience:
            if is_main_process(local_rank):
                logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            doc_emb = batch['doc_embedding'].to(device)
            bow = batch['bow'].to(device)
            output = model(doc_emb, bow, kl_weight=1.0)
            test_loss += output['total_loss'].item() * doc_emb.size(0)
    test_loss /= n_test

    history['test_loss'] = test_loss
    history['best_val_loss'] = best_val_loss
    history['epochs_trained'] = epoch + 1

    if is_main_process(local_rank):
        logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")

    # Return the base model (unwrap DDP if needed)
    base_model = model.module if use_ddp else model

    return {
        'model': base_model,
        'history': history,
        'best_val_loss': best_val_loss,
        'test_loss': test_loss
    }


def save_bow_artifacts(
    bow_matrix,
    vocab: List[str],
    vocab_embeddings: np.ndarray,
    bow_dir: str,
) -> str:
    """Atomically publish a committed canonical sparse BOW artifact set."""
    os.makedirs(bow_dir, exist_ok=True)
    sparse_matrix = (
        bow_matrix
        if sparse.issparse(bow_matrix)
        else sparse.csr_matrix(bow_matrix)
    )
    temporary_paths = {}
    try:
        descriptor, matrix_path = tempfile.mkstemp(
            prefix=".bow_matrix.",
            suffix=".npz",
            dir=bow_dir,
        )
        os.close(descriptor)
        sparse.save_npz(matrix_path, sparse_matrix)
        temporary_paths["matrix"] = Path(matrix_path)

        descriptor, embeddings_path = tempfile.mkstemp(
            prefix=".vocab_embeddings.",
            suffix=".npy",
            dir=bow_dir,
        )
        os.close(descriptor)
        np.save(embeddings_path, vocab_embeddings)
        temporary_paths["vocab_embeddings"] = Path(embeddings_path)

        descriptor, vocab_path = tempfile.mkstemp(
            prefix=".vocab.",
            suffix=".txt",
            dir=bow_dir,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            file_obj.write('\n'.join(vocab))
            file_obj.flush()
            os.fsync(file_obj.fileno())
        temporary_paths["vocab"] = Path(vocab_path)

        manifest = commit_bow_artifacts(
            bow_dir,
            temporary_paths,
            {
                "matrix": {
                    "shape": list(sparse_matrix.shape),
                    "format": "scipy_csr_npz",
                },
                "vocab_embeddings": {
                    "shape": list(vocab_embeddings.shape),
                    "dtype": str(vocab_embeddings.dtype),
                },
                "vocab": {
                    "count": len(vocab),
                    "encoding": "utf-8",
                },
            },
        )
        return manifest["signature"]
    finally:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()


def load_bow_artifacts(bow_dir: str):
    """Load canonical sparse BOW artifacts, with dense legacy fallback."""
    sparse_path = os.path.join(bow_dir, "bow_matrix.npz")
    dense_path = os.path.join(bow_dir, "bow_matrix.npy")
    manifest_path = os.path.join(bow_dir, BOW_MANIFEST_NAME)
    vocab_path = os.path.join(bow_dir, "vocab.txt")
    vocab_emb_path = os.path.join(bow_dir, "vocab_embeddings.npy")

    manifest = None
    if os.path.exists(manifest_path):
        manifest, signature = validate_bow_manifest(bow_dir)
        bow_matrix = sparse.load_npz(sparse_path)
    elif os.path.exists(sparse_path):
        raise ValueError("canonical BOW matrix exists without a valid manifest")
    elif os.path.exists(dense_path):
        bow_matrix = np.load(dense_path)
        signature = file_set_signature(
            bow_dir,
            ("bow_matrix.npy", "vocab_embeddings.npy", "vocab.txt"),
            "theta.bow-artifacts.legacy-dense",
        )
    else:
        raise FileNotFoundError(f"No BOW matrix found in {bow_dir}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    vocab_embeddings = np.load(vocab_emb_path)
    if manifest is not None:
        artifacts = manifest["artifacts"]
        if list(bow_matrix.shape) != artifacts["matrix"]["shape"]:
            raise ValueError("BOW matrix shape mismatch")
        if list(vocab_embeddings.shape) != artifacts["vocab_embeddings"]["shape"]:
            raise ValueError("vocab embedding shape mismatch")
        if str(vocab_embeddings.dtype) != artifacts["vocab_embeddings"]["dtype"]:
            raise ValueError("vocab embedding dtype mismatch")
        if len(vocab) != artifacts["vocab"]["count"]:
            raise ValueError("vocab count mismatch")
    return bow_matrix, vocab, vocab_embeddings, signature


def save_results(
    model,
    history: Dict,
    vocab: List[str],
    doc_embeddings: np.ndarray,
    bow_matrix: sparse.csr_matrix,
    vocab_embeddings: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    device: torch.device,
    timestamps: Optional[np.ndarray] = None
) -> str:
    """Save training results to result/{dataset}/{mode}/ subdirectories"""
    # Create all output directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.bow_dir, exist_ok=True)
    os.makedirs(config.evaluation_dir, exist_ok=True)
    os.makedirs(config.visualization_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Saving results to {config.result_dir}")

    model.eval()
    with torch.no_grad():
        # Get theta for all documents (batch processing for large datasets)
        n_docs = doc_embeddings.shape[0]
        if n_docs > 100000:
            # Process in batches to avoid GPU OOM
            batch_size = 10000
            theta_parts = []
            for start in range(0, n_docs, batch_size):
                end = min(start + batch_size, n_docs)
                batch_emb = torch.tensor(doc_embeddings[start:end], dtype=torch.float32).to(device)
                batch_theta = model.get_theta(batch_emb).cpu().numpy()
                theta_parts.append(batch_theta)
                del batch_emb
            theta = np.concatenate(theta_parts, axis=0)
            del theta_parts
        else:
            doc_emb_tensor = torch.tensor(doc_embeddings, dtype=torch.float32).to(device)
            theta = model.get_theta(doc_emb_tensor).cpu().numpy()

        # Get beta
        beta = model.get_beta().cpu().numpy()

        # Get topic embeddings
        topic_emb = model.get_topic_embeddings().cpu().numpy()

        # Get topic words
        topic_words = model.get_topic_words(top_k=20, vocab=vocab)

    # Save model outputs to model_dir (fixed filenames without timestamp)
    np.save(os.path.join(config.model_dir, "theta.npy"), theta)
    np.save(os.path.join(config.model_dir, "beta.npy"), beta)
    np.save(os.path.join(config.model_dir, "topic_embeddings.npy"), topic_emb)

    # Save topic words
    topic_words_dict = {str(k): words for k, words in topic_words}
    with open(os.path.join(config.model_dir, "topic_words.json"), 'w') as f:
        json.dump(topic_words_dict, f, indent=2, ensure_ascii=False)

    # Save training history
    with open(os.path.join(config.model_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    # Save model weights
    torch.save(model.state_dict(), os.path.join(config.model_dir, "etm_model.pt"))

    # Save label mapping for supervised mode (for inference)
    if config.embedding.mode == 'supervised' and label_encoder is not None:
        label_mapping = {
            "index_to_label": {int(i): str(label) for i, label in enumerate(label_encoder.classes_)},
            "label_to_index": {str(label): int(i) for i, label in enumerate(label_encoder.classes_)},
            "num_classes": int(num_classes)
        }
        label_mapping_path = os.path.join(config.model_dir, "label_mapping.json")
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Label mapping saved to {label_mapping_path}")

    # Save config to exp_dir (parent of model_dir) with mode suffix
    mode = config.embedding.mode if hasattr(config.embedding, 'mode') else 'unsupervised'
    config_path = os.path.join(config.exp_dir, f"config_{mode}.json")
    config.save(config_path)
    logger.info(f"Config saved to {config_path}")

    # Save timestamps if available (for temporal analysis)
    if timestamps is not None and len(timestamps) > 0:
        ts_path = os.path.join(config.result_dir, "timestamps.npy")
        np.save(ts_path, timestamps)
        logger.info(f"Timestamps saved to {ts_path}")

    # Log top words
    logger.info("=" * 60)
    logger.info("Top 10 words per topic:")
    for topic_idx, words in topic_words[:10]:
        word_str = ", ".join([w for w, _ in words[:10]])
        logger.info(f"  Topic {topic_idx}: {word_str}")

    logger.info(f"Results saved with timestamp: {timestamp}")
    return timestamp


def run_evaluation(
    config: PipelineConfig,
    logger: logging.Logger,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Run evaluation on trained model"""
    from evaluation.topic_metrics import compute_all_metrics
    from evaluation.topic_metrics import TopicMetrics

    # Load results from model_dir (fixed filenames)
    theta_path = os.path.join(config.model_dir, "theta.npy")
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"No results found in {config.model_dir}")

    logger.info(f"Evaluating results from: {config.model_dir}")

    # Load results from model_dir
    theta = np.load(os.path.join(config.model_dir, "theta.npy"))
    beta = np.load(os.path.join(config.model_dir, "beta.npy"))

    with open(os.path.join(config.model_dir, "topic_words.json"), 'r') as f:
        topic_words = json.load(f)

    # Load BOW matrix from bow_dir (or regenerate if not exists)
    try:
        bow_matrix, _, _, _ = load_bow_artifacts(config.bow_dir)
        logger.info(f"Loaded BOW from {config.bow_dir}")
    except FileNotFoundError:
        texts, _, _ = load_texts(config, logger)
        bow_matrix, vocab = generate_bow(texts, config, logger)

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_all_metrics(
        beta=beta,
        theta=theta,
        doc_term_matrix=bow_matrix,
        top_k_coherence=config.evaluation.top_k_coherence,
        top_k_diversity=config.evaluation.top_k_diversity
    )

    # Log metrics (7 Core Metrics Standard)
    logger.info("=" * 60)
    logger.info("7 Core Metrics Results:")
    logger.info(f"  1. TD:          {metrics.get('TD', 0):.4f}")
    logger.info(f"  2. iRBO:        {metrics.get('iRBO', 0):.4f}")
    logger.info(f"  3. NPMI:        {metrics.get('NPMI', 0):.4f}")
    logger.info(f"  4. C_V:         {metrics.get('C_V', 0):.4f}")
    logger.info(f"  5. UMass:       {metrics.get('UMass', 0):.4f}")
    logger.info(f"  6. Exclusivity: {metrics.get('Exclusivity', 0):.4f}")
    logger.info(f"  7. PPL:         {metrics.get('PPL', 1000):.2f}")
    logger.info("=" * 60)

    # Save metrics to exp_dir (fixed filename)
    # Save metrics to exp_dir with mode suffix
    mode = config.embedding.mode if hasattr(config.embedding, 'mode') else 'unsupervised'
    metrics_path = os.path.join(config.exp_dir, f"metrics_{mode}.json")

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def run_visualization(
    config: PipelineConfig,
    logger: logging.Logger,
    timestamp: Optional[str] = None,
    use_wordcloud: bool = True
):
    """
    Generate visualizations using the unified visualization generator.
    Delegates entirely to run_all_visualizations() which handles:
    - VisualizationGenerator (global + per-topic charts)
    - TopicVisualizer (additional charts: wordclouds, pyLDAvis, etc.)
    - Summary report
    """
    from visualization.topic_visualizer import load_etm_results
    from visualization import run_all_visualizations

    # Find latest results if no timestamp
    if timestamp is None:
        result_files = sorted(Path(config.model_dir).glob("theta_*.npy"), reverse=True)
        if not result_files:
            raise FileNotFoundError(f"No results found in {config.model_dir}")
        timestamp = result_files[0].stem.replace("theta_", "")

    logger.info(f"Generating visualizations for timestamp: {timestamp}")

    # Save topic words to dedicated folder (before visualization)
    try:
        results = load_etm_results(config.model_dir, timestamp)
        topic_words_dir = os.path.join(config.result_dir, "topic_words")
        os.makedirs(topic_words_dir, exist_ok=True)

        topic_words_path = os.path.join(topic_words_dir, f"topic_words_{timestamp}.json")
        with open(topic_words_path, 'w', encoding='utf-8') as f:
            json.dump(results['topic_words'], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved topic words to {topic_words_path}")

        topic_words_txt_path = os.path.join(topic_words_dir, f"topic_words_{timestamp}.txt")
        with open(topic_words_txt_path, 'w', encoding='utf-8') as f:
            for topic_id, words in results['topic_words']:
                word_list = [f"{word}({prob:.4f})" for word, prob in words]
                f.write(f"Topic {topic_id}: {', '.join(word_list)}\n")
        logger.info(f"Saved topic words text to {topic_words_txt_path}")
    except Exception as e:
        logger.warning(f"Failed to save topic words: {e}")

    # Determine visualization parameters
    # Use exp_dir to ensure visualization finds the correct model outputs
    viz_result_dir = config.output_base_dir
    viz_model_size = config.model_size
    viz_dataset = config.data.dataset
    viz_mode = config.embedding.mode
    # Extract experiment ID from exp_dir (e.g., "exp_20260426_202624")
    viz_model_exp = os.path.basename(config.exp_dir) if config.exp_dir != config.dataset_base_dir else None

    try:
        output_dir = run_all_visualizations(
            result_dir=viz_result_dir,
            dataset=viz_dataset,
            mode=viz_mode,
            model_size=viz_model_size,
            model_exp=viz_model_exp,
            output_dir=config.visualization_dir,
            language=config.visualization.language,
            dpi=config.visualization.dpi
        )
        logger.info(f"All visualizations saved to {output_dir}")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        logger.error(traceback.format_exc())


def run_train(config: PipelineConfig, logger: logging.Logger, local_rank: int = -1, world_size: int = 1):
    """Run training pipeline with optional DDP support

    Args:
        config: Pipeline configuration
        logger: Logger instance
        local_rank: Local rank for DDP (-1 for single GPU)
        world_size: Total number of GPUs for DDP
    """
    use_ddp = local_rank >= 0 and world_size > 1
    device = setup_device(config, local_rank)

    # Load data (with optional timestamps for temporal analysis)
    texts, _, timestamps = load_texts(config, logger)

    # Check if BOW already exists (shared across modes)
    bow_sparse_path = os.path.join(config.bow_dir, "bow_matrix.npz")
    bow_path = os.path.join(config.bow_dir, "bow_matrix.npy")
    vocab_path = os.path.join(config.bow_dir, "vocab.txt")
    vocab_emb_path = os.path.join(config.bow_dir, "vocab_embeddings.npy")

    def bow_artifact_signature_if_valid():
        manifest_path = os.path.join(config.bow_dir, BOW_MANIFEST_NAME)
        if os.path.exists(manifest_path):
            if not has_valid_bow_manifest(config.bow_dir):
                return None
            _, signature = validate_bow_manifest(config.bow_dir)
            return signature
        if os.path.exists(bow_sparse_path):
            return None
        if (
            os.path.exists(bow_path)
            and os.path.exists(vocab_path)
            and os.path.exists(vocab_emb_path)
        ):
            return file_set_signature(
                config.bow_dir,
                ("bow_matrix.npy", "vocab_embeddings.npy", "vocab.txt"),
                "theta.bow-artifacts.legacy-dense",
            )
        return None

    if use_ddp:
        expected_bow_signature = run_rank_zero(
            bow_artifact_signature_if_valid,
            use_ddp,
            local_rank,
            "check BOW artifacts",
        )
    else:
        expected_bow_signature = bow_artifact_signature_if_valid()

    bow_exists = expected_bow_signature is not None
    if bow_exists:
        # Load existing BOW (shared across modes for fair comparison)
        logger.info(f"Loading existing BOW from {config.bow_dir}")
        if use_ddp:
            bow_matrix, vocab, vocab_embeddings, _ = run_all_ranks(
                lambda: load_bow_artifacts(config.bow_dir),
                use_ddp,
                "load BOW artifacts",
                signature_getter=lambda result: result[3],
                expected_signature=expected_bow_signature,
            )
        else:
            bow_matrix, vocab, vocab_embeddings, _ = load_bow_artifacts(
                config.bow_dir
            )
        logger.info(f"Loaded BOW: shape={bow_matrix.shape}, vocab_size={len(vocab)}")
        # Convert large dense BOW to sparse to save memory during training
        if not sparse.issparse(bow_matrix) and bow_matrix.shape[0] > 200000:
            logger.info(f"Converting BOW to sparse format to save memory ({bow_matrix.nbytes / 1e9:.1f} GB dense)")
            bow_matrix = sparse.csr_matrix(bow_matrix)
            logger.info(f"Sparse BOW: nnz={bow_matrix.nnz:,}, density={bow_matrix.nnz / (bow_matrix.shape[0]*bow_matrix.shape[1]):.4f}")
    else:
        # Generate BOW (first time for this dataset)
        def generate_bow_artifacts():
            logger.info(f"Generating new BOW for dataset {config.data.dataset}")
            generated_bow, generated_vocab = generate_bow(texts, config, logger)
            generated_embeddings = generate_vocab_embeddings(
                generated_vocab,
                config,
                logger,
            )
            signature = save_bow_artifacts(
                generated_bow,
                generated_vocab,
                generated_embeddings,
                config.bow_dir,
            )
            logger.info(f"BOW saved to {config.bow_dir} (shared across modes)")
            return signature

        if use_ddp:
            expected_bow_signature = run_rank_zero(
                generate_bow_artifacts,
                use_ddp,
                local_rank,
                "generate BOW artifacts",
            )
            bow_matrix, vocab, vocab_embeddings, _ = run_all_ranks(
                lambda: load_bow_artifacts(config.bow_dir),
                use_ddp,
                "load BOW artifacts",
                signature_getter=lambda result: result[3],
                expected_signature=expected_bow_signature,
            )
        else:
            logger.info(f"Generating new BOW for dataset {config.data.dataset}")
            bow_matrix, vocab = generate_bow(texts, config, logger)
            vocab_embeddings = generate_vocab_embeddings(vocab, config, logger)
            save_bow_artifacts(
                bow_matrix,
                vocab,
                vocab_embeddings,
                config.bow_dir,
            )
            logger.info(f"BOW saved to {config.bow_dir} (shared across modes)")

    # Load document embeddings
    doc_embeddings, labels = load_doc_embeddings(config, logger)

    # Auto-detect doc_embedding_dim from actual data
    actual_dim = doc_embeddings.shape[1]
    if config.model.doc_embedding_dim != actual_dim:
        logger.info(f"Auto-adjusting doc_embedding_dim: {config.model.doc_embedding_dim} -> {actual_dim}")
        config.model.doc_embedding_dim = actual_dim

    # Train model
    results = train_etm(doc_embeddings, bow_matrix, vocab_embeddings, config, logger, device, local_rank, world_size)

    # Save results (including timestamps if available)
    timestamp = run_rank_zero(
        lambda: save_results(
            results['model'], results['history'], vocab,
            doc_embeddings, bow_matrix, vocab_embeddings,
            config, logger, device,
            timestamps=timestamps
        ),
        use_ddp,
        local_rank,
        "save training results",
    )

    return timestamp


def run_pipeline(config: PipelineConfig, logger: logging.Logger,
                 skip_viz: bool = False, skip_eval: bool = False,
                 local_rank: int = -1, world_size: int = 1):
    """Run full pipeline: train + evaluate + visualize"""
    use_ddp = local_rank >= 0 and world_size > 1
    logger.info("=" * 60)
    logger.info("Running full ETM pipeline")
    logger.info(f"Dataset: {config.data.dataset}, Mode: {config.embedding.mode}")
    logger.info("=" * 60)

    # Train
    timestamp = run_train(config, logger, local_rank, world_size)

    if not skip_eval:
        def evaluate_training_results() -> None:
            run_evaluation(config, logger, timestamp)
            return None

        run_rank_zero(
            evaluate_training_results,
            use_ddp,
            local_rank,
            "evaluate training results",
        )
    elif is_main_process(local_rank):
        logger.info("[SKIP] Evaluation skipped")

    if not skip_viz:
        def visualize_training_results() -> None:
            run_visualization(config, logger, timestamp)
            return None

        run_rank_zero(
            visualize_training_results,
            use_ddp,
            local_rank,
            "visualize training results",
        )
    elif is_main_process(local_rank):
        logger.info("[SKIP] Visualization skipped")

    if is_main_process(local_rank):
        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Results: {config.result_dir}")
        logger.info(f"  - Embeddings: {config.embeddings_dir}")
        logger.info(f"  - BOW: {config.bow_dir}")
        logger.info(f"  - Model: {config.model_dir}")
        logger.info(f"  - Evaluation: {config.evaluation_dir}")
        logger.info(f"  - Visualization: {config.visualization_dir}")
        logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = create_parser()

    # Add DDP arguments
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of GPUs for distributed training')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    config = config_from_args(args)
    is_distributed = local_rank >= 0 and world_size > 1

    if is_distributed:
        setup_ddp(local_rank, world_size)
    else:
        configure_cuda_visible_devices(
            getattr(args, "gpu", None),
            default_gpu=config.gpu_id,
        )

    # Setup logging (only on main process for DDP)
    logger = setup_logging(config, local_rank, is_distributed)

    try:
        if args.command == "train":
            run_train(config, logger, local_rank, world_size)
        elif args.command == "evaluate":
            def evaluate_existing_results() -> None:
                run_evaluation(
                    config,
                    logger,
                    getattr(args, 'timestamp', None),
                )
                return None

            run_rank_zero(
                evaluate_existing_results,
                is_distributed,
                local_rank,
                "evaluate existing results",
            )
        elif args.command == "visualize":
            def visualize_existing_results() -> None:
                use_wordcloud = not getattr(args, 'no_wordcloud', False)
                run_visualization(
                    config,
                    logger,
                    getattr(args, 'timestamp', None),
                    use_wordcloud,
                )
                return None

            run_rank_zero(
                visualize_existing_results,
                is_distributed,
                local_rank,
                "visualize existing results",
            )
        elif args.command == "pipeline":
            skip_viz = getattr(args, 'skip_viz', False)
            skip_eval = getattr(args, 'skip_eval', False)
            run_pipeline(
                config, logger,
                skip_viz=skip_viz,
                skip_eval=skip_eval,
                local_rank=local_rank,
                world_size=world_size,
            )
        elif args.command == "clean":
            # Data cleaning
            from dataclean.main import main as clean_main
            sys.argv = ['main.py', 'convert', args.input, args.output, '--language', args.language]
            clean_main()
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        # Cleanup DDP
        cleanup_ddp()


if __name__ == "__main__":
    main()
