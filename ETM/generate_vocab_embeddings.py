#!/usr/bin/env python
"""
Generate vocabulary embeddings for ETM using Qwen model.

This script loads the global vocabulary and generates embeddings for each word
using the Qwen model. These embeddings are used as the semantic basis (rho)
for the ETM decoder.

Usage:
    python generate_vocab_embeddings.py --vocab_path /path/to/global_vocab.json
                                      --output_path /path/to/output/vocab_embeddings.npy
                                      --model_path /path/to/qwen_model
"""

import os
import sys
import argparse
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from engine_c.vocab_embedder import generate_vocab_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate vocabulary embeddings for ETM")
    parser.add_argument("--vocab_path", type=str, 
                        default="/root/autodl-tmp/ETM/outputs/engine_a/global_vocab.json",
                        help="Path to vocabulary JSON file")
    parser.add_argument("--output_path", type=str, 
                        default="/root/autodl-tmp/ETM/outputs/engine_c/vocab_embeddings.npy",
                        help="Path to save embeddings")
    parser.add_argument("--model_path", type=str, 
                        default="/root/autodl-tmp/qwen3_embedding_0.6B", 
                        help="Path to Qwen model")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for encoding")
    parser.add_argument("--no_normalize", action="store_true", 
                        help="Disable L2 normalization")
    parser.add_argument("--dev_mode", action="store_true", 
                        help="Print debug information")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    logger.info(f"Generating vocabulary embeddings from {args.vocab_path}")
    logger.info(f"Using Qwen model from {args.model_path}")
    logger.info(f"Output will be saved to {args.output_path}")
    
    # Generate embeddings
    try:
        embeddings = generate_vocab_embeddings(
            vocab_path=args.vocab_path,
            output_path=args.output_path,
            model_path=args.model_path,
            batch_size=args.batch_size,
            normalize=not args.no_normalize,
            dev_mode=args.dev_mode
        )
        
        logger.info(f"Successfully generated embeddings with shape {embeddings.shape}")
        logger.info(f"Embeddings saved to {args.output_path}")
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


if __name__ == "__main__":
    main()
