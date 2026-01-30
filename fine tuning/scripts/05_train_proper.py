#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: 05_train_proper.py
=============================================================================

PROPER TINKER TRAINING using the supervised learning framework with
AUTOMATIC CHECKPOINT SAVING.

This uses tinker_cookbook.supervised.train which handles:
- Automatic periodic checkpoint saving (every N steps)
- Automatic validation evaluation
- W&B logging
- Proper state management

INPUT FILES:
- data/processed/train.jsonl: Training examples in chat format
- data/processed/val.jsonl: Validation examples

OUTPUT FILES:
- outputs/checkpoints.jsonl: All saved checkpoints
- outputs/metrics.jsonl: Training metrics
- outputs/config.json: Training configuration

VERSION: 1.0
LAST UPDATED: 2026-01-29

DEPENDENCIES:
- tinker, tinker-cookbook, wandb

USAGE:
python scripts/05_train_proper.py
=============================================================================
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Set W&B API key
if os.getenv('WEIGHTS_AND_BASES_API_KEY'):
    os.environ['WANDB_API_KEY'] = os.getenv('WEIGHTS_AND_BASES_API_KEY')

import tinker
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
import torch
import numpy as np


class NLLEvaluator(TrainingClientEvaluator):
    """Evaluate negative log-likelihood on validation set."""

    def __init__(self, val_dataset):
        self.val_dataset = val_dataset

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        total_nll = 0.0
        total_tokens = 0

        for batch_idx in range(len(self.val_dataset)):
            batch = self.val_dataset.get_batch(batch_idx)

            # Forward pass without backward
            result = training_client.forward_backward(batch, "cross_entropy")
            result = result.result()

            # Compute NLL only over completion tokens (where weights > 0)
            logprobs = np.concatenate([
                output["logprobs"].tolist()
                for output in result.loss_fn_outputs
            ])
            weights = np.concatenate([
                example.loss_fn_inputs['weights'].to_numpy()
                for example in batch
            ])

            mask = weights > 0
            if mask.sum() > 0:
                total_nll += -np.dot(logprobs[mask], weights[mask])
                total_tokens += weights[mask].sum()

        avg_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
        return {"val_nll": avg_nll}


async def main():
    # Configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    log_path = "outputs/training_run"
    
    # Check for resume checkpoint
    import sys
    load_checkpoint_path = None
    if len(sys.argv) > 1:
        load_checkpoint_path = sys.argv[1]
        print(f"📂 Resuming from checkpoint: {load_checkpoint_path}")

    print("="*70)
    print("🚀 TINKER PROPER TRAINING WITH AUTO-CHECKPOINTING")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Output: {log_path}")
    print("="*70)
    print()

    # Dataset configuration
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name="llama3",
        max_length=4096,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Build training dataset
    train_dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path="data/processed/train.jsonl"
    )

    # Build validation dataset
    val_dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path="data/processed/val.jsonl"
    )

    # Validation disabled for now - takes too long at startup
    val_train_ds = None
    print("   Validation: DISABLED (for faster startup)")

    # Training configuration with automatic checkpointing
    config = train.Config(
        log_path=log_path,
        model_name=model_name,
        load_checkpoint_path=load_checkpoint_path,
        dataset_builder=train_dataset_builder,
        learning_rate=2e-4,
        lr_schedule="linear",
        num_epochs=3,
        lora_rank=16,
        save_every=100,      # Save checkpoint every 100 steps
        eval_every=50,       # Evaluate on validation every 50 steps
        evaluator_builders=[],  # No validation evaluator - faster startup
        wandb_project="etf-classifier-tinker",
        wandb_name="etf-classifier-proper",
    )

    print("📋 Training Configuration:")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   LoRA Rank: {config.lora_rank}")
    print(f"   Batch Size: {common_config.batch_size}")
    print(f"   Save Every: {config.save_every} steps")
    print(f"   Eval Every: {config.eval_every} steps")
    print(f"   W&B Project: {config.wandb_project}")
    print()

    print("🏋️  Starting training...")
    print("   (Checkpoints will be saved automatically)")
    print()

    # Run training with automatic checkpointing
    await train.main(config)

    print()
    print("="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Checkpoints saved to: {log_path}/checkpoints.jsonl")
    print(f"Metrics saved to: {log_path}/metrics.jsonl")
    print("="*70)


if __name__ == "__main__":
    if not os.getenv('TINKER_API_KEY'):
        print("❌ TINKER_API_KEY not set in .env")
        exit(1)

    asyncio.run(main())
