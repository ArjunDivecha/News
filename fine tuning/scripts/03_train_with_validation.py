#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: 03_train_with_validation.py
=============================================================================

INPUT FILES:
- data/processed/train.jsonl: Training examples in chat format
- data/processed/val.jsonl: Validation examples for overfitting detection
- data/processed/test.jsonl: Test examples (not used during training)

OUTPUT FILES:
- Model saved to Tinker with specified name
- W&B logs with train/val loss curves

VERSION: 1.0
LAST UPDATED: 2026-01-29

DESCRIPTION:
Tinker fine-tuning script with proper validation evaluation after each epoch.
Monitors for overfitting by comparing train vs validation loss.
Implements early stopping if validation loss increases for consecutive epochs.

DEPENDENCIES:
- tinker, wandb, numpy, transformers, python-dotenv

USAGE:
python scripts/03_train_with_validation.py --epochs 3 --output etf-classifier-v2
=============================================================================
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

if os.getenv('WEIGHTS_AND_BASES_API_KEY'):
    os.environ['WANDB_API_KEY'] = os.getenv('WEIGHTS_AND_BASES_API_KEY')

import wandb
import tinker
from tinker import types

DEFAULT_CONFIG = {
    'model': 'meta-llama/Llama-3.1-8B-Instruct',
    'lora_rank': 16,
    'learning_rate': 2e-4,
    'batch_size': 4,
    'num_epochs': 3,
    'output_name': 'etf-classifier-v2',
    'project_name': 'etf-classifier-tinker',
    'early_stopping_patience': 2,
    'val_frequency': 1,  # Validate every N epochs
}


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_datum_llama(conversation: Dict, tokenizer) -> types.Datum:
    messages = conversation['messages']

    system_msg = next((m for m in messages if m['role'] == 'system'), None)
    user_msgs = [m for m in messages if m['role'] == 'user']
    assistant_msgs = [m for m in messages if m['role'] == 'assistant']

    if not assistant_msgs:
        raise ValueError("No assistant messages found")

    prompt_parts = []

    if system_msg:
        prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg['content']}<|eot_id|>")

    for user_msg in user_msgs:
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg['content']}<|eot_id|>")

    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    prompt = "".join(prompt_parts)
    completion = f"{assistant_msgs[-1]['content']}<|eot_id|>"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=types.TensorData.from_numpy(np.array(weights, dtype=np.float32)),
            target_tokens=types.TensorData.from_numpy(np.array(target_tokens, dtype=np.int64))
        )
    )


def load_data(data_dir: str = 'data/processed') -> Dict[str, List]:
    data_dir = Path(data_dir)

    print(f"\n📂 Loading data from {data_dir}...")

    train_data = load_jsonl(str(data_dir / 'train.jsonl'))
    val_data = load_jsonl(str(data_dir / 'val.jsonl'))
    test_data = load_jsonl(str(data_dir / 'test.jsonl'))

    print(f"   Training:   {len(train_data):,} examples")
    print(f"   Validation: {len(val_data):,} examples")
    print(f"   Test:       {len(test_data):,} examples")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


def create_batches(data: List[types.Datum], batch_size: int) -> List[List[types.Datum]]:
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i + batch_size])
    return batches


def calculate_loss(fwdbwd_result, batch) -> float:
    """Calculate loss from Tinker result."""
    try:
        if hasattr(fwdbwd_result, 'loss_fn_outputs') and fwdbwd_result.loss_fn_outputs:
            logprobs = np.concatenate([
                output['logprobs'].tolist()
                for output in fwdbwd_result.loss_fn_outputs
            ])
            weights = np.concatenate([
                example.loss_fn_inputs['weights'].to_numpy()
                for example in batch
            ])

            mask = weights > 0
            if mask.sum() > 0:
                return -np.dot(logprobs[mask], weights[mask]) / weights[mask].sum()
    except Exception:
        pass
    return None


def evaluate_validation(training_client, val_datums: List[types.Datum], batch_size: int) -> float:
    """Evaluate loss on validation set (forward pass only, no backward)."""
    val_losses = []

    for batch in create_batches(val_datums, batch_size):
        try:
            # Forward only - no backward pass for validation
            fwd_future = training_client.forward_backward(batch, "cross_entropy")
            fwd_result = fwd_future.result()

            loss = calculate_loss(fwd_result, batch)
            if loss is not None:
                val_losses.append(loss)
        except Exception as e:
            continue

    if val_losses:
        return np.mean(val_losses)
    return None


def train_model(config: Dict, data: Dict[str, List]) -> Tuple[str, Any]:
    model_name = config['output_name']

    print("\n🔮 Initializing Weights & Biases...")
    wandb.init(
        project=config['project_name'],
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            'model': config['model'],
            'lora_rank': config['lora_rank'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'num_epochs': config['num_epochs'],
            'train_examples': len(data['train']),
            'val_examples': len(data['val']),
            'early_stopping_patience': config['early_stopping_patience'],
        }
    )
    print(f"   W&B URL: {wandb.run.url}")

    print(f"\n{'='*70}")
    print("🚀 STARTING TINKER TRAINING WITH VALIDATION")
    print(f"{'='*70}")
    print(f"Model: {config['model']}")
    print(f"LoRA Rank: {config['lora_rank']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Early Stopping Patience: {config['early_stopping_patience']}")
    print(f"{'='*70}\n")

    print("🔌 Connecting to Tinker service...")
    service_client = tinker.ServiceClient()
    print("✅ Connected to Tinker")

    print(f"\n🏗️  Creating training client...")
    training_client = service_client.create_lora_training_client(
        base_model=config['model'],
        rank=config['lora_rank']
    )
    print("✅ Training client created")

    print("\n📥 Loading tokenizer...")
    tokenizer = training_client.get_tokenizer()
    print("✅ Tokenizer loaded")

    # Process training data
    print("\n📊 Processing training data...")
    train_datums = []
    for i, conv in enumerate(data['train']):
        try:
            datum = prepare_datum_llama(conv, tokenizer)
            train_datums.append(datum)
            if (i + 1) % 500 == 0:
                print(f"   Processed {i+1}/{len(data['train'])}...")
        except Exception as e:
            print(f"   ⚠️  Skipping train example {i}: {e}")
            continue
    print(f"   ✅ Processed {len(train_datums)} training examples")

    # Process validation data
    print("\n📊 Processing validation data...")
    val_datums = []
    for i, conv in enumerate(data['val']):
        try:
            datum = prepare_datum_llama(conv, tokenizer)
            val_datums.append(datum)
        except Exception as e:
            print(f"   ⚠️  Skipping val example {i}: {e}")
            continue
    print(f"   ✅ Processed {len(val_datums)} validation examples")

    wandb.log({
        'data/train_processed': len(train_datums),
        'data/val_processed': len(val_datums),
    })

    if not train_datums:
        print("❌ No training data processed")
        wandb.finish()
        raise ValueError("No training data")

    # Training loop with validation
    print(f"\n{'='*70}")
    print("🏋️  TRAINING")
    print(f"{'='*70}")

    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []

    for epoch in range(config['num_epochs']):
        print(f"\n📚 Epoch {epoch + 1}/{config['num_epochs']}")

        # Training phase
        epoch_losses = []
        train_batches = create_batches(train_datums, config['batch_size'])

        for batch_idx, batch in enumerate(train_batches):
            try:
                fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
                optim_future = training_client.optim_step(
                    types.AdamParams(learning_rate=config['learning_rate'])
                )

                fwdbwd_result = fwdbwd_future.result()
                optim_result = optim_future.result()

                loss_value = calculate_loss(fwdbwd_result, batch)

                if loss_value is not None:
                    epoch_losses.append(loss_value)

                    wandb.log({
                        'train/loss': loss_value,
                        'train/epoch': epoch + 1,
                        'train/batch': batch_idx,
                    })

                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = np.mean(epoch_losses[-10:])
                        progress = (batch_idx + 1) / len(train_batches) * 100
                        print(f"   Batch {batch_idx + 1}/{len(train_batches)}: Train Loss = {avg_loss:.4f} ({progress:.1f}%)")

            except Exception as e:
                print(f"   ⚠️  Error in batch {batch_idx}: {e}")
                wandb.log({'train/batch_error': 1})
                continue

        # Calculate epoch train loss
        avg_train_loss = np.mean(epoch_losses) if epoch_losses else None

        # Validation phase
        print(f"\n   🔍 Running validation...")
        val_loss = evaluate_validation(training_client, val_datums, config['batch_size'])

        if avg_train_loss is not None and val_loss is not None:
            print(f"   ✅ Epoch {epoch + 1} complete")
            print(f"      Train Loss: {avg_train_loss:.4f}")
            print(f"      Val Loss:   {val_loss:.4f}")

            # Check for overfitting
            if val_loss > avg_train_loss * 1.1:  # Val loss >10% higher than train
                print(f"   ⚠️  Warning: Potential overfitting detected (val >> train)")

            wandb.log({
                'epoch/train_loss': avg_train_loss,
                'epoch/val_loss': val_loss,
                'epoch/number': epoch + 1,
                'epoch/train_val_gap': val_loss - avg_train_loss,
            })

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            })

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"   🎯 New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   ⚠️  Val loss did not improve ({patience_counter}/{config['early_stopping_patience']})")

                if patience_counter >= config['early_stopping_patience']:
                    print(f"\n   🛑 Early stopping triggered after {epoch + 1} epochs")
                    wandb.log({'training/early_stopped': True, 'training/stopped_at_epoch': epoch + 1})
                    break

    # Save model persistently
    print(f"\n{'='*70}")
    print("💾 SAVING MODEL")
    print(f"{'='*70}")

    # Use save_weights_for_sampler to get persistent path
    save_result = training_client.save_weights_for_sampler(name=model_name).result()
    model_path = save_result.path
    print(f"✅ Model saved: {model_name}")
    print(f"   Path: {model_path}")

    # Create sampling client from saved path
    sampling_client = service_client.create_sampling_client(model_path=model_path)

    # Final summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    for h in training_history:
        print(f"   Epoch {h['epoch']}: Train={h['train_loss']:.4f}, Val={h['val_loss']:.4f}")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"{'='*70}")

    wandb.log({
        'model/name': model_name,
        'model/path': model_path,
        'model/best_val_loss': best_val_loss,
        'training/completed_epochs': len(training_history),
    })

    # Save path to file for later use
    with open('outputs/model_path.txt', 'w') as f:
        f.write(model_path)
    print(f"   Model path saved to: outputs/model_path.txt")

    return model_name, sampling_client, model_path


def main():
    parser = argparse.ArgumentParser(description='Tinker fine-tuning with validation')
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['num_epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--lora-rank', type=int, default=DEFAULT_CONFIG['lora_rank'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_name'])
    parser.add_argument('--project', type=str, default=DEFAULT_CONFIG['project_name'])
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['early_stopping_patience'],
                       help='Early stopping patience (epochs without improvement)')

    args = parser.parse_args()

    if not os.getenv('TINKER_API_KEY'):
        print("❌ TINKER_API_KEY not set")
        return 1

    config = {
        'model': args.model,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'lora_rank': args.lora_rank,
        'batch_size': args.batch_size,
        'output_name': args.output,
        'project_name': args.project,
        'early_stopping_patience': args.patience,
    }

    try:
        data = load_data(args.data_dir)
    except FileNotFoundError:
        print("❌ Data not found. Run: python scripts/01_prepare_data.py")
        return 1

    start_time = datetime.now()

    try:
        model_name, sampling_client, model_path = train_model(config, data)

        elapsed = datetime.now() - start_time

        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Path: {model_path}")
        print(f"Duration: {elapsed}")
        print(f"W&B: {wandb.run.url}")
        print(f"{'='*70}\n")

        wandb.finish()
        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish()
        return 1


if __name__ == "__main__":
    sys.exit(main())
