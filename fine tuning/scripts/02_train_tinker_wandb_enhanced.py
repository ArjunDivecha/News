#!/usr/bin/env python3
"""
TINKER FINE-TUNING FOR ETF CLASSIFICATION WITH W&B, VALIDATION & EARLY STOPPING
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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
    'output_name': 'etf-classifier-v1',
    'project_name': 'etf-classifier-tinker',
    'eval_every': 50,  # Evaluate every N batches
    'early_stopping_patience': 5,  # Stop if no improvement for N evals
    'min_delta': 0.001,  # Minimum improvement to count
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


def create_batches(data: List[types.Datum], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def calculate_loss(fwdbwd_result, batch):
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


def evaluate(training_client, val_datums: List[types.Datum], batch_size: int) -> Tuple[float, int]:
    """Run evaluation on validation set. Returns (avg_loss, num_batches)."""
    val_losses = []
    num_batches = 0
    
    print(f"\n   📊 Running validation on {len(val_datums)} examples...")
    
    for batch in create_batches(val_datums, batch_size):
        try:
            # For validation, we only do forward pass (no backward/optim)
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            fwdbwd_result = fwdbwd_future.result()
            
            loss_value = calculate_loss(fwdbwd_result, batch)
            if loss_value is not None:
                val_losses.append(loss_value)
            
            num_batches += 1
            
        except Exception as e:
            print(f"      ⚠️  Val batch error: {e}")
            continue
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"   ✅ Validation complete | Avg Loss: {avg_val_loss:.4f} ({len(val_losses)} batches)")
    
    return avg_val_loss, num_batches


def train_model(config: Dict, data: Dict[str, List]) -> Tuple[str, Any, Dict]:
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
            'eval_every': config['eval_every'],
            'early_stopping_patience': config['early_stopping_patience'],
        }
    )
    print(f"   W&B URL: {wandb.run.url}")
    
    print(f"\n{'='*70}")
    print("🚀 STARTING TINKER TRAINING")
    print(f"{'='*70}")
    print(f"Model: {config['model']}")
    print(f"LoRA Rank: {config['lora_rank']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Eval Every: {config['eval_every']} batches")
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
            print(f"   ⚠️  Skipping example {i}: {e}")
            continue
    
    print(f"   ✅ Processed {len(train_datums)} training examples")
    
    # Process validation data
    print("\n📊 Processing validation data...")
    val_datums = []
    for i, conv in enumerate(data['val']):
        try:
            datum = prepare_datum_llama(conv, tokenizer)
            val_datums.append(datum)
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(data['val'])}...")
        except Exception as e:
            print(f"   ⚠️  Skipping val example {i}: {e}")
            continue
    
    print(f"   ✅ Processed {len(val_datums)} validation examples")
    
    wandb.log({
        'data/train_processed': len(train_datums),
        'data/train_raw': len(data['train']),
        'data/val_processed': len(val_datums),
        'data/val_raw': len(data['val']),
    })
    
    if not train_datums:
        print("❌ No training data processed")
        wandb.finish()
        raise ValueError("No training data")
    
    print(f"\n{'='*70}")
    print("🏋️  TRAINING")
    print(f"{'='*70}")
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    total_steps = 0
    evals_without_improvement = 0
    
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': best_val_loss,
        'best_epoch': 0,
        'stopped_early': False,
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\n📚 Epoch {epoch + 1}/{config['num_epochs']}")
        
        epoch_losses = []
        num_batches = 0
        
        for batch_idx, batch in enumerate(create_batches(train_datums, config['batch_size'])):
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
                    training_history['train_losses'].append(loss_value)
                    
                    wandb.log({
                        'train/loss': loss_value,
                        'train/epoch': epoch + 1,
                        'train/batch': batch_idx,
                    })
                    
                    if (batch_idx + 1) % 10 == 0:
                        avg_loss = np.mean(epoch_losses[-10:])
                        progress = (batch_idx + 1) * config['batch_size'] / len(train_datums) * 100
                        print(f"   Batch {batch_idx + 1}: Loss = {avg_loss:.4f} ({progress:.1f}%)")
                else:
                    if (batch_idx + 1) % 10 == 0:
                        progress = (batch_idx + 1) * config['batch_size'] / len(train_datums) * 100
                        print(f"   Batch {batch_idx + 1}: ({progress:.1f}%)")
                
                num_batches += 1
                total_steps += 1
                
                # Run validation every N batches
                if val_datums and total_steps > 0 and total_steps % config['eval_every'] == 0:
                    val_loss, val_batches = evaluate(training_client, val_datums, config['batch_size'])
                    
                    wandb.log({
                        'val/loss': val_loss,
                        'val/batches': val_batches,
                        'val/step': total_steps,
                    })
                    
                    training_history['val_losses'].append({
                        'step': total_steps,
                        'epoch': epoch + 1,
                        'loss': val_loss
                    })
                    
                    # Check for improvement
                    if val_loss < (best_val_loss - config['min_delta']):
                        improvement = best_val_loss - val_loss
                        best_val_loss = val_loss
                        patience_counter = 0
                        training_history['best_val_loss'] = best_val_loss
                        training_history['best_epoch'] = epoch + 1
                        print(f"   🎉 New best validation loss! Improvement: {improvement:.4f}")
                        
                        # Save best model checkpoint
                        checkpoint_name = f"{model_name}-best"
                        training_client.save_weights_and_get_sampling_client(name=checkpoint_name)
                        print(f"   💾 Saved checkpoint: {checkpoint_name}")
                    else:
                        patience_counter += 1
                        print(f"   ⏳ No improvement ({patience_counter}/{config['early_stopping_patience']})")
                    
                    if patience_counter >= config['early_stopping_patience']:
                        print(f"\n   🛑 Early stopping triggered after {patience_counter} evaluations without improvement")
                        training_history['stopped_early'] = True
                        break
                
            except Exception as e:
                print(f"   ⚠️  Error in batch {batch_idx}: {e}")
                wandb.log({'train/batch_error': 1})
                continue
        
        # End of epoch validation (if not already evaluated recently)
        if val_datums and patience_counter < config['early_stopping_patience']:
            # Only run if we haven't evaluated in the last 20 batches
            if num_batches > 20:
                val_loss, val_batches = evaluate(training_client, val_datums, config['batch_size'])
                
                wandb.log({
                    'val/loss_epoch': val_loss,
                    'val/epoch': epoch + 1,
                })
        
        if epoch_losses:
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"   ✅ Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")
            wandb.log({
                'epoch/avg_loss': avg_epoch_loss,
                'epoch/number': epoch + 1,
            })
        
        if training_history['stopped_early']:
            break
    
    print(f"\n{'='*70}")
    print("💾 SAVING FINAL MODEL")
    print(f"{'='*70}")
    
    sampling_client = training_client.save_weights_and_get_sampling_client(name=model_name)
    print(f"✅ Model saved: {model_name}")
    
    wandb.log({
        'model/name': model_name,
        'model/path': sampling_client.model_path,
        'training/best_val_loss': training_history['best_val_loss'],
        'training/best_epoch': training_history['best_epoch'],
        'training/stopped_early': training_history['stopped_early'],
    })
    
    return model_name, sampling_client, training_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['num_epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--lora-rank', type=int, default=DEFAULT_CONFIG['lora_rank'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_name'])
    parser.add_argument('--project', type=str, default=DEFAULT_CONFIG['project_name'])
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--eval-every', type=int, default=DEFAULT_CONFIG['eval_every'])
    parser.add_argument('--patience', type=int, default=DEFAULT_CONFIG['early_stopping_patience'])
    
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
        'eval_every': args.eval_every,
        'early_stopping_patience': args.patience,
    }
    
    try:
        data = load_data(args.data_dir)
    except FileNotFoundError:
        print("❌ Data not found. Run: python scripts/01_prepare_data.py")
        return 1
    
    start_time = datetime.now()
    
    try:
        model_name, sampling_client, history = train_model(config, data)
        
        elapsed = datetime.now() - start_time
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Best Val Loss: {history['best_val_loss']:.4f} (Epoch {history['best_epoch']})")
        print(f"Stopped Early: {history['stopped_early']}")
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
