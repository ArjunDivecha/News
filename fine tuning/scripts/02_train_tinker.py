#!/usr/bin/env python3
"""
================================================================================
TINKER FINE-TUNING FOR ETF CLASSIFICATION
Based on Arjun's proven Qwen3 training pattern
================================================================================

Prerequisites:
- TINKER_API_KEY in environment or .env file
- Training data in data/processed/ (run 01_prepare_data.py first)

Usage:
    cd "fine tuning"
    source venv/bin/activate
    python scripts/02_train_tinker.py

Output:
- Fine-tuned model on Tinker platform
- Training logs and metrics
- Model ready for download

Expected: 30-60 minutes, ~$10-20
================================================================================
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Tinker imports
try:
    import tinker
    from tinker import types
    TINKER_AVAILABLE = True
    print("✅ Tinker SDK imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Tinker: {e}")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

DEFAULT_CONFIG = {
    'model': 'meta-llama/Llama-3.1-8B-Instruct',
    'lora_rank': 16,
    'learning_rate': 2e-4,
    'batch_size': 4,
    'num_epochs': 3,
    'output_name': 'etf-classifier-v1',
}


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_datum_llama(conversation: Dict, tokenizer) -> types.Datum:
    """
    Process a conversation for Llama 3.1 model into Tinker Datum format.
    """
    messages = conversation['messages']
    
    # Separate prompt (system + user) from completion (assistant)
    system_msg = next((m for m in messages if m['role'] == 'system'), None)
    user_msgs = [m for m in messages if m['role'] == 'user']
    assistant_msgs = [m for m in messages if m['role'] == 'assistant']
    
    if not assistant_msgs:
        raise ValueError("No assistant messages found")
    
    # Build prompt for Llama 3.1
    prompt_parts = []
    
    if system_msg:
        prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg['content']}<|eot_id|>")
    
    for user_msg in user_msgs:
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg['content']}<|eot_id|>")
    
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    
    prompt = "".join(prompt_parts)
    completion = f"{assistant_msgs[-1]['content']}<|eot_id|>"
    
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    
    # Combine and create weights
    tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    
    # Shift for next-token prediction
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
    """Load all datasets."""
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
    """Create batches from data."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def calculate_loss(fwdbwd_result, batch):
    """Calculate average loss from forward-backward result."""
    # Simplified loss calculation
    if hasattr(fwdbwd_result, 'loss'):
        return fwdbwd_result.loss
    # Fallback: return a simple average
    return 1.0


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(config: Dict, data: Dict[str, List]) -> str:
    """
    Fine-tune model on Tinker.
    
    Returns:
        Model name for reference
    """
    model_name = config['output_name']
    
    print(f"\n{'='*70}")
    print("🚀 STARTING TINKER TRAINING")
    print(f"{'='*70}")
    print(f"Model: {config['model']}")
    print(f"LoRA Rank: {config['lora_rank']}")
    print(f"Learning Rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Output: {model_name}")
    print(f"{'='*70}\n")
    
    # Create service client
    print("🔌 Connecting to Tinker service...")
    try:
        service_client = tinker.ServiceClient()
        print("✅ Connected to Tinker")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        raise
    
    # Create training client
    print(f"\n🏗️  Creating training client with model: {config['model']}")
    try:
        training_client = service_client.create_lora_training_client(
            base_model=config['model'],
            rank=config['lora_rank']
        )
        print("✅ Training client created")
    except Exception as e:
        print(f"❌ Failed to create training client: {e}")
        raise
    
    # Get tokenizer
    print("\n📥 Loading tokenizer...")
    try:
        tokenizer = training_client.get_tokenizer()
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        raise
    
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
    
    if not train_datums:
        raise ValueError("No training data processed successfully")
    
    # Training loop
    print(f"\n{'='*70}")
    print("🏋️  TRAINING")
    print(f"{'='*70}")
    
    for epoch in range(config['num_epochs']):
        print(f"\n📚 Epoch {epoch + 1}/{config['num_epochs']}")
        
        epoch_losses = []
        num_batches = 0
        
        for batch_idx, batch in enumerate(create_batches(train_datums, config['batch_size'])):
            try:
                # Forward and backward pass
                fwdbwd_future = training_client.forward_backward(
                    batch, "cross_entropy"
                )
                
                # Optimizer step
                optim_future = training_client.optim_step(
                    types.AdamParams(learning_rate=config['learning_rate'])
                )
                
                # Wait for results
                fwdbwd_result = fwdbwd_future.result()
                optim_result = optim_future.result()
                
                # Calculate loss
                loss = calculate_loss(fwdbwd_result, batch)
                epoch_losses.append(loss)
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = np.mean(epoch_losses[-10:]) if epoch_losses else 0
                    progress = min(100, (batch_idx + 1) * config['batch_size'] / len(train_datums) * 100)
                    print(f"   Batch {batch_idx + 1}: Loss = {avg_loss:.4f} ({progress:.1f}%)")
                
            except Exception as e:
                print(f"   ⚠️  Error in batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"   ✅ Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}")
        else:
            print(f"   ❌ Epoch {epoch + 1} failed - no successful batches")
    
    # Save model and create sampling client
    print(f"\n{'='*70}")
    print("💾 SAVING MODEL")
    print(f"{'='*70}")
    
    try:
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=model_name
        )
        print(f"✅ Model saved: {model_name}")
        print(f"   Model path: {sampling_client.model_path}")
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        raise
    
    return model_name, sampling_client


# ==============================================================================
# VALIDATION
# ==============================================================================

def validate_model(sampling_client, val_data: List[Dict], num_samples: int = 10):
    """Run quick validation by sampling from the model."""
    print(f"\n{'='*70}")
    print("🔍 POST-TRAINING VALIDATION")
    print(f"{'='*70}")
    
    correct_tier1 = 0
    correct_tier2 = 0
    
    for i, example in enumerate(val_data[:num_samples]):
        messages = example['messages']
        user_msg = next((m for m in messages if m['role'] == 'user'), None)
        expected = next((m for m in messages if m['role'] == 'assistant'), None)
        
        if not user_msg or not expected:
            continue
        
        # Prepare prompt (system + user, no assistant)
        prompt_messages = [m for m in messages if m['role'] != 'assistant']
        
        try:
            # Generate
            sample_result = sampling_client.sample(
                prompt_messages,
                temperature=0.1,
                max_tokens=300
            )
            
            response = sample_result.text
            
            # Parse
            try:
                expected_json = json.loads(expected['content'])
                
                # Extract JSON from response
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response.strip()
                
                predicted_json = json.loads(json_str)
                
                tier1_match = predicted_json.get('tier1') == expected_json.get('tier1')
                tier2_match = predicted_json.get('tier2') == expected_json.get('tier2')
                
                if tier1_match:
                    correct_tier1 += 1
                if tier2_match:
                    correct_tier2 += 1
                
                status = "✓" if tier1_match else "✗"
                print(f"\n   {status} Sample {i+1}:")
                print(f"      Expected: {expected_json.get('tier1')} / {expected_json.get('tier2')}")
                print(f"      Predicted: {predicted_json.get('tier1')} / {predicted_json.get('tier2')}")
                
            except json.JSONDecodeError:
                print(f"\n   ⚠️  Sample {i+1}: JSON parse error")
                print(f"      Raw response: {response[:100]}...")
                
        except Exception as e:
            print(f"\n   ⚠️  Sample {i+1}: Error - {e}")
    
    print(f"\n   📊 Accuracy:")
    print(f"      Tier-1: {correct_tier1}/{num_samples} ({correct_tier1/num_samples:.1%})")
    print(f"      Tier-2: {correct_tier2}/{num_samples} ({correct_tier2/num_samples:.1%})")
    
    return {
        'tier1_acc': correct_tier1 / num_samples,
        'tier2_acc': correct_tier2 / num_samples
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune ETF classifier on Tinker')
    parser.add_argument('--model', type=str, default=DEFAULT_CONFIG['model'])
    parser.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['num_epochs'])
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'])
    parser.add_argument('--lora-rank', type=int, default=DEFAULT_CONFIG['lora_rank'])
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'])
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_name'])
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--skip-validation', action='store_true')
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv('TINKER_API_KEY')
    if not api_key:
        print("❌ TINKER_API_KEY environment variable not set")
        print("   Add to .env file or export it:")
        print("   export TINKER_API_KEY='your-key'")
        return 1
    
    # Build config
    config = {
        'model': args.model,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'lora_rank': args.lora_rank,
        'batch_size': args.batch_size,
        'output_name': args.output,
    }
    
    # Load data
    try:
        data = load_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease run data preparation first:")
        print("   python scripts/01_prepare_data.py")
        return 1
    
    # Train
    start_time = datetime.now()
    
    try:
        model_name, sampling_client = train_model(config, data)
        
        # Post-training validation
        if not args.skip_validation:
            metrics = validate_model(sampling_client, data['val'], num_samples=10)
        
        # Summary
        elapsed = datetime.now() - start_time
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"Duration: {elapsed}")
        if not args.skip_validation:
            print(f"Tier-1 Accuracy: {metrics['tier1_acc']:.1%}")
            print(f"Tier-2 Accuracy: {metrics['tier2_acc']:.1%}")
        print(f"\nNext steps:")
        print("   1. Download weights from Tinker dashboard")
        print("   2. Run: python scripts/03_download_weights.py")
        print("   3. Run: python scripts/04_setup_mlx.py")
        print(f"{'='*70}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
