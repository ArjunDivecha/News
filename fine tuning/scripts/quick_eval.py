#!/usr/bin/env python3
"""Quick evaluation of the trained model on validation set."""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

load_dotenv()

import tinker
from tinker import types
from transformers import AutoTokenizer


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(conversation):
    """Build prompt for inference."""
    messages = conversation['messages']
    system_msg = next((m for m in messages if m['role'] == 'system'), None)
    user_msgs = [m for m in messages if m['role'] == 'user']
    
    prompt_parts = []
    if system_msg:
        prompt_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg['content']}<|eot_id|>")
    for user_msg in user_msgs:
        prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg['content']}<|eot_id|>")
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(prompt_parts)


def extract_expected(conv):
    for msg in conv['messages']:
        if msg['role'] == 'assistant':
            try:
                return json.loads(msg['content'])
            except:
                return None
    return None


def parse_output(text):
    try:
        text = text.strip()
        if text.startswith('{'):
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(text):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            return json.loads(text[:end_idx])
    except:
        pass
    return None


def main():
    # Final checkpoint from training (use sampler_weights path)
    model_path = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"
    val_path = "data/processed/val.jsonl"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    print("="*70)
    print("🧪 VALIDATION EVALUATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Val set: {val_path}")
    
    # Load validation data
    val_data = load_jsonl(val_path)
    print(f"Samples: {len(val_data)}")
    print()
    
    # Load tokenizer (using compatible public model)
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    
    # Connect to Tinker
    print("🔌 Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    print("✅ Model loaded\n")
    
    # Evaluate
    tier1_correct = 0
    tier2_correct = 0
    exact_match = 0
    errors = 0
    
    print("🔄 Running evaluation...")
    for i, conv in enumerate(val_data):
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(val_data)}...")
        
        prompt_text = build_prompt(conv)
        expected = extract_expected(conv)
        
        try:
            # Tokenize prompt
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt = types.ModelInput.from_ints(prompt_tokens)
            
            # Sample with parameters
            params = types.SamplingParams(max_tokens=256, temperature=0.0)
            future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
            result = future.result()
            
            # Decode output
            output_tokens = result.samples[0].tokens
            predicted_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            predicted = parse_output(predicted_text)
            
            if predicted and expected:
                if predicted.get('tier1') == expected.get('tier1'):
                    tier1_correct += 1
                if predicted.get('tier2') == expected.get('tier2'):
                    tier2_correct += 1
                if (predicted.get('tier1') == expected.get('tier1') and 
                    predicted.get('tier2') == expected.get('tier2')):
                    exact_match += 1
            else:
                errors += 1
                
        except Exception as e:
            errors += 1
            if i < 5:  # Print first few errors
                print(f"   Error on {i}: {e}")
    
    # Results
    n = len(val_data)
    print()
    print("="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    print(f"Total samples: {n}")
    print(f"Errors: {errors}")
    print()
    print(f"Tier-1 Accuracy: {tier1_correct/n*100:.1f}%")
    print(f"Tier-2 Accuracy: {tier2_correct/n*100:.1f}%")
    print(f"Exact Match:     {exact_match/n*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()
