#!/usr/bin/env python3
"""Quick test of the trained model on a few examples."""

import os
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

import tinker
from tinker import types


def load_jsonl(file_path, n=5):
    """Load first n examples."""
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def build_prompt(conv):
    """Build prompt for inference."""
    messages = conv['messages']
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
                return msg['content']
    return None


def main():
    model_path = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"
    
    print("="*70)
    print("🧪 MODEL TEST (5 examples)")
    print("="*70)
    print(f"Model: {model_path}")
    
    # Load tokenizer (Tinker's native Llama-3 tokenizer)
    print("\n📥 Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("thinkingmachineslabinc/meta-llama-3-tokenizer")
    
    # Connect to Tinker
    print("🔌 Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    print("✅ Model loaded\n")
    
    # Load 5 examples
    examples = load_jsonl("data/processed/val.jsonl", n=5)
    print(f"Testing on {len(examples)} examples...\n")
    
    for i, conv in enumerate(examples):
        prompt = build_prompt(conv)
        expected = extract_expected(conv)
        
        print(f"--- Example {i+1} ---")
        
        # Show user input
        for msg in conv['messages']:
            if msg['role'] == 'user':
                for line in msg['content'].split('\n'):
                    if line.startswith('Ticker:'):
                        print(f"Ticker: {line.replace('Ticker:', '').strip()}")
                    if line.startswith('Name:'):
                        print(f"Name: {line.replace('Name:', '').strip()}")
        
        print(f"Expected: {json.dumps(expected, indent=2)}")
        
        try:
            # Tokenize and sample
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_input = types.ModelInput.from_ints(tokens)
            params = types.SamplingParams(max_tokens=256, temperature=0.0)
            
            future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
            result = future.result()
            
            output_tokens = result.sequences[0].tokens
            predicted_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            print(f"Predicted: {predicted_text[:200]}...")
            
            # Try to parse as JSON
            try:
                pred_json = json.loads(predicted_text.strip())
                if isinstance(pred_json, dict):
                    tier1_match = pred_json.get('tier1') == expected.get('tier1')
                    tier2_match = pred_json.get('tier2') == expected.get('tier2')
                    print(f"Tier-1 Match: {tier1_match}")
                    print(f"Tier-2 Match: {tier2_match}")
            except:
                print("(Could not parse as JSON)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


if __name__ == "__main__":
    main()
