#!/usr/bin/env python3
"""Simple synchronous evaluation."""

import os
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tqdm import tqdm

load_dotenv()

import tinker
from tinker import types


def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(conv):
    messages = conv['messages']
    system_msg = next((m for m in messages if m['role'] == 'system'), None)
    user_msgs = [m for m in messages if m['role'] == 'user']
    
    parts = []
    if system_msg:
        parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg['content']}<|eot_id|>")
    for user_msg in user_msgs:
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg['content']}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def extract_expected(conv):
    for msg in conv['messages']:
        if msg['role'] == 'assistant':
            try:
                return json.loads(msg['content'])
            except:
                return None
    return None


def extract_ticker(conv):
    for msg in conv['messages']:
        if msg['role'] == 'user':
            for line in msg['content'].split('\n'):
                if line.startswith('Ticker:'):
                    return line.replace('Ticker:', '').strip()
    return ''


def main():
    model_path = "tinker://3a0f33d5-93be-5c30-b756-c183909f1804:train:0/sampler_weights/final"
    
    print("="*70)
    print("🧪 VALIDATION EVALUATION")
    print("="*70)
    
    print("\n📥 Loading data...")
    val_data = load_jsonl("data/processed/val.jsonl")
    print(f"   {len(val_data)} examples")
    
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("thinkingmachineslabinc/meta-llama-3-tokenizer")
    
    print("🔌 Connecting to Tinker...")
    service = tinker.ServiceClient()
    client = service.create_sampling_client(model_path=model_path)
    print("✅ Ready!\n")
    
    # Evaluate
    tier1_correct = tier2_correct = exact_match = errors = 0
    results = []
    
    for conv in tqdm(val_data, desc="Evaluating"):
        prompt = build_prompt(conv)
        expected = extract_expected(conv)
        ticker = extract_ticker(conv)
        
        try:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_input = types.ModelInput.from_ints(tokens)
            params = types.SamplingParams(max_tokens=256, temperature=0.0)
            
            future = client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
            result = future.result()
            
            output_tokens = result.sequences[0].tokens
            predicted_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            
            try:
                predicted = json.loads(predicted_text)
                
                t1_match = predicted.get('tier1') == expected.get('tier1')
                t2_match = predicted.get('tier2') == expected.get('tier2')
                
                if t1_match:
                    tier1_correct += 1
                if t2_match:
                    tier2_correct += 1
                if t1_match and t2_match:
                    exact_match += 1
                    
                results.append({
                    'ticker': ticker,
                    't1_exp': expected.get('tier1'),
                    't1_pred': predicted.get('tier1'),
                    't1_ok': t1_match,
                    't2_exp': expected.get('tier2'),
                    't2_pred': predicted.get('tier2'),
                    't2_ok': t2_match,
                })
            except:
                errors += 1
                
        except Exception as e:
            errors += 1
    
    n = len(val_data)
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"Total: {n}")
    print(f"Errors: {errors}")
    print(f"Tier-1 Accuracy: {tier1_correct/n*100:.1f}%")
    print(f"Tier-2 Accuracy: {tier2_correct/n*100:.1f}%")
    print(f"Exact Match: {exact_match/n*100:.1f}%")
    print("="*70)
    
    # Show some errors
    print("\n❌ Some errors:")
    for r in results:
        if not r['t1_ok'] or not r['t2_ok']:
            print(f"   {r['ticker']}: T1={r['t1_exp']}→{r['t1_pred']} T2={r['t2_exp']}→{r['t2_pred']}")


if __name__ == "__main__":
    main()
