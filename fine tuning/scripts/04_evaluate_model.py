#!/usr/bin/env python3
"""
=============================================================================
SCRIPT NAME: 04_evaluate_model.py
=============================================================================

INPUT FILES:
- data/processed/val.jsonl: Validation examples
- data/processed/test.jsonl: Test examples

OUTPUT FILES:
- outputs/evaluation_results.xlsx: Detailed evaluation metrics
- outputs/evaluation_results.json: JSON format results

VERSION: 1.0
LAST UPDATED: 2026-01-29

DESCRIPTION:
Post-training evaluation script for fine-tuned ETF classifier.
Runs inference on val/test sets and computes classification accuracy.
Checks for overfitting by comparing performance across splits.

DEPENDENCIES:
- tinker, pandas, numpy, openpyxl

USAGE:
python scripts/04_evaluate_model.py --model etf-classifier-v2
=============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()

import tinker
from tinker import types


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_expected_output(conversation: Dict) -> Dict:
    """Extract the expected classification from the assistant message."""
    for msg in conversation['messages']:
        if msg['role'] == 'assistant':
            try:
                return json.loads(msg['content'])
            except json.JSONDecodeError:
                return {'raw': msg['content']}
    return {}


def extract_input_info(conversation: Dict) -> Dict:
    """Extract ticker and name from user message."""
    for msg in conversation['messages']:
        if msg['role'] == 'user':
            content = msg['content']
            info = {}
            for line in content.split('\n'):
                if line.startswith('Ticker:'):
                    info['ticker'] = line.replace('Ticker:', '').strip()
                elif line.startswith('Name:'):
                    info['name'] = line.replace('Name:', '').strip()
            return info
    return {}


def build_prompt(conversation: Dict) -> str:
    """Build the prompt for inference (without assistant response)."""
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


def parse_model_output(output: str) -> Dict:
    """Parse the model's JSON output."""
    try:
        # Try to extract JSON from the output
        output = output.strip()
        if output.startswith('{'):
            # Find the end of JSON
            brace_count = 0
            end_idx = 0
            for i, c in enumerate(output):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            json_str = output[:end_idx]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    return {'raw': output, 'parse_error': True}


def evaluate_classification(expected: Dict, predicted: Dict) -> Dict:
    """Compare expected vs predicted classification."""
    results = {
        'tier1_match': False,
        'tier2_match': False,
        'tier3_overlap': 0.0,
        'exact_match': False,
    }

    if 'parse_error' in predicted:
        return results

    # Tier 1 match
    if expected.get('tier1') == predicted.get('tier1'):
        results['tier1_match'] = True

    # Tier 2 match
    if expected.get('tier2') == predicted.get('tier2'):
        results['tier2_match'] = True

    # Tier 3 tag overlap (Jaccard similarity)
    expected_tags = set(expected.get('tier3_tags', []))
    predicted_tags = set(predicted.get('tier3_tags', []))
    if expected_tags or predicted_tags:
        intersection = expected_tags & predicted_tags
        union = expected_tags | predicted_tags
        results['tier3_overlap'] = len(intersection) / len(union) if union else 0.0

    # Exact match (all tiers match)
    results['exact_match'] = results['tier1_match'] and results['tier2_match']

    return results


def run_evaluation(model_name: str, data: List[Dict], split_name: str,
                   sampling_client, tokenizer) -> Tuple[List[Dict], Dict]:
    """Run evaluation on a dataset split."""
    print(f"\n📊 Evaluating on {split_name} set ({len(data)} examples)...")

    results = []
    metrics = defaultdict(list)

    for i, conv in enumerate(data):
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(data)}...")

        try:
            # Build prompt
            prompt = build_prompt(conv)
            
            # Tokenize and get model prediction
            prompt_input = tokenize_prompt(prompt, tokenizer)
            params = types.SamplingParams(max_tokens=256, temperature=0.0)
            future = sampling_client.sample(prompt=prompt_input, sampling_params=params, num_samples=1)
            result = future.result()
            
            # Decode output
            output_tokens = result.sequences[0].tokens
            predicted_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

            # Parse outputs
            expected = extract_expected_output(conv)
            predicted = parse_model_output(predicted_text)
            input_info = extract_input_info(conv)

            # Evaluate
            eval_result = evaluate_classification(expected, predicted)

            # Store result
            result = {
                'index': i,
                'ticker': input_info.get('ticker', ''),
                'name': input_info.get('name', ''),
                'expected_tier1': expected.get('tier1', ''),
                'expected_tier2': expected.get('tier2', ''),
                'expected_tier3': ', '.join(expected.get('tier3_tags', [])),
                'predicted_tier1': predicted.get('tier1', ''),
                'predicted_tier2': predicted.get('tier2', ''),
                'predicted_tier3': ', '.join(predicted.get('tier3_tags', [])) if isinstance(predicted.get('tier3_tags'), list) else '',
                'tier1_match': eval_result['tier1_match'],
                'tier2_match': eval_result['tier2_match'],
                'tier3_overlap': eval_result['tier3_overlap'],
                'exact_match': eval_result['exact_match'],
                'parse_error': 'parse_error' in predicted,
            }
            results.append(result)

            # Aggregate metrics
            metrics['tier1_match'].append(eval_result['tier1_match'])
            metrics['tier2_match'].append(eval_result['tier2_match'])
            metrics['tier3_overlap'].append(eval_result['tier3_overlap'])
            metrics['exact_match'].append(eval_result['exact_match'])
            metrics['parse_error'].append('parse_error' in predicted)

        except Exception as e:
            print(f"   ⚠️  Error on example {i}: {e}")
            results.append({
                'index': i,
                'error': str(e),
            })
            continue

    # Calculate summary metrics
    summary = {
        'split': split_name,
        'total_examples': len(data),
        'evaluated': len([r for r in results if 'error' not in r]),
        'tier1_accuracy': np.mean(metrics['tier1_match']) if metrics['tier1_match'] else 0,
        'tier2_accuracy': np.mean(metrics['tier2_match']) if metrics['tier2_match'] else 0,
        'tier3_avg_overlap': np.mean(metrics['tier3_overlap']) if metrics['tier3_overlap'] else 0,
        'exact_match_rate': np.mean(metrics['exact_match']) if metrics['exact_match'] else 0,
        'parse_error_rate': np.mean(metrics['parse_error']) if metrics['parse_error'] else 0,
    }

    print(f"\n   📈 {split_name.upper()} Results:")
    print(f"      Tier-1 Accuracy: {summary['tier1_accuracy']*100:.1f}%")
    print(f"      Tier-2 Accuracy: {summary['tier2_accuracy']*100:.1f}%")
    print(f"      Tier-3 Avg Overlap: {summary['tier3_avg_overlap']*100:.1f}%")
    print(f"      Exact Match Rate: {summary['exact_match_rate']*100:.1f}%")
    print(f"      Parse Error Rate: {summary['parse_error_rate']*100:.1f}%")

    return results, summary


def tokenize_prompt(prompt: str, tokenizer) -> types.ModelInput:
    """Tokenize prompt for Tinker API."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return types.ModelInput.from_ints(tokens)


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned ETF classifier')
    parser.add_argument('--model', type=str, required=True,
                       help='Name of the fine-tuned model to evaluate')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing val.jsonl and test.jsonl')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save evaluation results')
    parser.add_argument('--splits', type=str, default='val,test',
                       help='Comma-separated list of splits to evaluate (val,test)')

    args = parser.parse_args()

    if not os.getenv('TINKER_API_KEY'):
        print("❌ TINKER_API_KEY not set")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    splits_to_eval = [s.strip() for s in args.splits.split(',')]

    data = {}
    for split in splits_to_eval:
        file_path = data_dir / f'{split}.jsonl'
        if file_path.exists():
            data[split] = load_jsonl(str(file_path))
            print(f"✅ Loaded {len(data[split])} examples from {split}.jsonl")
        else:
            print(f"⚠️  {file_path} not found, skipping {split} split")

    if not data:
        print("❌ No data to evaluate")
        return 1

    # Connect to Tinker and load model
    print(f"\n🔌 Connecting to Tinker...")
    service_client = tinker.ServiceClient()

    print(f"📥 Loading model: {args.model}")
    sampling_client = service_client.create_sampling_client(args.model)
    print("✅ Model loaded")
    
    # Load tokenizer (Tinker's native Llama-3 tokenizer)
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("thinkingmachineslabinc/meta-llama-3-tokenizer")
    print("✅ Tokenizer loaded")

    # Run evaluations
    all_results = {}
    all_summaries = {}

    for split_name, split_data in data.items():
        results, summary = run_evaluation(
            args.model, split_data, split_name, sampling_client, tokenizer
        )
        all_results[split_name] = results
        all_summaries[split_name] = summary

    # Check for overfitting
    print(f"\n{'='*70}")
    print("🔍 OVERFITTING ANALYSIS")
    print(f"{'='*70}")

    if 'val' in all_summaries and 'test' in all_summaries:
        val_acc = all_summaries['val']['tier1_accuracy']
        test_acc = all_summaries['test']['tier1_accuracy']
        gap = val_acc - test_acc

        print(f"   Val Tier-1 Accuracy:  {val_acc*100:.1f}%")
        print(f"   Test Tier-1 Accuracy: {test_acc*100:.1f}%")
        print(f"   Gap: {gap*100:.1f}%")

        if gap > 0.05:  # >5% gap
            print("   ⚠️  WARNING: Significant gap between val and test - possible overfitting!")
        elif gap > 0.02:  # >2% gap
            print("   ⚠️  Note: Small gap between val and test")
        else:
            print("   ✅ Good generalization (val ≈ test)")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save to Excel
    excel_path = output_dir / f'evaluation_results_{timestamp}.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_df = pd.DataFrame(all_summaries).T
        summary_df.to_excel(writer, sheet_name='Summary')

        # Detail sheets for each split
        for split_name, results in all_results.items():
            df = pd.DataFrame(results)
            df.to_excel(writer, sheet_name=f'{split_name}_details', index=False)

    print(f"\n✅ Results saved to: {excel_path}")

    # Save to JSON
    json_path = output_dir / f'evaluation_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'model': args.model,
            'timestamp': timestamp,
            'summaries': all_summaries,
            'results': {k: v[:10] for k, v in all_results.items()},  # First 10 only for JSON
        }, f, indent=2, default=str)
    print(f"✅ Summary saved to: {json_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    for split_name, summary in all_summaries.items():
        print(f"\n{split_name.upper()}:")
        print(f"   Tier-1 Accuracy: {summary['tier1_accuracy']*100:.1f}%")
        print(f"   Tier-2 Accuracy: {summary['tier2_accuracy']*100:.1f}%")
        print(f"   Exact Match: {summary['exact_match_rate']*100:.1f}%")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
