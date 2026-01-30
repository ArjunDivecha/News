# Fine-Tuning: Asset Classification Model

Fine-tuned Llama 3.1 8B model for financial asset taxonomy classification, replacing expensive Anthropic API calls with fast, private inference.

## Overview

| Metric | Haiku (Before) | Fine-Tuned (After) | Improvement |
|--------|----------------|-------------------|-------------|
| Cost | ~$50-100/run | $0 | 100% savings |
| Speed | ~60 min | ~22 min | 3x faster |
| Latency | ~2,500ms/asset | ~1,400ms/asset | 1.8x faster |
| Privacy | External API | Private | Full control |

## Model Details

- **Base Model**: Llama 3.1 8B
- **Training Platform**: Tinker (thinkingmachines.ai)
- **Training Data**: 4,436 examples from classified asset datasets
- **Fine-Tuned Model**: `arjun-fund-classifier-v1-r7j1`

## Directory Structure

```
fine tuning/
├── README.md               # This file
├── INFERENCE.md            # Detailed inference guide
├── IMPLEMENTATION_PLAN.md  # Original implementation plan
├── requirements.txt        # Python dependencies
├── .env                    # API keys (gitignored)
│
├── data/
│   └── processed/          # Training data JSONs
│       ├── sample_examples.json
│       └── statistics.json
│
├── scripts/
│   ├── 01_prepare_data.py       # Data preparation
│   ├── 02_train_tinker.py       # Tinker training (basic)
│   ├── 02_train_tinker_wandb.py # Training with W&B logging
│   ├── 03_train_with_validation.py
│   ├── 04_evaluate_model.py     # Model evaluation
│   ├── 05_train_proper.py       # Production training
│   ├── compare_models.py        # Haiku vs Fine-tuned comparison
│   ├── predict_funds.py         # Batch prediction
│   ├── predict_single.py        # Single asset prediction
│   ├── classify_final_1000.py   # Classify Final 1000 list
│   ├── quick_eval.py            # Quick evaluation
│   └── test_model.py            # Model testing
│
└── outputs/                # Classification outputs
    └── final_1000/         # Final 1000 classification results
```

## Quick Start

### 1. Setup Environment

```bash
cd "fine tuning"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy from .env or set directly
export TINKER_API_KEY="your-tinker-api-key"
```

### 3. Run Classification

```bash
# Classify Final 1000 Asset Master List
python scripts/classify_final_1000.py

# Compare with Haiku classifications
python scripts/compare_models.py

# Single asset prediction
python scripts/predict_single.py "AAPL"
```

## Training (If Needed)

### Prepare Data

```bash
python scripts/01_prepare_data.py
# Output: data/processed/train.jsonl (~3,800 examples)
```

### Train on Tinker

```bash
export TINKER_API_KEY="your-key"
python scripts/05_train_proper.py
# Duration: ~30-60 minutes
# Cost: ~$10-20
```

### Evaluate

```bash
python scripts/04_evaluate_model.py
```

## Inference Scripts

### classify_final_1000.py
Classifies the Final 1000 Asset Master List and compares with existing Haiku classifications.

**Input**: `Step 2 Data Processing - Final1000/Final 1000 Asset Master List.xlsx`  
**Outputs**:
- `outputs/final_1000/Final_1000_FineTuned_Classified.xlsx`
- `outputs/final_1000/classification_comparison_report.xlsx`

**Usage**:
```bash
python scripts/classify_final_1000.py           # Full run
python scripts/classify_final_1000.py --limit 50  # Test with 50
python scripts/classify_final_1000.py --demo    # Simulation mode
```

### compare_models.py
Head-to-head comparison between Haiku and fine-tuned model classifications.

### predict_funds.py
Batch prediction for fund lists.

### predict_single.py
Quick single-asset classification.

## Model Performance

Based on validation set (450 examples):

| Metric | Score |
|--------|-------|
| Tier-1 Accuracy | ~80% |
| Tier-2 Accuracy | ~65% |
| Tier-3 Accuracy | ~50% |

Agreement with Haiku on test set: ~80% Tier-1, ~70% Tier-2

## Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
tqdm>=4.65.0
tinker>=0.1.0
transformers>=4.30.0
anthropic>=0.20.0
```

## API Configuration

The `.env` file should contain:
```bash
TINKER_API_KEY=your_tinker_api_key
WEIGHTS_AND_BASES_API_KEY=your_wandb_key  # Optional
```

## Troubleshooting

**Tinker connection fails**  
- Verify API key: `echo $TINKER_API_KEY`
- Check network connectivity
- Ensure venv is activated

**Python 3.14 Pydantic warning**  
- This is a known compatibility warning, doesn't affect functionality

**Model path not found**  
- Use: `arjun-fund-classifier-v1-r7j1`
- Or check available models in Tinker dashboard

---

**Last Updated**: 2026-01-30  
**Model Version**: arjun-fund-classifier-v1-r7j1  
**Training Platform**: Tinker (thinkingmachines.ai)
