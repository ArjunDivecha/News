# Fine-Tuning Implementation Plan

## Asset Classification Pipeline: Anthropic → Fine-Tuned Local Model

**Project:** News Data Collection Pipeline  
**Platform:** Tinker (Training) + MLX on M4 Max 128GB (Inference)  
**Model:** Llama 3.1 8B Instruct (fine-tuned)  
**Timeline:** 1-2 weeks  
**Status:** Ready to begin

---

## Executive Summary

This plan migrates the asset classification pipeline from **Anthropic Haiku API calls** to a **fine-tuned Llama 3.1 8B model** running locally on the M4 Max via MLX.

### Why This Migration?

| Metric | Current (Anthropic) | After Migration | Improvement |
|--------|---------------------|-----------------|-------------|
| **Cost per run** | ~$50-100 | ~$0 | 100% savings |
| **Speed** | ~60 minutes | ~2-5 minutes | 12-30x faster |
| **Latency/asset** | ~2-5 seconds | ~25-50ms | 40-200x faster |
| **Privacy** | Data to external API | Completely local | Full control |
| **Offline capable** | No | Yes | Anywhere operation |

### Hardware Advantages

- **M4 Max 128GB**: Can run 8B-70B models locally with room to spare
- **MLX framework**: 2-5x faster than PyTorch on Apple Silicon
- **Unified memory**: No GPU/CPU transfer bottlenecks

---

## Directory Structure

```
fine tuning/
├── IMPLEMENTATION_PLAN.md      # This file
├── README.md                    # Quick start guide (created after completion)
├── data/                        # Training/validation data
│   ├── raw/                     # Symlinks to classified files
│   ├── processed/               # JSONL format for Tinker
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── statistics.json
│   └── validation/              # Comparison results
├── scripts/                     # All automation scripts
│   ├── 01_prepare_data.py       # Data preparation
│   ├── 02_train_tinker.py       # Tinker fine-tuning
│   ├── 03_download_weights.py   # Weight download & conversion
│   ├── 04_setup_mlx.py          # MLX environment setup
│   ├── 05_validate.py           # Head-to-head comparison
│   └── 06_migrate_scripts.py    # Production migration helper
├── models/                      # Downloaded/converted models
│   ├── tinker_export/           # Raw Tinker output
│   └── mlx_ready/               # MLX-converted format
├── inference/                   # Local inference engine
│   ├── mlx_classifier.py        # Main classifier class
│   └── unified_classifier.py    # Drop-in replacement module
├── validation/                  # Validation & testing
│   └── compare_models.py        # Anthropic vs MLX comparison
└── notebooks/                   # Jupyter notebooks for exploration
    ├── data_exploration.ipynb
    └── error_analysis.ipynb
```

---

## Phase 1: Data Preparation (Days 1-2)

### Goal
Extract ~4,933 classified assets from existing Excel files and format for Tinker fine-tuning.

### Input Files (from parent directories)

| Source | File Path | Expected Count |
|--------|-----------|----------------|
| ETFs | `../Step 2 Data Processing - Final1000/ETF Master List Classified.xlsx` | ~1,619 |
| Bloomberg Indices | `../Step 2 Data Processing - Final1000/Filtered Bloomberg Indices Classified.xlsx` | ~438 |
| Goldman Baskets | `../Step 1 Data Collection/GSCB_FLAGSHIP_coverage_Classified.xlsx` | ~2,667 |
| Thematic ETFs | `../Step 2 Data Processing - Final1000/Thematic ETFs Classified.xlsx` | ~231 |

### Output

- `data/processed/train.jsonl` (~3,800 examples, 85%)
- `data/processed/val.jsonl` (~450 examples, 10%)
- `data/processed/test.jsonl` (~230 examples, 5%)
- `data/processed/statistics.json` (distribution analysis)

### Script

**File:** `scripts/01_prepare_data.py`

```bash
# Run from fine tuning/ directory
python scripts/01_prepare_data.py
```

### Data Format (JSONL)

Each line is a chat completion example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert financial asset classification specialist..."
    },
    {
      "role": "user",
      "content": "Classify this asset:\nTicker: VOO\nName: Vanguard S&P 500 ETF\nDescription: Invests in stocks in the S&P 500 Index..."
    },
    {
      "role": "assistant",
      "content": "{\"ticker\": \"VOO\", \"tier1\": \"Equities\", \"tier2\": \"Global Indices\", \"tier3_tags\": [\"Equity\", \"US\", \"Passive\", \"Large Cap\"]}"
    }
  ],
  "metadata": {
    "source": "ETF",
    "original_tier1": "Equities",
    "original_tier2": "Global Indices"
  }
}
```

### Success Criteria

- [ ] All 4 source files loaded successfully
- [ ] Stratified split by Tier-1 category
- [ ] No data leakage between train/val/test
- [ ] Sample examples inspected and validated

---

## Phase 2: Fine-Tuning on Tinker (Days 3-5)

### Goal
Train LoRA adapter on Tinker GPUs using the prepared dataset.

### Configuration

```python
{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "lora_rank": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "max_seq_length": 2048
}
```

### Script

**File:** `scripts/02_train_tinker.py`

```bash
# Set API key
export TINKER_API_KEY="your-key-here"

# Run training
python scripts/02_train_tinker.py
```

### Expected Output

- Checkpoints saved every 500 steps
- Final model: `etf-classifier-v1`
- Validation metrics after each epoch:
  - Tier-1 Accuracy
  - Tier-2 Accuracy
  - Loss curves

### Duration

- ~30-60 minutes on Tinker GPUs
- Cost: ~$10-20

### Success Criteria

- [ ] Training completes without errors
- [ ] Final Tier-1 accuracy ≥ 90% on validation
- [ ] Loss converges (stable/decreasing)
- [ ] Checkpoints downloadable

---

## Phase 3: Download & Convert Weights (Day 6)

### Goal
Get fine-tuned weights from Tinker and convert to MLX-compatible format.

### Script

**File:** `scripts/03_download_weights.py`

```bash
python scripts/03_download_weights.py --model etf-classifier-v1
```

### Conversion Options

Tinker may export in different formats. We'll handle:

1. **HuggingFace PEFT** → MLX via `mlx-lm` conversion
2. **PyTorch checkpoints** → MLX weight conversion
3. **Tinker native** → Custom conversion script

### Output

- `models/tinker_export/` - Raw downloaded weights
- `models/mlx_ready/` - MLX-optimized format

### Success Criteria

- [ ] Weights downloaded successfully
- [ ] MLX conversion completes
- [ ] Can load model with `mlx-lm`
- [ ] Test inference runs without errors

---

## Phase 4: MLX Inference Setup (Days 7-8)

### Goal
Build optimized inference engine for M4 Max.

### Script

**File:** `scripts/04_setup_mlx.py`

```bash
# Setup environment
python scripts/04_setup_mlx.py

# Activate
source mlx_env/bin/activate
```

### Inference Engine

**File:** `inference/mlx_classifier.py`

Features:
- MLX-optimized for Apple Silicon
- 4-bit quantization support
- Batch processing
- Benchmarking tools

### Test Usage

```python
from inference.mlx_classifier import MLXETFClassifier

# Load model (~10-20 seconds)
classifier = MLXETFClassifier(
    adapter_path='models/mlx_ready/etf-classifier-v1'
)

# Classify single asset
result = classifier.classify(
    ticker="VOO",
    name="Vanguard S&P 500 ETF",
    description="Invests in stocks in the S&P 500 Index..."
)
# ~25-50ms

# Batch classify
results = classifier.classify_batch(assets_list)
# ~20-40 assets/second
```

### Expected Performance (M4 Max)

| Batch Size | Throughput | Latency/asset |
|------------|------------|---------------|
| 1 | 20-40/sec | 25-50ms |
| 100 | 30-50/sec | 20-33ms |
| 1000 | 35-45/sec | 22-28ms |

### Success Criteria

- [ ] Model loads on M4 Max
- [ ] Single inference < 100ms
- [ ] Batch 5,000 assets in < 5 minutes
- [ ] Memory usage < 32GB

---

## Phase 5: Validation & Comparison (Days 9-11)

### Goal
Compare fine-tuned MLX model vs Anthropic Haiku on held-out test set.

### Script

**File:** `scripts/05_validate.py`

```bash
python scripts/05_validate.py \
    --test-data data/processed/test.jsonl \
    --samples 200 \
    --output validation/results/
```

### Comparison Metrics

| Metric | Anthropic (Baseline) | MLX (Target) |
|--------|---------------------|--------------|
| Tier-1 Accuracy | ~96% | ≥94% |
| Tier-2 Accuracy | ~89% | ≥85% |
| Avg Inference Time | ~2,500ms | ≤50ms |
| Cost per 1K assets | ~$10-20 | $0 |

### Minimum Acceptance Criteria

| Metric | Minimum | Target |
|--------|---------|--------|
| Tier-1 Accuracy | ≥90% | ≥95% |
| Tier-2 Accuracy | ≥80% | ≥85% |
| Speedup | ≥10x | ≥50x |
| Cost savings | 100% | 100% |

### If Validation Fails

**Option A: More Training**
- Increase epochs (3 → 5)
- Increase LoRA rank (16 → 32)
- Add more training data

**Option B: Model Swap**
- Try Qwen 2.5 7B instead of Llama 3.1 8B
- Some models work better for structured output

**Option C: Hybrid Approach**
- Use MLX for 90% of classifications
- Fallback to Anthropic for edge cases

### Success Criteria

- [ ] Tier-1 accuracy within 5% of Anthropic
- [ ] Tier-2 accuracy within 10% of Anthropic
- [ ] Speedup ≥ 50x
- [ ] Detailed error analysis completed

---

## Phase 6: Production Migration (Days 12-14)

### Goal
Replace Anthropic calls in all 6 classification scripts.

### Unified Classifier Module

**File:** `inference/unified_classifier.py`

Provides:
- Drop-in replacement for existing code
- Auto-detects best available backend
- Environment variable overrides
- Backward compatibility

### Migration Pattern

**Before (existing code):**
```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    messages=[...]
)
result = json.loads(response.content[0].text)
```

**After (new code):**
```python
from inference.unified_classifier import get_classifier

classifier = get_classifier()  # Auto-uses MLX
result = classifier.classify(
    ticker=ticker,
    name=name,
    description=description,
    ...
)
```

### Scripts to Update

1. `../Step 2 Data Processing - Final1000/etf_classifier.py`
2. `../Step 2 Data Processing - Final1000/classify_bloomberg_full.py`
3. `../Step 2 Data Processing - Final1000/classify_goldman_full.py`
4. `../Step 2 Data Processing - Final1000/classify_etfs_full.py`
5. `../Step 2 Data Processing - Final1000/classify_thematic_etfs_full.py`
6. `../Step 2 Data Processing - Final1000/unified_asset_classifier.py`

### Migration Helper

**File:** `scripts/06_migrate_scripts.py`

```bash
# Creates backup and applies patches
python scripts/06_migrate_scripts.py --dry-run
python scripts/06_migrate_scripts.py --apply
```

### Rollback Plan

If issues arise:
```bash
# Switch back to Anthropic
export CLASSIFIER_BACKEND=anthropic

# Or restore from backup
python scripts/06_migrate_scripts.py --rollback
```

### Success Criteria

- [ ] All 6 scripts updated
- [ ] Backward compatibility maintained
- [ ] Full pipeline runs successfully
- [ ] Documentation updated

---

## Daily Checklist

### Day 1: Data Prep Script
- [ ] Create `scripts/01_prepare_data.py`
- [ ] Test on sample data
- [ ] Verify JSONL output format

### Day 2: Run Data Prep
- [ ] Process all 4 source files
- [ ] Verify stratified split
- [ ] Inspect sample examples

### Day 3: Training Script
- [ ] Create `scripts/02_train_tinker.py`
- [ ] Test Tinker connection
- [ ] Submit training job

### Day 4-5: Monitor Training
- [ ] Track loss curves
- [ ] Verify validation accuracy
- [ ] Download final weights

### Day 6: Weight Conversion
- [ ] Download from Tinker
- [ ] Convert to MLX format
- [ ] Test load

### Day 7-8: MLX Setup
- [ ] Create inference engine
- [ ] Run benchmarks
- [ ] Optimize for M4 Max

### Day 9-10: Validation
- [ ] Run comparison script
- [ ] Analyze errors
- [ ] Document findings

### Day 11: Fix Issues (if needed)
- [ ] Retrain if accuracy low
- [ ] Tune hyperparameters
- [ ] Validate again

### Day 12-13: Migration
- [ ] Create unified classifier
- [ ] Update all 6 scripts
- [ ] Test full pipeline

### Day 14: Documentation
- [ ] Write README.md
- [ ] Document API
- [ ] Create troubleshooting guide

---

## Commands Quick Reference

```bash
# Setup
cd "fine tuning"
python scripts/01_prepare_data.py

# Training
export TINKER_API_KEY="..."
python scripts/02_train_tinker.py

# Download & Convert
python scripts/03_download_weights.py --model etf-classifier-v1

# MLX Setup
python scripts/04_setup_mlx.py
source mlx_env/bin/activate

# Validation
python scripts/05_validate.py --samples 200

# Migration
python scripts/06_migrate_scripts.py --apply

# Full Pipeline Test
cd ..
python "Step 2 Data Processing - Final1000/etf_classifier.py"
```

---

## Environment Variables

```bash
# Required
export TINKER_API_KEY="your-tinker-api-key"

# Optional (with defaults)
export CLASSIFIER_BACKEND="mlx"  # or "anthropic"
export MLX_ADAPTER_PATH="models/mlx_ready/etf-classifier-v1"
export CLASSIFIER_TEMPERATURE="0.1"
export CLASSIFIER_MAX_TOKENS="300"
```

---

## Troubleshooting

### Issue: Tinker training fails
**Solution:**
- Check API key
- Verify JSONL format
- Reduce batch size
- Check Tinker status page

### Issue: MLX model loads slowly
**Solution:**
- First load downloads base model (~16GB)
- Subsequent loads use cache
- Ensure SSD storage for models/

### Issue: Accuracy lower than expected
**Solution:**
- Check data quality (any mislabeled training examples?)
- Increase LoRA rank (16 → 32)
- Add more training epochs
- Try Qwen model instead of Llama

### Issue: Out of memory on M4
**Solution:**
- Enable 4-bit quantization (already default)
- Reduce max_seq_length
- Close other applications
- M4 Max 128GB should handle 8B easily

---

## Post-Implementation

### Monitoring
- Track classification accuracy over time
- Monitor inference speed
- Log API cost savings

### Future Improvements
- Fine-tune larger model (13B or 70B) if needed
- Add support for more asset types
- Implement active learning (retrain on errors)
- Quantize to 3-bit or 2-bit for even faster inference

### Maintenance
- Quarterly retraining with new data
- Monitor for model drift
- Update base model when new versions released

---

## Appendix

### A. Tinker Documentation
- Model lineup: https://tinker-docs.thinkingmachines.ai/model-lineup
- Fine-tuning guide: https://tinker-docs.thinkingmachines.ai/

### B. MLX Resources
- MLX docs: https://ml-explore.github.io/mlx/
- mlx-lm: https://github.com/ml-explore/mlx-examples/tree/main/llms

### C. Model Options

| Model | Size | Pros | Cons |
|-------|------|------|------|
| **Llama 3.1 8B** (recommended) | 8B | Excellent instruction following | Larger download |
| **Qwen 2.5 7B** | 7B | Great JSON output | Less proven |
| **Mistral 7B** | 7B | Fast inference | Slightly lower quality |

### D. Cost Comparison

| Approach | Setup | Per-Run | Annual (weekly) |
|----------|-------|---------|-----------------|
| Anthropic Haiku | $0 | $50-100 | $2,600-5,200 |
| Fine-tuned MLX | $20 | $0 | $20 |
| **Savings** | - | **100%** | **~$5,000** |

---

## Approval & Next Steps

**Ready to begin:** ☐  
**Start date:** ___________  
**Expected completion:** ___________

**Next immediate action:**
1. Create `scripts/01_prepare_data.py`
2. Run data preparation
3. Begin Tinker fine-tuning

---

*Last updated: 2026-01-29*
*Author: Claude (AI Assistant)*
*Platform: Tinker + MLX on M4 Max*
