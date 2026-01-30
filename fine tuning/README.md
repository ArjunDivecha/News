# Fine-Tuning Project: Asset Classification Pipeline

This directory contains everything needed to fine-tune and deploy a local asset classification model, replacing Anthropic API calls with a fast, private MLX-based classifier running on M4 Max.

## 🎯 Goal

Replace Anthropic Haiku ($50-100/run, 60 minutes) with fine-tuned Llama 3.1 8B ($0/run, 2-5 minutes).

## 📁 Directory Structure

```
fine tuning/
├── IMPLEMENTATION_PLAN.md   # Complete implementation guide
├── data/
│   ├── raw/                 # Symlinks to source Excel files
│   ├── processed/           # JSONL training data
│   └── validation/          # Comparison results
├── scripts/
│   ├── 01_prepare_data.py   # Data preparation
│   ├── 02_train_tinker.py   # Tinker fine-tuning
│   ├── 03_download_weights.py
│   ├── 04_setup_mlx.py
│   ├── 05_validate.py
│   └── 06_migrate_scripts.py
├── models/
│   ├── tinker_export/       # Downloaded from Tinker
│   └── mlx_ready/           # Converted for MLX
├── inference/
│   ├── mlx_classifier.py    # MLX inference engine
│   └── unified_classifier.py # Drop-in replacement
└── validation/
    └── compare_models.py    # Head-to-head comparison
```

## 🚀 Quick Start

### Phase 1: Data Preparation

```bash
cd "fine tuning"
python scripts/01_prepare_data.py
```

Expected output:
- `data/processed/train.jsonl` (~3,800 examples)
- `data/processed/val.jsonl` (~450 examples)
- `data/processed/test.jsonl` (~230 examples)

### Phase 2: Fine-Tuning on Tinker

```bash
export TINKER_API_KEY="your-api-key"
python scripts/02_train_tinker.py
```

Duration: ~30-60 minutes  
Cost: ~$10-20

### Phase 3: Setup Local Inference

```bash
python scripts/04_setup_mlx.py
source mlx_env/bin/activate
```

### Phase 4: Validate

```bash
python scripts/05_validate.py --samples 200
```

### Phase 5: Migrate Production Scripts

```bash
python scripts/06_migrate_scripts.py --apply
```

## 📊 Expected Performance

| Metric | Before (Anthropic) | After (MLX) | Improvement |
|--------|-------------------|-------------|-------------|
| Cost | ~$50-100/run | $0 | 100% savings |
| Speed | ~60 min | ~2-5 min | 12-30x faster |
| Latency | ~2,500ms | ~35ms | 70x faster |
| Privacy | External API | Local | Full control |

## 🛠️ Requirements

- Mac M4 Max with 128GB RAM
- Python 3.11+
- Tinker API access
- ~50GB free disk space (for models)

## 📖 Full Documentation

See `IMPLEMENTATION_PLAN.md` for complete details.

## 🆘 Troubleshooting

**Issue:** Tinker training fails  
**Fix:** Check API key, verify JSONL format

**Issue:** MLX model loads slowly  
**Fix:** First load downloads ~16GB, subsequent loads are fast

**Issue:** Lower accuracy than expected  
**Fix:** See Phase 5 options in IMPLEMENTATION_PLAN.md

## 📅 Timeline

- **Days 1-2:** Data preparation
- **Days 3-5:** Fine-tuning on Tinker
- **Days 6-8:** MLX setup & optimization
- **Days 9-11:** Validation & comparison
- **Days 12-14:** Production migration

Total: 1-2 weeks

## 💰 Cost Savings

- **Setup cost:** ~$20 (Tinker training)
- **Per-run savings:** ~$50-100
- **Annual savings (weekly runs):** ~$5,000

---

**Status:** Ready to start Phase 1  
**Next step:** Create and run `scripts/01_prepare_data.py`
