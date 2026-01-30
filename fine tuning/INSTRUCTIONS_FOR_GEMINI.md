# Instructions for Running Tinker Fine-Tuning

## Context
Training an ETF classifier using Tinker's supervised learning framework with automatic checkpointing and validation evaluation.

## Files
- **Training Script**: `scripts/05_train_proper.py` (FIXED - validation calculation corrected)
- **Data**: `data/processed/train.jsonl` (4,208 examples), `data/processed/val.jsonl` (493 examples)
- **Output**: `outputs/training_run/`
- **Log**: `training_proper.log`

## Training Configuration
- Model: meta-llama/Llama-3.1-8B-Instruct
- LoRA Rank: 16
- Learning Rate: 2e-4
- Batch Size: 4
- Epochs: 3 (3,156 total steps)
- **Saves checkpoint every 100 steps**
- **Evaluates validation every 50 steps**
- Expected duration: ~40-50 minutes

## Step 1: Start Training

```bash
cd "/Users/arjundivecha/Dropbox/AAA Backup/A Working/News/fine tuning"
venv/bin/python scripts/05_train_proper.py > training_proper.log 2>&1 &
echo "Training started with PID: $!"
```

## Step 2: Verify Training Started (wait 60 seconds)

```bash
tail -50 training_proper.log
```

**Expected output:**
- W&B initialization
- "🚀 TINKER PROPER TRAINING WITH AUTO-CHECKPOINTING"
- "Starting epoch 0"
- Training progress logs

## Step 3: Monitor Progress (every 5-10 minutes)

### Check current step and metrics:
```bash
tail -1 outputs/training_run/metrics.jsonl | jq '{step, epoch, train_nll: .train_mean_nll, val_nll}'
```

### Calculate ETA:
```python
python3 << 'EOF'
import json
with open('outputs/training_run/metrics.jsonl') as f:
    for line in f: pass
    last = json.loads(line)
current = last['step']
total = 3156
eta_min = (total - current) * last['time/step'] / 60
print(f"Step {current}/{total} ({current/total*100:.1f}%) - ETA: {eta_min:.0f} min")
EOF
```

## Step 4: VERIFY CHECKPOINTS ARE BEING SAVED

**CRITICAL - Check after step 100:**

```bash
# Should show checkpoints at steps 100, 200, 300, etc.
cat outputs/training_run/checkpoints.jsonl | jq -r '"Step " + .name + ": " + .sampler_path'
```

**Expected output:**
```
Step 000100: tinker://[session-id]:train:0/sampler_weights/000100
Step 000200: tinker://[session-id]:train:0/sampler_weights/000200
...
```

**If checkpoints.jsonl doesn't exist after step 100, SOMETHING IS WRONG - STOP AND INVESTIGATE.**

## Step 5: Monitor Validation Loss

```bash
# Check train vs validation loss trend
cat outputs/training_run/metrics.jsonl | jq 'select(.val_nll) | {step, train: (.train_mean_nll * 1000 | round / 1000), val: (.val_nll * 1000 | round / 1000)}'
```

**What to look for:**
- Train and val should be similar magnitude (both ~0.1-0.3 range)
- If val is 10x higher than train (e.g., train=0.2, val=2.0), **VALIDATION CALCULATION IS BROKEN - STOP TRAINING**
- Small increase in val over time is normal (overfitting)

## Step 6: Wait for Completion

Training completes when:
- Step reaches 3156
- Log shows "✅ TRAINING COMPLETE"
- Process exits

## Step 7: Verify Final Checkpoints

```bash
# Should have ~31 checkpoints (steps 100, 200, 300, ..., 3100)
wc -l outputs/training_run/checkpoints.jsonl
cat outputs/training_run/checkpoints.jsonl | tail -5
```

## Step 8: Post-Training Evaluation

```bash
# Run evaluation script on saved checkpoints
venv/bin/python scripts/04_evaluate_model.py
```

## Troubleshooting

### Training won't start:
- Check: `tail -100 training_proper.log` for error messages
- Common issues: TINKER_API_KEY not set, module import errors

### No checkpoints being saved:
- Check: `ls -lh outputs/training_run/`
- Should see: config.json, metrics.jsonl, checkpoints.jsonl (after step 100)

### Validation loss looks wrong (10x training):
- **STOP TRAINING IMMEDIATELY**
- Bug in validation calculation - do not waste compute

### Process died:
- Check: `tail -200 training_proper.log` for errors
- Check: `ps aux | grep train` to see if still running

## W&B Monitoring

Training logs to: https://wandb.ai/arjun-divecha-dancing-elephant/etf-classifier-tinker

Can monitor real-time:
- train_mean_nll (should decrease)
- val_nll (should be similar magnitude to train_mean_nll)

## Important Notes

1. **DO NOT restart training unless there's a clear error** - each restart wastes compute
2. **Always verify checkpoints are being saved** - training is useless without them
3. **Validation should match training magnitude** - if not, there's a bug
4. **Expected final metrics**: Train NLL ~0.08-0.12, Val NLL ~0.10-0.15
5. **All checkpoints are saved to Tinker cloud** - they persist after training
