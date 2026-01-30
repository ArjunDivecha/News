#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
exec python3 -u scripts/02_train_tinker_wandb.py 2>&1 | tee training_live.log
