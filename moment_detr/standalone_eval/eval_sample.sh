#!/usr/bin/env bash
# Usage: bash standalone_eval/eval_sample.sh
submission_path=/home/mohit/moment_detr/run_on_video/second_sem_summarization_with_qwen3_vl/cartoon/prompt_0/temp_0.2/output.jsonl
gt_path=data/highlight_val_release.jsonl
save_path=/home/mohit/moment_detr/standalone_eval/results/sample_val_preds_metrics.json

PYTHONPATH=$PYTHONPATH:. python standalone_eval/eval.py \
--submission_path ${submission_path} \
--gt_path ${gt_path} \
--save_path ${save_path}
