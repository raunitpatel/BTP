#!/bin/bash
set -o pipefail
export PYTHONUNBUFFERED=1

echo "Starting structured experiment runs..."

# -------- TIMESTAMP + LOG DIR --------
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
log_dir="logs/exp_${timestamp}"
mkdir -p "$log_dir"

# -------- BASE COMMAND --------
BASE_CMD="python3 -u sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl"

# -------- STYLES --------
declare -A STYLES
STYLES=(
  ["cinematic"]="dataset/images/val_style_cinematic/"
  ["realistic"]="dataset/images/val_style_realistic/"
  ["cartoon"]="dataset/images/val_style_cartoon/"
  ["scribble"]="dataset/images/val_style_scribble/"
)

# -------- OUTPUT ROOT --------
BASE_OUT="exps_use/second_sem_part_2_summarization_with_internvl2"

# -------- PROMPTS & TEMPS --------
PROMPTS=(0 1 2 3 4 5 6 7 8 9)
TEMPS=(0.2)

# -------- COUNTERS --------
skipped=0
completed=0
failed=0

# -------- MAIN LOOP --------
for STYLE in "${!STYLES[@]}"; do
    IMAGE_DIR="${STYLES[$STYLE]}"

    echo "==============================="
    echo "STYLE: $STYLE"
    echo "==============================="

    for PROMPT_ID in "${PROMPTS[@]}"; do
        for TEMP in "${TEMPS[@]}"; do

            # -------- STRUCTURED OUTPUT PATH --------
            OUT_DIR="${BASE_OUT}/${STYLE}/prompt_${PROMPT_ID}/temp_${TEMP}"
            DES_PATH="${OUT_DIR}/output.jsonl"

            # -------- RESUME LOGIC: SKIP IF OUTPUT EXISTS AND IS VALID JSONL --------
            if [ -f "$DES_PATH" ] && python3 -c "
import sys
lines = open('$DES_PATH').readlines()
if not lines:
    sys.exit(1)
import json
[json.loads(l) for l in lines if l.strip()]
" 2>/dev/null; then
                echo "SKIPPING (already done): STYLE=$STYLE | PROMPT=$PROMPT_ID | TEMP=$TEMP"
                ((skipped++))
                continue
            fi

            # -------- REMOVE CORRUPT/INCOMPLETE OUTPUT IF EXISTS --------
            rm -f "$DES_PATH"

            mkdir -p "$OUT_DIR"

            # -------- SAVE CONFIG --------
            echo "{
  \"style\": \"${STYLE}\",
  \"prompt_id\": ${PROMPT_ID},
  \"temperature\": ${TEMP}
}" > "${OUT_DIR}/config.json"

            # -------- BUILD COMMAND --------
            CMD="$BASE_CMD \
                --image_dir $IMAGE_DIR \
                --des_path $DES_PATH \
                --style $STYLE \
                --prompt_id $PROMPT_ID \
                --temperature $TEMP"

            echo "Running: STYLE=$STYLE | PROMPT=$PROMPT_ID | TEMP=$TEMP"

            # -------- EXECUTION WITH ERROR HANDLING --------
            if $CMD > "$log_dir/${STYLE}_p${PROMPT_ID}_t${TEMP}.log" 2>&1; then
                echo "Saved → $DES_PATH"
                ((completed++))
            else
                echo "FAILED: STYLE=$STYLE | PROMPT=$PROMPT_ID | TEMP=$TEMP — check $log_dir/${STYLE}_p${PROMPT_ID}_t${TEMP}.log"
                ((failed++))
                # Remove incomplete output so it reruns next time
                rm -f "$DES_PATH"
            fi

            echo "-------------------------------------"

        done
    done

    echo "Completed STYLE: $STYLE"
    echo ""
done

echo "==============================="
echo "ALL RUNS FINISHED"
echo "  Skipped  (already done): $skipped"
echo "  Completed (this run):    $completed"
echo "  Failed:                  $failed"
echo "==============================="