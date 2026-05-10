import os
import json

def load_jsonl(file_path):
    """Loads a JSONL file into a list of dicts."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {file_path}: {e}")
    return data


def save_file(des_file_name, data):
    """Saves a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(des_file_name), exist_ok=True)
    with open(des_file_name, 'w', encoding='utf-8') as output:
        for element in data:
            output.write(json.dumps(element) + '\n')
    print(f"✅ Saved: {des_file_name}")


# --- File paths ---
ground_truth_path = "/home/mohit/moment_detr/run_on_video/icq_highlight_release.jsonl"
input_folder  = "/home/mohit/icq-benchmark/exps_use/second_sem_part_2_summarization_with_internvl2"
output_folder = "/home/mohit/moment_detr/run_on_video/second_sem_part_2_summarization_with_internvl2"

# --- Load ground truth once ---
ground_truth_list = load_jsonl(ground_truth_path)
ground_truth_dict = {item["qid"]: item for item in ground_truth_list}

# --- Walk the entire folder tree ---
for root, dirs, files in os.walk(input_folder):
    for filename in files:
        if not (filename.endswith(".jsonl") or filename.endswith(".jsonl.jsonl")):
            continue

        input_path = os.path.join(root, filename)

        # Mirror the subfolder structure inside output_folder
        rel_path = os.path.relpath(input_path, input_folder)

        # Fix double extension: output.jsonl.jsonl → output.jsonl
        if filename.endswith(".jsonl.jsonl"):
            fixed_filename = filename[:-len(".jsonl.jsonl")] + ".jsonl"
            rel_path = os.path.join(os.path.dirname(rel_path), fixed_filename)

        output_path = os.path.join(output_folder, rel_path)

        # Load predictions
        prediction_list = load_jsonl(input_path)

        # Merge 'vid' from ground truth
        merged = []
        for pred_item in prediction_list:
            qid = pred_item.get("qid")
            if qid in ground_truth_dict:
                pred_item["vid"] = ground_truth_dict[qid]["vid"]
                merged.append(pred_item)
            else:
                print(f"⚠️  qid {qid} not found in ground truth, skipping.")

        # Save merged file
        save_file(output_path, merged)
        print(f"✅ Merged {len(merged)} items → {output_path}")

print("🎯 All files processed successfully.")