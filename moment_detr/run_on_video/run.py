import torch
import torch.nn.functional as F
import numpy as np
import os  # <-- ADDED IMPORT
import json
from utils.basic_utils import load_jsonl

from run_on_video.data_utils import ClipFeatureExtractor
from run_on_video.model_utils import build_inference_model
from utils.tensor_utils import pad_sequences_1d
from moment_detr.span_utils import span_cxw_to_xx
from utils.basic_utils import l2_normalize_np_array



def save_file(des_file_name, data):
    with open(f'{des_file_name}.jsonl', 'w') as output:
        for element in data:
            output.write(json.dumps(element) + '\n')
    output.close()
    print("Done.")


class MomentDETRPredictor:
    # --- MODIFIED __init__ to accept feature_dir ---
    def __init__(self, ckpt_path, clip_model_name_or_path="ViT-B/32", device="cuda", feature_dir="clip_features"):
        self.clip_len = 2  # seconds
        self.device = torch.device(device)
        self.feature_dir = feature_dir  # <-- ADDED
        
        print("Loading feature extractors (for text)...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1/self.clip_len, size=224, centercrop=True,
            model_name_or_path=clip_model_name_or_path, device=device
        )
        print("Loading trained Moment-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)

        # --- MODIFIED SECTION: Load pre-computed features ---
        # # Get basename, e.g., "RoripwjYFp8_60.0_210.0.mp4"
        # video_basename = os.path.basename(video_path)
        # Get file stem and add .npy, e.g., "RoripwjYFp8_60.0_210.0.npy"
        feature_filename = f"{video_path}.npz"
        feature_path = os.path.join(self.feature_dir, feature_filename)

        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Pre-computed feature file not found at: {feature_path}\n"
                f"Please make sure features are pre-extracted and saved in '{self.feature_dir}'"
            )
        
        print(f"Loading pre-computed video features from {feature_path}...")
        
        # --- FIX IS HERE ---
        # Load the .npz file (which is a dict-like object)
        npz_data = np.load(feature_path)
        # npz_data = np.load(feature_path)
        print(f"Available keys in {feature_path}: {list(npz_data.keys())}") # <-- ADD THIS LINE
        
        # Access the array using its key. 'arr_0' is the default.
        try:
            video_feats_np = npz_data['features']
        except KeyError:
            print(f"Error: Default key 'arr_0' not found in {feature_path}.")
            print(f"Available keys are: {list(npz_data.keys())}")
            print("Please update the code to use the correct key.")
            npz_data.close()
            raise
        
        # Now convert the np.ndarray to a torch tensor
        video_feats = torch.from_numpy(video_feats_np).to(self.device).float()

        # It's good practice to close the file handle
        npz_data.close()
       
        
       

        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, "The positional embedding of this pretrained MomentDETR only support video up " \
                                "to 150 secs (i.e., 75 2-sec clips) in length"
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        
        # Text features are still extracted normally
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
        query_feats, dtype=torch.float32, device=self.device, fixed_length=None)
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask
        )

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in MomentDETR's decoder, not the input text query
        prob = F.softmax(outputs["pred_logits"], -1)  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)
        _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            _score = _saliency_scores[j, :int(valid_vid_lengths[j])].tolist()
            _score = [round(e, 4) for e in _score]
            saliency_scores.append(_score)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
                pred_saliency_scores=saliency_scores[idx]  # List(float), len==n_frames, scores for each frame
            )
            predictions.append(cur_query_pred)

        return predictions
def run_example():
    base_dir = "/home/mohit/moment_detr/run_on_video/second_sem_part_2_summarization_with_internvl2"

    styles   = ["cartoon", "cinematic", "realistic", "scribble"]
    prompts  = ["prompt_0", "prompt_1", "prompt_2", "prompt_3", "prompt_4", "prompt_5", "prompt_6", "prompt_7", "prompt_8", "prompt_9"]
    temps    = ["temp_0.2"]

    ckpt_path = "run_on_video/moment_detr_ckpt/model_best.ckpt"
    clip_model_name_or_path = "ViT-B/32"
    precomputed_feature_dir = "/home/mohit/moment_detr/clip_features"

    output_base_dir = "/home/mohit/moment_detr/run_on_video/prediction_files_second_sem_part_2_summarization_with_internvl2"
    os.makedirs(output_base_dir, exist_ok=True)

    print("Building Moment-DETR model...")
    moment_detr_predictor = MomentDETRPredictor(
        ckpt_path=ckpt_path,
        clip_model_name_or_path=clip_model_name_or_path,
        device="cuda",
        feature_dir=precomputed_feature_dir
    )

    print("Running predictions...")

    for style in styles:
        for prompt in prompts:
            for temp in temps:

                input_path = os.path.join(base_dir, style, prompt, temp, "output.jsonl")

                if not os.path.exists(input_path):
                    print(f"⚠️ Missing: {input_path}")
                    continue

                queries = load_jsonl(input_path)
                print(f"Processing: {style} | {prompt} | {temp} ({len(queries)} queries)")

                prediction_list = []

                for query_data in queries:
                    video_path = query_data["vid"]
                    query_text = query_data["query"]

                    predictions = moment_detr_predictor.localize_moment(
                        video_path=video_path,
                        query_list=[query_text]
                    )

                    result = {
                        "qid": query_data["qid"],
                        "query": query_data["query"],
                        "vid": query_data["vid"],
                        "pred_relevant_windows": predictions[0]["pred_relevant_windows"],
                        "pred_saliency_scores": predictions[0]["pred_saliency_scores"],
                    }

                    prediction_list.append(result)

                # mirror same directory structure in output
                save_dir = os.path.join(output_base_dir, style, prompt, temp)
                os.makedirs(save_dir, exist_ok=True)

                output_path = os.path.join(save_dir, "pred_output.jsonl")
                save_file(output_path.replace(".jsonl", ""), prediction_list)

                print(f"✅ Saved: {output_path}")

    print("🎯 All predictions completed.")

    
if __name__ == "__main__":
    run_example()