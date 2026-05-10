from transformers import (
    AutoModel, AutoTokenizer,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoModelForVision2Seq, AutoProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image
from utils import load_jsonl, save_file, remove_extra_spaces, remove_prompt
import argparse
from tqdm import tqdm


# ============================================================
# MODEL SELECTION — uncomment the model you want to use
# ============================================================

# ---------- Model 1: InternVL2-8B ----------
model_id = "OpenGVLab/InternVL2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
).eval()
ACTIVE_MODEL = "internvl2"

# ---------- Model 2: Qwen3-VL-8B ----------
# model_id = "Qwen/Qwen3-VL-8B-Instruct"
# processor = AutoProcessor.from_pretrained(model_id)
# model = AutoModelForVision2Seq.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     pad_token_id=processor.tokenizer.pad_token_id
# )
# model.to("cuda:0")
# ACTIVE_MODEL = "qwen"

# ---------- Model 3: LLaVA-1.6 Mistral 7B ----------
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
# model = LlavaNextForConditionalGeneration.from_pretrained(
#     "llava-hf/llava-v1.6-mistral-7b-hf",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     pad_token_id=processor.tokenizer.pad_token_id
# )
# model.to("cuda:0")
# ACTIVE_MODEL = "llava"


# ============================================================
# IMAGE PREPROCESSING — only used for InternVL2
# ============================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_image_internvl(image_or_path, input_size=448):
    """Load and preprocess image for InternVL2 — returns float16 tensor on cuda."""
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path.convert("RGB")
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.float16).to("cuda:0")
    return pixel_values


# ============================================================
# PROMPT BUILDERS
# ============================================================

def get_add_messages(image, new_type, new_detail, prompt_id):

    # --- InternVL2: plain string with <image>\n prefix ---
    variants_internvl2 = [
        # 0: Color
        (
            f"<image>\nImagine {new_type} {new_detail} has been added to this scene. "
            "Describe the resulting image in at most 2 sentences, focusing on how the color palette changes. "
            "Output only the description, no explanations."
        ),
        # 1: Object Placement / Position
        (
            f"<image>\nAssume {new_type} {new_detail} is now placed in this scene. "
            "In 2 sentences, describe where it sits relative to other objects and how it fits spatially. "
            "Output only the description."
        ),
        # 2: Object Addition
        (
            f"<image>\nPicture {new_type} {new_detail} added to this image. "
            "Write a 2-sentence description of the updated scene, noting how the new element fits with existing objects. "
            "Output only the description, no explanations."
        ),
        # 3: Object Removal / Replacement
        (
            f"<image>\nImagine a similar element in the scene has been replaced by {new_type} {new_detail}. "
            "Describe the updated image in 2 sentences, noting what changed. "
            "Output only the description."
        ),
        # 4: Lighting / Time of Day
        (
            f"<image>\nAssume {new_type} {new_detail} is now present, changing the lighting or time of day. "
            "In 2 sentences, describe the scene with its new lighting atmosphere and shadows. "
            "Output only the description."
        ),
        # 5: Weather / Environment
        (
            f"<image>\nPicture the scene with {new_type} {new_detail} introduced into the environment. "
            "Describe the resulting weather or environmental conditions in 2 sentences. "
            "Output only the description."
        ),
        # 6: Human Expression / Pose
        (
            f"<image>\nImagine {new_type} {new_detail} has been added and influences a person's expression or pose. "
            "If no person is present, describe the scene with this addition in 2 sentences. "
            "Output only the description."
        ),
        # 7: Texture / Material
        (
            f"<image>\nAssume {new_type} {new_detail} has been added, introducing a new texture or material. "
            "Describe the scene in 2 sentences, focusing on how textures and surfaces now look. "
            "Output only the description."
        ),
        # 8: Scale / Size
        (
            f"<image>\nPicture {new_type} {new_detail} placed in the scene. "
            "In 2 sentences, describe its size relative to surrounding objects and the sense of scale it creates. "
            "Output only the description."
        ),
        # 9: Quantity / Count
        (
            f"<image>\nImagine {new_type} {new_detail} added to the scene, changing the count of objects. "
            "Describe the updated scene in 2 sentences, clearly mentioning the quantity of relevant elements. "
            "Output only the description."
        ),
    ]

    # --- Qwen / LLaVA: list-of-dicts with role/content format ---
    variants_qwen_llava = [
        # 0: Color
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine {new_type} {new_detail} has been added to this scene. "
                "Describe the resulting image in at most 2 sentences, focusing on how the color palette changes. "
                "Output only the description, no explanations."
            )},
        ]}],
        # 1: Object Placement / Position
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume {new_type} {new_detail} is now placed in this scene. "
                "In 2 sentences, describe where it sits relative to other objects and how it fits spatially. "
                "Output only the description."
            )},
        ]}],
        # 2: Object Addition
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture {new_type} {new_detail} added to this image. "
                "Write a 2-sentence description of the updated scene, noting how the new element fits with existing objects. "
                "Output only the description, no explanations."
            )},
        ]}],
        # 3: Object Removal / Replacement
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine a similar element in the scene has been replaced by {new_type} {new_detail}. "
                "Describe the updated image in 2 sentences, noting what changed. "
                "Output only the description."
            )},
        ]}],
        # 4: Lighting / Time of Day
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume {new_type} {new_detail} is now present, changing the lighting or time of day. "
                "In 2 sentences, describe the scene with its new lighting atmosphere and shadows. "
                "Output only the description."
            )},
        ]}],
        # 5: Weather / Environment
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture the scene with {new_type} {new_detail} introduced into the environment. "
                "Describe the resulting weather or environmental conditions in 2 sentences. "
                "Output only the description."
            )},
        ]}],
        # 6: Human Expression / Pose
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine {new_type} {new_detail} has been added and influences a person's expression or pose. "
                "If no person is present, describe the scene with this addition in 2 sentences. "
                "Output only the description."
            )},
        ]}],
        # 7: Texture / Material
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume {new_type} {new_detail} has been added, introducing a new texture or material. "
                "Describe the scene in 2 sentences, focusing on how textures and surfaces now look. "
                "Output only the description."
            )},
        ]}],
        # 8: Scale / Size
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture {new_type} {new_detail} placed in the scene. "
                "In 2 sentences, describe its size relative to surrounding objects and the sense of scale it creates. "
                "Output only the description."
            )},
        ]}],
        # 9: Quantity / Count
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine {new_type} {new_detail} added to the scene, changing the count of objects. "
                "Describe the updated scene in 2 sentences, clearly mentioning the quantity of relevant elements. "
                "Output only the description."
            )},
        ]}],
    ]

    # --- PaliGemma2: simple "<image> text" prompt ---
    variants_paligemma = [
        # 0: Color
        (
            f"<image> Imagine {new_type} {new_detail} has been added to this scene. "
            "Describe the resulting image in at most 2 sentences, focusing on how the color palette changes. "
            "Output only the description, no explanations."
        ),
        # 1: Object Placement / Position
        (
            f"<image> Assume {new_type} {new_detail} is now placed in this scene. "
            "In 2 sentences, describe where it sits relative to other objects and how it fits spatially. "
            "Output only the description."
        ),
        # 2: Object Addition
        (
            f"<image> Picture {new_type} {new_detail} added to this image. "
            "Write a 2-sentence description of the updated scene, noting how the new element fits with existing objects. "
            "Output only the description, no explanations."
        ),
        # 3: Object Removal / Replacement
        (
            f"<image> Imagine a similar element in the scene has been replaced by {new_type} {new_detail}. "
            "Describe the updated image in 2 sentences, noting what changed. "
            "Output only the description."
        ),
        # 4: Lighting / Time of Day
        (
            f"<image> Assume {new_type} {new_detail} is now present, changing the lighting or time of day. "
            "In 2 sentences, describe the scene with its new lighting atmosphere and shadows. "
            "Output only the description."
        ),
        # 5: Weather / Environment
        (
            f"<image> Picture the scene with {new_type} {new_detail} introduced into the environment. "
            "Describe the resulting weather or environmental conditions in 2 sentences. "
            "Output only the description."
        ),
        # 6: Human Expression / Pose
        (
            f"<image> Imagine {new_type} {new_detail} has been added and influences a person's expression or pose. "
            "If no person is present, describe the scene with this addition in 2 sentences. "
            "Output only the description."
        ),
        # 7: Texture / Material
        (
            f"<image> Assume {new_type} {new_detail} has been added, introducing a new texture or material. "
            "Describe the scene in 2 sentences, focusing on how textures and surfaces now look. "
            "Output only the description."
        ),
        # 8: Scale / Size
        (
            f"<image> Picture {new_type} {new_detail} placed in the scene. "
            "In 2 sentences, describe its size relative to surrounding objects and the sense of scale it creates. "
            "Output only the description."
        ),
        # 9: Quantity / Count
        (
            f"<image> Imagine {new_type} {new_detail} added to the scene, changing the count of objects. "
            "Describe the updated scene in 2 sentences, clearly mentioning the quantity of relevant elements. "
            "Output only the description."
        ),
    ]

    if ACTIVE_MODEL == "internvl2":
        return variants_internvl2[prompt_id]
    elif ACTIVE_MODEL == "paligemma":
        return variants_paligemma[prompt_id]
    else:
        return variants_qwen_llava[prompt_id]


def get_adjust_messages(image, mod_type, mod_detail, original_detail, prompt_id):

    # --- InternVL2 ---
    variants_internvl2 = [
        # 0: Color
        (
            f"<image>\nImagine the {mod_type} in this scene has changed from {mod_detail} to {original_detail}. "
            "Describe the updated image in at most 2 sentences, focusing on how the color shift affects the scene. "
            "Output only the description, no explanations."
        ),
        # 1: Object Placement / Position
        (
            f"<image>\nAssume the position of {mod_type} has been adjusted from {mod_detail} to {original_detail}. "
            "In 2 sentences, describe where it now sits relative to other objects in the scene. "
            "Output only the description."
        ),
        # 2: Object Addition
        (
            f"<image>\nPicture the scene where {mod_type} has been updated from {mod_detail} to {original_detail}. "
            "Write a 2-sentence description of the revised scene, noting how this change affects the composition. "
            "Output only the description, no explanations."
        ),
        # 3: Object Removal / Replacement
        (
            f"<image>\nImagine {mod_type} {mod_detail} has been replaced by {original_detail} in the scene. "
            "Describe the updated image in 2 sentences, focusing on what changed with this replacement. "
            "Output only the description."
        ),
        # 4: Lighting / Time of Day
        (
            f"<image>\nAssume the lighting has shifted from {mod_detail} to {original_detail} for {mod_type}. "
            "In 2 sentences, describe how the scene's atmosphere, shadows, and time of day now appear. "
            "Output only the description."
        ),
        # 5: Weather / Environment
        (
            f"<image>\nPicture the environment where {mod_type} has changed from {mod_detail} to {original_detail}. "
            "Describe the resulting weather or environmental conditions in 2 sentences. "
            "Output only the description."
        ),
        # 6: Human Expression / Pose
        (
            f"<image>\nImagine the {mod_type} of a person in the scene has shifted from {mod_detail} to {original_detail}. "
            "If no person is visible, describe the overall scene change in 2 sentences. "
            "Output only the description."
        ),
        # 7: Texture / Material
        (
            f"<image>\nAssume the texture or material of {mod_type} has changed from {mod_detail} to {original_detail}. "
            "Describe the scene in 2 sentences, focusing on how the surfaces and materials now appear. "
            "Output only the description."
        ),
        # 8: Scale / Size
        (
            f"<image>\nPicture {mod_type} resized from {mod_detail} to {original_detail} within the scene. "
            "In 2 sentences, describe how this size change affects the proportions and spatial relationships. "
            "Output only the description."
        ),
        # 9: Quantity / Count
        (
            f"<image>\nImagine the number of {mod_type} has changed from {mod_detail} to {original_detail} in the scene. "
            "Describe the updated scene in 2 sentences, clearly reflecting the new quantity. "
            "Output only the description."
        ),
    ]

    # --- Qwen / LLaVA ---
    variants_qwen_llava = [
        # 0: Color
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine the {mod_type} in this scene has changed from {mod_detail} to {original_detail}. "
                "Describe the updated image in at most 2 sentences, focusing on how the color shift affects the scene. "
                "Output only the description, no explanations."
            )},
        ]}],
        # 1: Object Placement / Position
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume the position of {mod_type} has been adjusted from {mod_detail} to {original_detail}. "
                "In 2 sentences, describe where it now sits relative to other objects in the scene. "
                "Output only the description."
            )},
        ]}],
        # 2: Object Addition
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture the scene where {mod_type} has been updated from {mod_detail} to {original_detail}. "
                "Write a 2-sentence description of the revised scene, noting how this change affects the composition. "
                "Output only the description, no explanations."
            )},
        ]}],
        # 3: Object Removal / Replacement
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine {mod_type} {mod_detail} has been replaced by {original_detail} in the scene. "
                "Describe the updated image in 2 sentences, focusing on what changed with this replacement. "
                "Output only the description."
            )},
        ]}],
        # 4: Lighting / Time of Day
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume the lighting has shifted from {mod_detail} to {original_detail} for {mod_type}. "
                "In 2 sentences, describe how the scene's atmosphere, shadows, and time of day now appear. "
                "Output only the description."
            )},
        ]}],
        # 5: Weather / Environment
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture the environment where {mod_type} has changed from {mod_detail} to {original_detail}. "
                "Describe the resulting weather or environmental conditions in 2 sentences. "
                "Output only the description."
            )},
        ]}],
        # 6: Human Expression / Pose
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine the {mod_type} of a person in the scene has shifted from {mod_detail} to {original_detail}. "
                "If no person is visible, describe the overall scene change in 2 sentences. "
                "Output only the description."
            )},
        ]}],
        # 7: Texture / Material
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Assume the texture or material of {mod_type} has changed from {mod_detail} to {original_detail}. "
                "Describe the scene in 2 sentences, focusing on how the surfaces and materials now appear. "
                "Output only the description."
            )},
        ]}],
        # 8: Scale / Size
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Picture {mod_type} resized from {mod_detail} to {original_detail} within the scene. "
                "In 2 sentences, describe how this size change affects the proportions and spatial relationships. "
                "Output only the description."
            )},
        ]}],
        # 9: Quantity / Count
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                f"Imagine the number of {mod_type} has changed from {mod_detail} to {original_detail} in the scene. "
                "Describe the updated scene in 2 sentences, clearly reflecting the new quantity. "
                "Output only the description."
            )},
        ]}],
    ]

    # --- PaliGemma2 ---
    variants_paligemma = [
        # 0: Color
        (
            f"<image> Imagine the {mod_type} in this scene has changed from {mod_detail} to {original_detail}. "
            "Describe the updated image in at most 2 sentences, focusing on how the color shift affects the scene. "
            "Output only the description, no explanations."
        ),
        # 1: Object Placement / Position
        (
            f"<image> Assume the position of {mod_type} has been adjusted from {mod_detail} to {original_detail}. "
            "In 2 sentences, describe where it now sits relative to other objects in the scene. "
            "Output only the description."
        ),
        # 2: Object Addition
        (
            f"<image> Picture the scene where {mod_type} has been updated from {mod_detail} to {original_detail}. "
            "Write a 2-sentence description of the revised scene, noting how this change affects the composition. "
            "Output only the description, no explanations."
        ),
        # 3: Object Removal / Replacement
        (
            f"<image> Imagine {mod_type} {mod_detail} has been replaced by {original_detail} in the scene. "
            "Describe the updated image in 2 sentences, focusing on what changed with this replacement. "
            "Output only the description."
        ),
        # 4: Lighting / Time of Day
        (
            f"<image> Assume the lighting has shifted from {mod_detail} to {original_detail} for {mod_type}. "
            "In 2 sentences, describe how the scene's atmosphere, shadows, and time of day now appear. "
            "Output only the description."
        ),
        # 5: Weather / Environment
        (
            f"<image> Picture the environment where {mod_type} has changed from {mod_detail} to {original_detail}. "
            "Describe the resulting weather or environmental conditions in 2 sentences. "
            "Output only the description."
        ),
        # 6: Human Expression / Pose
        (
            f"<image> Imagine the {mod_type} of a person in the scene has shifted from {mod_detail} to {original_detail}. "
            "If no person is visible, describe the overall scene change in 2 sentences. "
            "Output only the description."
        ),
        # 7: Texture / Material
        (
            f"<image> Assume the texture or material of {mod_type} has changed from {mod_detail} to {original_detail}. "
            "Describe the scene in 2 sentences, focusing on how the surfaces and materials now appear. "
            "Output only the description."
        ),
        # 8: Scale / Size
        (
            f"<image> Picture {mod_type} resized from {mod_detail} to {original_detail} within the scene. "
            "In 2 sentences, describe how this size change affects the proportions and spatial relationships. "
            "Output only the description."
        ),
        # 9: Quantity / Count
        (
            f"<image> Imagine the number of {mod_type} has changed from {mod_detail} to {original_detail} in the scene. "
            "Describe the updated scene in 2 sentences, clearly reflecting the new quantity. "
            "Output only the description."
        ),
    ]

    if ACTIVE_MODEL == "internvl2":
        return variants_internvl2[prompt_id]
    elif ACTIVE_MODEL == "paligemma":
        return variants_paligemma[prompt_id]
    else:
        return variants_qwen_llava[prompt_id]
        
# ============================================================
# GENERATION HELPER
# ============================================================

internvl2_generation_config = dict(
    do_sample=True,
    max_new_tokens=80,
    min_new_tokens=20,
)

def run_inference(image_pil, prompt_or_messages, temperature):
    """
    Unified inference for all four models.
      image_pil          : PIL.Image
      prompt_or_messages : str  (InternVL2, PaliGemma2)
                           list (Qwen, LLaVA)
      temperature        : float
    Returns cleaned caption string.
    """

    # ---- InternVL2 ----
    if ACTIVE_MODEL == "internvl2":
        pixel_values = load_image_internvl(image_pil)
        internvl2_generation_config["temperature"] = temperature
        response = model.chat(
            tokenizer,
            pixel_values,
            prompt_or_messages,
            internvl2_generation_config
        )
        return remove_extra_spaces(response)

    # ---- PaliGemma2 ----
    elif ACTIVE_MODEL == "paligemma":
        # processor handles image injection + tokenisation together
        inputs = processor(
            text=[prompt_or_messages],
            images=[image_pil],
            return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                min_new_tokens=20,
                max_new_tokens=150,
                do_sample=True,
                temperature=temperature,
            )

        # Decode only the newly generated tokens (strip the prompt)
        decoded = processor.decode(output[0][input_len:], skip_special_tokens=True)
        return remove_extra_spaces(decoded.strip())

    # ---- Qwen / LLaVA (shared processor path) ----
    else:
        text_prompt = processor.apply_chat_template(
            prompt_or_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt], images=[image_pil], return_tensors="pt"
        ).to("cuda:0")

        output = model.generate(
            **inputs,
            min_new_tokens=20,
            max_new_tokens=80,
            do_sample=True,
            temperature=temperature
        )
        decoded = processor.batch_decode(output, skip_special_tokens=True)[0]

        # Strip prompt prefix
        if "[/INST]" in decoded:          # LLaVA-Next / Mistral backbone
            decoded = decoded.split("[/INST]")[-1].strip()
        elif "assistant" in decoded:       # Qwen-style models
            decoded = decoded.split("assistant")[-1].strip()

        return remove_extra_spaces(decoded)


# ============================================================
# MAIN SUMMARIZATION LOOP
# ============================================================

def summarize_caption(raw_data, source_dir, style, prompt_id, temperature):
    adjusted_caps = []

    for data in tqdm(raw_data, desc="Generating summaries", total=len(raw_data)):
        if not data['has_modification']:
            continue

        print("qid:", data['qid'])
        path = os.path.join(source_dir, f"qid{data['qid']}.jpg")
        image = Image.open(path)

        caption = ""

        if style == "":
            for i in range(len(data['new_detail_type'])):
                new_type   = data['new_detail_type'][i]
                new_detail = data['added_details'][i]
                messages   = get_add_messages(image, new_type, new_detail, prompt_id)
                caption    = run_inference(image, messages, temperature)
        else:
            for i in range(len(data['refinement_type'])):
                mod_type        = data['refinement_type'][i]
                original_detail = data['removed_details'][i]
                mod_detail      = data['modified_details'][i]
                messages        = get_adjust_messages(image, mod_type, mod_detail, original_detail, prompt_id)
                caption         = run_inference(image, messages, temperature)

        print("new caption:", caption)
        adjusted_caps.append({'qid': data['qid'], 'query': caption})

    print("New captions are generated.")
    return adjusted_caps


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarization")
    parser.add_argument('--gt_file',     type=str,   required=True)
    parser.add_argument('--image_dir',   type=str,   required=False)
    parser.add_argument('--style',       type=str,   required=True)
    parser.add_argument('--des_path',    type=str,   required=True)
    parser.add_argument('--prompt_id',   type=int,   required=True)
    parser.add_argument('--temperature', type=float, required=True)

    args = parser.parse_args()

    annotations  = load_jsonl(args.gt_file)
    gen_captions = summarize_caption(
        annotations, args.image_dir, args.style, args.prompt_id, args.temperature
    )
    save_file(args.des_path, gen_captions)


# ============================================================
# RUN COMMANDS
# ============================================================

# --- InternVL2 / Qwen / LLaVA / PaliGemma2 (same CLI for all) ---

# python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl \
#   --image_dir dataset/images/val_style_cinematic/ \
#   --des_path exps_use/summarization_2/adjusted_caption_style_cinematic_sum \
#   --style cinematic --prompt_id 0 --temperature 0.7

# python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl \
#   --image_dir dataset/images/val_style_realistic/ \
#   --des_path exps_use/summarization_2/adjusted_caption_style_realistic_sum \
#   --style realistic --prompt_id 0 --temperature 0.7

# python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl \
#   --image_dir dataset/images/val_style_cartoon/ \
#   --des_path exps_use/summarization_2/adjusted_caption_style_cartoon_sum \
#   --style cartoon --prompt_id 0 --temperature 0.7

# python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl \
#   --image_dir dataset/images/val_style_scribble/ \
#   --des_path exps_use/summarization_2/adjusted_caption_style_scribble_sum \
#   --style scribble --prompt_id 0 --temperature 0.7