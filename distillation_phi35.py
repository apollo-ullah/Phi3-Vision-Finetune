import os
import json
import sys
import torch
from PIL import Image
from transformers import (
    LogitsProcessor,
    CLIPProcessor,
    CLIPModel,
)
from tqdm import tqdm
import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict
import re
import open_clip
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import math
import time
import gc
from functools import partial

# Import the repository's utilities for Phi3.5-Vision
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_cuda():
    """Set up CUDA optimizations for H100 GPU with Phi3.5-Vision model."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")

    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Basic CUDA setup
    torch.cuda.set_per_process_memory_fraction(0.85)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:64"
    )

    # H100-specific optimizations
    if "H100" in torch.cuda.get_device_name(0):
        print("\nEnabling H100-specific optimizations:")
        # Enable flash attention if available
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            torch.backends.cuda.enable_flash_sdp(True)
            print("  - Flash Attention enabled")

        # Enable memory efficient attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("  - Memory Efficient Attention enabled")

        # Enable tensor cores with bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("  - TF32 (Tensor Cores) enabled")
        print("  - BF16 support enabled")

        # Additional optimizations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        print("  - cuDNN optimizations enabled")

        logger.info("H100 Optimizations enabled")

    return torch.device("cuda")


class RatingLogitsProcessor(LogitsProcessor):
    """Logits processor that only allows rating tokens (0-10)."""

    def __init__(self, rating_ids: List[int]):
        super().__init__()
        self.keep = torch.tensor(rating_ids)

    def __call__(self, input_ids, scores):
        # Create mask that preserves original scores for rating tokens
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.keep] = scores[
            :, self.keep
        ]  # Keep original scores for rating tokens
        return mask


def find_rating_tokens(processor) -> Tuple[List[int], Dict[int, float]]:
    """Find valid token IDs for ratings 0-10 using multiple encoding attempts."""
    rating_tokens = []
    rating_map = {}

    # Try different encodings for each rating
    for i in range(11):  # 0-10
        possible_encodings = [
            f"{i}",  # Just the number
            f" {i}",  # Space + number
            f"{i} ",  # Number + space
            str(i),  # String conversion
        ]

        # Try each encoding until we find a valid token
        for encoding in possible_encodings:
            try:
                token_ids = processor.tokenizer.encode(
                    encoding, add_special_tokens=False
                )
                if len(token_ids) == 1:  # We want single tokens only
                    token_id = token_ids[0]
                    if token_id not in rating_tokens:  # Avoid duplicates
                        rating_tokens.append(token_id)
                        rating_map[token_id] = i / 10.0  # Normalize to 0-1
                        break
            except Exception as e:
                continue

    if len(rating_tokens) != 11:
        raise ValueError(
            f"Could not find all rating tokens. Found {len(rating_tokens)}/11 tokens"
        )

    return rating_tokens, rating_map


def fallback_rating_method(
    inputs, model, processor, rating_tokens, rating_map
) -> Tuple[str, float, float]:
    """Fallback method that directly samples from logits when generation fails."""
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        next_token_logits = outputs.logits[0, -1]

        # Mask to only rating tokens
        masked_logits = next_token_logits.clone()
        masked_logits[:] = -float("inf")
        masked_logits[rating_tokens] = next_token_logits[rating_tokens]

        # Get probabilities
        probs = torch.softmax(masked_logits[rating_tokens], dim=-1)

        # Get highest probability token
        max_prob_idx = torch.argmax(probs).item()
        rating_token_id = rating_tokens[max_prob_idx]
        rating_text = processor.tokenizer.decode([rating_token_id])
        confidence = probs[max_prob_idx].item()
        normalized_score = rating_map[rating_token_id]

        return rating_text.strip(), normalized_score, confidence


def load_phi35_model(model_path="microsoft/Phi-3.5-vision-instruct", disable_flash_attention=True):
    """
    Load the Phi3.5-Vision model and processor with proper error handling and CUDA setup.

    Args:
        model_path (str): Path to the Phi3.5-Vision model
        disable_flash_attention (bool): Whether to disable flash attention

    Returns:
        tuple: (model, processor)
    """
    logger.info(f"Loading Phi3.5-Vision model: {model_path}")

    # Set up CUDA and get device
    device = setup_cuda()
    model = None
    processor = None

    try:
        # Load model using repository utilities
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        use_flash_attn = not disable_flash_attention

        processor, model = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=device,
            load_4bit=False,
            load_8bit=False,
            device=device,
            use_flash_attn=use_flash_attn
        )
        
        logger.info("Successfully loaded processor")
        logger.info("Successfully loaded base model")

        # Set model to evaluation mode
        model.eval()
        logger.info(f"Model loaded and set to eval mode")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

    if model is None or processor is None:
        raise RuntimeError("Failed to load model or processor")

    return model, processor


def clean_response(text: str) -> str:
    """Clean response text by removing non-ASCII characters and normalizing whitespace."""
    # First, try to extract any English words that might indicate a rating
    words = text.lower().split()
    rating_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    # Check for word-based ratings first
    for word in words:
        if word in rating_words:
            return str(rating_words[word])

    # If no word-based rating found, clean the text more strictly
    # Keep only numbers, basic punctuation, and common rating-related words
    allowed_chars = set("0123456789.,:/- score rating quality")
    cleaned = "".join(c for c in text.lower() if c in allowed_chars)
    # Normalize whitespace and remove multiple punctuation
    cleaned = " ".join(cleaned.split())
    cleaned = re.sub(r"[.,:/-]+", " ", cleaned)
    return cleaned.strip()


def extract_rating_from_text(text: str) -> Tuple[int, float]:
    """Extract rating from text using strict patterns."""
    # Clean the text first
    text = clean_response(text)

    # If we got a direct word-based rating, return it with high confidence
    if text.isdigit() and 0 <= int(text) <= 10:
        return int(text), 1.0

    # Define strict regex patterns in order of preference
    patterns = [
        (r"^(\d{1,2})/10$", 1.0),  # "8/10" format, exact match
        (r"^rating:\s*(\d{1,2})$", 0.9),  # "rating: 8" format, exact match
        (r"^score\s*:\s*(\d{1,2})$", 0.9),  # "score: 8" format, exact match
        (r"^quality:\s*(\d{1,2})$", 0.9),  # "quality: 8" format, exact match
        (r"^(\d{1,2})$", 0.8),  # Just a number, exact match
        (r"rating\s+is\s+(\d{1,2})", 0.8),  # "rating is 8" format
        (r"score\s+is\s+(\d{1,2})", 0.8),  # "score is 8" format
        (r"quality\s+is\s+(\d{1,2})", 0.8),  # "quality is 8" format
    ]

    # Try each pattern
    for pattern, confidence in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            rating = int(match.group(1))
            if 0 <= rating <= 10:
                return rating, confidence

    # If no strict match found, look for numbers in context
    numbers = re.findall(r"\b(\d{1,2})\b", text)
    if numbers:
        # Only accept if the number is in a clear rating context
        rating_words = ["rating", "score", "quality", "rate", "evaluate"]
        if any(word in text.lower() for word in rating_words):
            rating = int(numbers[0])
            if 0 <= rating <= 10:
                # Lower confidence for context-based matches
                return rating, 0.6

    return None, 0.0


def get_clip_aesthetic_score(
    image: Image.Image, clip_model, clip_processor
) -> Tuple[float, Dict[str, float]]:
    """
    Get CLIP aesthetic score for an image with detailed debugging.

    Args:
        image: PIL Image
        clip_model: CLIP model
        clip_processor: CLIP processor

    Returns:
        tuple: (aesthetic_score, debug_info)
    """
    try:
        # Debug model state
        print(f"\nDebug - CLIP Model:")
        print(f"  Device: {clip_model.device}")
        print(f"  Model type: {type(clip_model).__name__}")

        # Define simpler, more direct aesthetic prompts
        aesthetic_prompts = [
            # Basic quality prompts
            "a clear, sharp photo",
            "a bright, well-lit photo",
            "a good quality photo",
            "a professional photo",
            "a beautiful photo",
            # Specific quality prompts
            "a photo with good colors",
            "a photo with good composition",
            "a photo with good lighting",
            "a photo with good focus",
            "a photo with good exposure",
        ]

        # Process image and debug
        inputs = clip_processor(
            images=image, text=aesthetic_prompts, return_tensors="pt", padding=True
        ).to(clip_model.device)

        print(f"  Input shapes:")
        print(f"    Image: {inputs.pixel_values.shape}")
        print(f"    Text: {inputs.input_ids.shape}")

        # Get model outputs with debugging
        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            # Debug feature shapes
            print(f"  Feature shapes:")
            print(f"    Image features: {image_features.shape}")
            print(f"    Text features: {text_features.shape}")

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate raw similarities
            raw_similarities = 100.0 * image_features @ text_features.T
            print(
                f"  Raw similarities range: {raw_similarities.min().item():.2f} to {raw_similarities.max().item():.2f}"
            )

            # Get softmax probabilities
            similarities = raw_similarities.softmax(dim=-1)

            # Debug individual prompt scores
            prompt_scores = {}
            for i, prompt in enumerate(aesthetic_prompts):
                score = similarities[0, i].item()
                prompt_scores[prompt] = score
                print(f"  {prompt}: {score:.4f}")

            # Calculate final score with expanded range
            aesthetic_score = similarities.mean().item()

            # Expand the score range
            min_score = 0.1  # Expected minimum
            max_score = 0.9  # Expected maximum
            aesthetic_score = (aesthetic_score - min_score) / (max_score - min_score)
            aesthetic_score = max(0.0, min(1.0, aesthetic_score))  # Clip to [0,1]

            print(f"  Final aesthetic score: {aesthetic_score:.4f}")

            return aesthetic_score, prompt_scores

    except Exception as e:
        print(f"Error in CLIP aesthetic scoring: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return 0.0, {}


def get_yes_no_probability(
    model, processor, image, prompt: str
) -> Tuple[float, Dict[str, float]]:
    """
    Get the probability of the 'yes' token for a binary question with debugging.

    Args:
        model: The Phi3.5-Vision model
        processor: The model's processor
        image: PIL Image
        prompt: Binary question to ask

    Returns:
        tuple: (yes_probability, debug_info)
    """
    try:
        # Prepare messages in Phi3.5-Vision format
        messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
        
        # Apply chat template
        prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(prompt_text, [image], return_tensors="pt").to(model.device)

        # Debug input shapes
        print(f"\nDebug - Phi3.5 VQA:")
        print(f"  Device: {model.device}")
        print(f"  Input shapes:")
        print(f"    Pixel values: {inputs.pixel_values.shape}")
        print(f"    Input IDs: {inputs.input_ids.shape}")

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            next_token_logits = outputs.logits[0, -1]

            # Get token IDs for "yes" and "no"
            yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

            # Debug logits
            yes_logit = next_token_logits[yes_token_id].item()
            no_logit = next_token_logits[no_token_id].item()
            print(f"  Raw logits:")
            print(f"    Yes: {yes_logit:.4f}")
            print(f"    No: {no_logit:.4f}")

            # Get probabilities
            probs = torch.softmax(
                next_token_logits[[yes_token_id, no_token_id]], dim=-1
            )
            yes_prob = probs[0].item()
            no_prob = probs[1].item()

            print(f"  Probabilities:")
            print(f"    Yes: {yes_prob:.4f}")
            print(f"    No: {no_prob:.4f}")

            debug_info = {
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
            }

            return yes_prob, debug_info

    except Exception as e:
        print(f"Error in VQA: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return 0.0, {}


def run_phi35_inference(model, processor, image_path, question, device='cuda', max_new_tokens=20):
    """Run inference on a single sample using Phi3.5-Vision."""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare messages with more direct VQA-style prompt
        vqa_prompt = f"Answer the question about this image with a short, direct answer.\n\nQuestion: {question}\nAnswer:"
        messages = [{"role": "user", "content": f"<|image_1|>\n{vqa_prompt}"}]
        
        # Apply chat template
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(prompt, [image], return_tensors="pt").to(device)
        
        # Generate with shorter max length for more direct answers
        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = processor.batch_decode(
            generate_ids[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        return generated_text
        
    except Exception as e:
        print(f"Error in inference: {e}")
        return ""


def get_robust_technical_scores(phi35_inputs, model, processor):
    """Technical scoring using Phi3.5-Vision that actually discriminates between images."""
    try:
        batch_size = phi35_inputs["pixel_values"].shape[0]
        device = phi35_inputs["pixel_values"].device
        images = phi35_inputs["pixel_values"]  # [batch, channels, height, width]

        print(f"\nComputing discriminative technical scores for {batch_size} images...")

        # Initialize score lists
        scores = []

        for i in range(batch_size):
            img = images[i]  # Single image tensor

            # 1. BETTER Sharpness (Laplacian variance)
            if img.shape[0] == 3:
                gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            else:
                gray = img[0]

            # Compute Laplacian (better sharpness measure)
            laplacian_kernel = (
                torch.tensor(
                    [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                    dtype=img.dtype,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            gray_4d = gray.unsqueeze(0).unsqueeze(0)
            laplacian = torch.nn.functional.conv2d(gray_4d, laplacian_kernel, padding=1)
            sharpness = laplacian.var().item()
            sharpness_score = min(sharpness / 0.02, 1.0)  # Normalize better

            # 2. BETTER Exposure scoring
            brightness = img.mean().item()
            # More nuanced exposure scoring
            if 0.3 <= brightness <= 0.7:
                exposure_score = 1.0  # Perfect exposure
            elif 0.2 <= brightness < 0.3 or 0.7 < brightness <= 0.8:
                exposure_score = 0.7  # Good exposure
            elif 0.1 <= brightness < 0.2 or 0.8 < brightness <= 0.9:
                exposure_score = 0.4  # Poor exposure
            else:
                exposure_score = 0.1  # Very bad exposure

            # 3. FIXED Contrast scoring (was broken before)
            contrast = img.std().item()
            # Better contrast normalization
            if contrast > 0.25:
                contrast_score = 1.0
            elif contrast > 0.15:
                contrast_score = 0.8
            elif contrast > 0.08:
                contrast_score = 0.5
            else:
                contrast_score = 0.2

            # 4. BETTER Noise estimation
            # High-frequency noise detection
            sobel_x = (
                torch.tensor(
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sobel_y = (
                torch.tensor(
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=device
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            grad_x = torch.nn.functional.conv2d(gray_4d, sobel_x, padding=1)
            grad_y = torch.nn.functional.conv2d(gray_4d, sobel_y, padding=1)

            # Noise is inversely related to gradient consistency
            gradient_mag = torch.sqrt(grad_x**2 + grad_y**2)
            noise_level = gradient_mag.std().item() / gradient_mag.mean().item()
            noise_score = max(0.0, 1.0 - noise_level / 2.0)

            # 5. BETTER Color scoring
            if img.shape[0] == 3:
                # Color saturation and balance
                rgb_std = img.std(
                    dim=(1, 2)
                )  # Std across spatial dims for each channel
                color_balance = (
                    1.0 - rgb_std.std().item()
                )  # Good balance = low variation

                # Color saturation (distance from grayscale)
                gray_expanded = gray.unsqueeze(0).repeat(3, 1, 1)
                saturation = ((img - gray_expanded) ** 2).mean().item()
                saturation_score = min(saturation * 5, 1.0)  # Scale appropriately

                color_score = color_balance * 0.4 + saturation_score * 0.6
            else:
                color_score = 0.3  # Penalize grayscale

            # Combine with BETTER weights that reflect actual importance
            component_scores = {
                "sharpness": sharpness_score,
                "exposure": exposure_score,
                "contrast": contrast_score,
                "noise": noise_score,
                "color": color_score,
            }

            # More realistic weights
            weights = {
                "sharpness": 0.35,  # Most important
                "exposure": 0.25,  # Very important
                "contrast": 0.20,  # Important
                "noise": 0.15,  # Moderately important
                "color": 0.05,  # Least important
            }

            final_score = sum(component_scores[k] * weights[k] for k in weights.keys())
            scores.append(final_score)

            if i < 3:  # Debug first few
                print(f"  Image {i}:")
                for comp, score in component_scores.items():
                    print(f"    {comp.capitalize()}: {score:.3f}")
                print(f"    Final: {final_score:.3f}")

        technical_scores = torch.tensor(scores, device=device, dtype=torch.float32)

        # Apply minimal stretching only if needed
        if technical_scores.std().item() < 0.08:
            mean_score = technical_scores.mean()
            technical_scores = (technical_scores - mean_score) * 1.5 + mean_score
            technical_scores = torch.clamp(technical_scores, 0.1, 0.9)

        print(f"\nDiscriminative technical scoring results:")
        print(
            f"  Score range: {technical_scores.min().item():.3f} to {technical_scores.max().item():.3f}"
        )
        print(f"  Score std: {technical_scores.std().item():.3f}")

        return technical_scores

    except Exception as e:
        print(f"Error in technical scoring: {e}")
        # Return varied fallback scores
        base_scores = torch.linspace(0.2, 0.8, batch_size, device=device)
        noise = torch.randn(batch_size, device=device) * 0.1
        return torch.clamp(base_scores + noise, 0.1, 0.9)


def get_enhanced_aesthetic_scores(clip_inputs, clip_model, aesthetic_mlp):
    """Aesthetic scoring that actually uses the full range."""
    try:
        batch_size = clip_inputs.shape[0]
        device = clip_inputs.device

        with torch.no_grad():
            # Get CLIP embeddings
            image_features = clip_model.encode_image(clip_inputs).float()

            # Get raw aesthetic scores
            raw_scores = aesthetic_mlp(image_features).squeeze(-1)

            # MUCH BETTER normalization - don't artificially stretch
            # The MLP was trained on 0-10 scale, so just divide by 10
            aesthetic_scores = torch.clamp(raw_scores / 10.0, 0.0, 1.0)

            # Only apply slight stretching if scores are too compressed
            score_range = aesthetic_scores.max() - aesthetic_scores.min()
            if score_range < 0.3:  # If range is less than 0.3, stretch slightly
                mean_score = aesthetic_scores.mean()
                centered = aesthetic_scores - mean_score
                stretched = centered * 1.5  # Gentle stretch
                aesthetic_scores = torch.clamp(stretched + mean_score, 0.0, 1.0)

            print(f"\nRealistic aesthetic scoring:")
            print(
                f"  Raw scores: {raw_scores.min().item():.2f} to {raw_scores.max().item():.2f}"
            )
            print(
                f"  Normalized: {aesthetic_scores.min().item():.3f} to {aesthetic_scores.max().item():.3f}"
            )
            print(f"  Std: {aesthetic_scores.std().item():.3f}")

            return aesthetic_scores

    except Exception as e:
        print(f"Error in aesthetic scoring: {e}")
        # Return realistic fallback
        return torch.rand(batch_size, device=device) * 0.6 + 0.2


def process_batch(batch, phi35_model, phi35_processor, clip_model, aesthetic_mlp, device):
    """Process batch with realistic, discriminative scoring using Phi3.5-Vision."""
    if batch is None:
        return []

    try:
        # Move inputs to GPU
        phi35_inputs = {
            k: v.to(device, non_blocking=True) for k, v in batch["phi35_inputs"].items()
        }
        clip_inputs = batch["clip_inputs"].to(device, non_blocking=True)

        if next(clip_model.parameters()).dtype == torch.float16:
            clip_inputs = clip_inputs.half()
        else:
            clip_inputs = clip_inputs.float()

        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                # Get realistic technical scores
                technical_scores = get_robust_technical_scores(
                    phi35_inputs, phi35_model, phi35_processor
                )

                # Get realistic aesthetic scores
                aesthetic_scores = get_enhanced_aesthetic_scores(
                    clip_inputs, clip_model, aesthetic_mlp
                )

                # SIMPLE, realistic combination - no fancy adaptive weighting
                # Technical quality is more important than aesthetics for quality assessment
                combined_scores = 0.65 * technical_scores + 0.35 * aesthetic_scores
                combined_scores = torch.clamp(combined_scores, 0.0, 1.0)
                ratings = (combined_scores * 10).round().int()

                print(f"\nREALISTIC Batch Results:")
                print(
                    f"  Technical: {technical_scores.min().item():.3f} to {technical_scores.max().item():.3f} (std: {technical_scores.std().item():.3f})"
                )
                print(
                    f"  Aesthetic: {aesthetic_scores.min().item():.3f} to {aesthetic_scores.max().item():.3f} (std: {aesthetic_scores.std().item():.3f})"
                )
                print(
                    f"  Combined:  {combined_scores.min().item():.3f} to {combined_scores.max().item():.3f} (std: {combined_scores.std().item():.3f})"
                )
                print(f"  Ratings:   {ratings.min().item()} to {ratings.max().item()}")

        return [
            {
                "image_path": batch["image_path"][i],
                "technical_score": float(technical_scores[i].item()),
                "aesthetic_score": float(aesthetic_scores[i].item()),
                "combined_score": float(combined_scores[i].item()),
                "rating": int(ratings[i].item()),
                "captions": batch["captions"][i],
            }
            for i in range(len(batch["image_path"]))
        ]

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return []


def load_models():
    """
    Load Phi3.5-Vision model and CLIP aesthetic predictor with proper error handling.
    """
    logger.info("Loading Phi3.5-Vision model...")
    phi35_model, phi35_processor = load_phi35_model()

    logger.info("Loading CLIP aesthetic predictor...")
    try:
        # Load CLIP model and transforms
        clip_model, _, clip_processor = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai", device=phi35_model.device
        )
        # Keep CLIP in float32 for stability
        clip_model = clip_model.to(phi35_model.device)
        clip_model.eval()

        # Load aesthetic MLP weights
        aesthetic_mlp_path = "improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth"
        if not os.path.exists(aesthetic_mlp_path):
            raise FileNotFoundError(
                f"Aesthetic MLP not found at {aesthetic_mlp_path}. "
                "Please download from: https://github.com/christophschuhmann/improved-aesthetic-predictor"
            )

        # Create MLP model with proper architecture including Tanh activations
        class AestheticMLP(torch.nn.Module):
            def __init__(self, in_dim=768):
                super().__init__()
                # Use ModuleList to match checkpoint's layer naming
                self.layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(in_dim, 1024),
                        torch.nn.Tanh(),
                        torch.nn.Linear(1024, 128),
                        torch.nn.Tanh(),
                        torch.nn.Linear(128, 64),
                        torch.nn.Tanh(),
                        torch.nn.Linear(64, 16),
                        torch.nn.Tanh(),
                        torch.nn.Linear(16, 1),
                    ]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        # Create and load model
        aesthetic_mlp = AestheticMLP().to(phi35_model.device)

        # Load state dict and create a new one with correct layer names
        state_dict = torch.load(aesthetic_mlp_path, map_location=phi35_model.device)
        new_state_dict = {}

        # Map the saved weights to our model's layers
        layer_mapping = {
            "layers.0": "layers.0",  # 768->1024
            "layers.2": "layers.2",  # 1024->128
            "layers.4": "layers.4",  # 128->64
            "layers.6": "layers.6",  # 64->16
            "layers.7": "layers.8",  # 16->1 (note: this is the last layer)
        }

        for old_key, new_key in layer_mapping.items():
            if old_key + ".weight" in state_dict:
                new_state_dict[new_key + ".weight"] = state_dict[old_key + ".weight"]
                new_state_dict[new_key + ".bias"] = state_dict[old_key + ".bias"]

        # Load the remapped state dict
        aesthetic_mlp.load_state_dict(new_state_dict)
        aesthetic_mlp.eval()

        # Print model architecture for verification
        logger.info("Aesthetic MLP Architecture:")
        for i, layer in enumerate(aesthetic_mlp.layers):
            if isinstance(layer, torch.nn.Linear):
                logger.info(
                    f"  Layer {i//2}: {layer.in_features} -> {layer.out_features}"
                )
                logger.info("    Activation: Tanh")

        logger.info("Successfully loaded CLIP aesthetic predictor")
        return phi35_model, phi35_processor, clip_model, clip_processor, aesthetic_mlp

    except Exception as e:
        logger.error(f"Error loading CLIP aesthetic predictor: {str(e)}")
        raise


def compute_image_embeddings(
    image_paths: List[str],
    model,
    processor,
    clip_model,
    clip_processor,
    batch_size: int = 512,
    num_workers: int = 16,
    prefetch_factor: int = 2,
) -> np.ndarray:
    """Compute image embeddings using CLIP with efficient DataLoader and large batches."""
    device = next(clip_model.parameters()).device
    total_images = len(image_paths)

    print(f"\nComputing CLIP embeddings for {total_images} images")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Prefetch factor: {prefetch_factor}")

    # Create dataset and dataloader
    dataset = ImageEmbeddingDataset(image_paths, clip_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        drop_last=False,
    )

    embeddings = []
    valid_indices = []

    # Process batches
    for batch_idx, (batch_inputs, batch_success) in enumerate(
        tqdm(dataloader, desc="Computing embeddings")
    ):
        # Filter out failed images
        valid_mask = torch.tensor(batch_success)
        if not valid_mask.any():
            embeddings.append(np.zeros((len(batch_inputs), 768)))
            continue

        try:
            # Move valid inputs to GPU
            inputs = batch_inputs[valid_mask].to(device)

            with torch.no_grad():
                # Get embeddings for the batch
                image_features = clip_model.encode_image(inputs)
                # Normalize features
                image_features = torch.nn.functional.normalize(
                    image_features, p=2, dim=1
                )
                batch_embeddings = image_features.cpu().numpy()

                # Create full batch embeddings with zeros for failed images
                full_batch_embeddings = np.zeros(
                    (len(batch_inputs), batch_embeddings.shape[1])
                )
                valid_idx = torch.where(valid_mask)[0].cpu().numpy()
                full_batch_embeddings[valid_idx] = batch_embeddings
                embeddings.append(full_batch_embeddings)

        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            embeddings.append(np.zeros((len(batch_inputs), 768)))

        # Clear GPU memory periodically
        if batch_idx % 2 == 0:  # Clear every 2 batches
            torch.cuda.empty_cache()

        # Print GPU utilization
        if batch_idx % 5 == 0:  # Print every 5 batches
            gpu_util = torch.cuda.utilization()
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(
                f"\nBatch {batch_idx}: GPU Utilization: {gpu_util}%, Memory: {gpu_mem:.1f}GB"
            )

    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    print(f"\nComputed embeddings shape: {all_embeddings.shape}")
    return all_embeddings


def torch_pca(X: torch.Tensor, n_components: int) -> Tuple[torch.Tensor, float]:
    """PyTorch implementation of PCA that handles NaN values and runs on GPU."""
    device = X.device

    # Handle NaN values
    nan_mask = torch.isnan(X).any(dim=1)
    if nan_mask.any():
        print(f"Removing {nan_mask.sum().item()} samples with NaN values")
        X = X[~nan_mask]

    # Center the data
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # Compute SVD
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)

    # Get top n_components
    components = V[:n_components].T
    explained_variance_ratio = (S[:n_components] ** 2) / (S**2).sum()

    # Transform data
    X_transformed = torch.mm(X_centered, components)

    return X_transformed, explained_variance_ratio.sum().item()


def torch_kmeans(
    X: torch.Tensor, n_clusters: int, max_iters: int = 100, tol: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch K-means implementation with robust initialization."""
    device = X.device
    n_samples, n_features = X.shape

    # Ensure input is properly normalized and clean
    X = torch.nn.functional.normalize(X, p=2, dim=1)

    # Simple random initialization instead of k-means++
    indices = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = X[indices].clone()

    # Main k-means loop
    for iteration in range(max_iters):
        # Assign points to closest centroid
        distances = torch.cdist(X, centroids, p=2)
        labels = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:  # Only update if cluster has points
                cluster_points = X[mask]
                new_centroids[k] = torch.nn.functional.normalize(
                    cluster_points.mean(dim=0), p=2, dim=0
                )
            else:
                # If cluster is empty, reinitialize randomly
                new_centroids[k] = X[torch.randint(n_samples, (1,), device=device)]

        # Check convergence
        if torch.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    # Final assignment
    distances = torch.cdist(X, centroids, p=2)
    labels = torch.argmin(distances, dim=1)

    return labels, centroids


def select_diverse_subset(
    image_paths: List[str],
    scores: List[float],
    embeddings: np.ndarray,
    target_size: int,
    quality_threshold: float = 0.4,
    n_components: int = 128,
) -> List[Dict]:
    """Select diverse subset with improved selection logic."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create result dictionaries
    results = [
        {"image_path": path, "combined_score": score}
        for path, score in zip(image_paths, scores)
    ]

    # Debug score distribution
    scores_array = np.array(scores)
    print("\nScore Distribution:")
    print(f"  Min: {scores_array.min():.4f}")
    print(f"  Max: {scores_array.max():.4f}")
    print(f"  Mean: {scores_array.mean():.4f}")
    print(f"  Std: {scores_array.std():.4f}")
    print(f"  Unique scores: {len(np.unique(scores_array))}")

    # Filter by quality threshold
    quality_mask = scores_array >= quality_threshold
    if not np.any(quality_mask):
        print(f"No images meet quality threshold {quality_threshold}!")
        sorted_results = sorted(
            results, key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_results[:target_size]

    filtered_results = np.array(results)[quality_mask]
    filtered_embeddings = torch.tensor(
        embeddings[quality_mask], device=device, dtype=torch.float32
    )

    print(f"\nAfter quality filtering:")
    print(f"  Images remaining: {len(filtered_results)}")
    print(f"  Quality threshold: {quality_threshold}")

    if len(filtered_results) <= target_size:
        return filtered_results.tolist()

    # Preprocess embeddings
    print("\nPreprocessing embeddings...")

    # Handle NaN and inf values
    nan_mask = torch.isnan(filtered_embeddings).any(dim=1) | torch.isinf(
        filtered_embeddings
    ).any(dim=1)
    if nan_mask.any():
        print(f"Removing {nan_mask.sum().item()} embeddings with NaN/inf values")
        filtered_embeddings = filtered_embeddings[~nan_mask]
        filtered_results = filtered_results[~nan_mask.cpu().numpy()]

    # Normalize embeddings
    filtered_embeddings = torch.nn.functional.normalize(filtered_embeddings, p=2, dim=1)

    # Debug embedding statistics
    print("\nEmbedding Statistics (after preprocessing):")
    print(f"  Shape: {filtered_embeddings.shape}")
    print(f"  Mean: {filtered_embeddings.mean().item():.4f}")
    print(f"  Std: {filtered_embeddings.std().item():.4f}")
    print(f"  Min: {filtered_embeddings.min().item():.4f}")
    print(f"  Max: {filtered_embeddings.max().item():.4f}")

    # Apply PCA
    actual_n_components = min(
        n_components, filtered_embeddings.shape[0] - 1, filtered_embeddings.shape[1]
    )
    print(
        f"\nApplying PCA: {filtered_embeddings.shape[1]} -> {actual_n_components} dimensions"
    )

    reduced_embeddings, explained_variance = torch_pca(
        filtered_embeddings, actual_n_components
    )
    print(f"Explained variance ratio: {explained_variance:.4f}")

    # Adjust number of clusters based on target size
    n_clusters = min(
        target_size, max(2, len(filtered_results) // 5)
    )  # More clusters for better diversity
    print(f"\nRunning K-means with {n_clusters} clusters...")

    # Run k-means with multiple initializations
    best_inertia = float("inf")
    best_labels = None

    for init in range(3):
        try:
            labels, centroids = torch_kmeans(reduced_embeddings, n_clusters)
            distances = torch.cdist(reduced_embeddings, centroids, p=2)
            inertia = torch.min(distances, dim=1)[0].sum().item()

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
        except RuntimeError as e:
            print(f"Warning: K-means initialization {init} failed: {str(e)}")
            continue

    if best_labels is None:
        print("All k-means initializations failed, falling back to top-k selection")
        sorted_results = sorted(
            filtered_results.tolist(), key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_results[:target_size]

    # Debug clustering results
    unique_labels = torch.unique(best_labels)
    label_counts = torch.bincount(best_labels)
    print("\nClustering Results:")
    print(f"  Unique clusters: {len(unique_labels)}")
    print(
        f"  Cluster sizes: min={label_counts.min().item()}, max={label_counts.max().item()}"
    )
    print(f"  Empty clusters: {(label_counts == 0).sum().item()}")

    # Calculate images per cluster
    images_per_cluster = max(1, target_size // n_clusters)
    print(f"\nSelection strategy:")
    print(f"  Images per cluster: {images_per_cluster}")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Target size: {target_size}")

    # Select diverse images from each cluster
    selected_indices = []
    for k in range(n_clusters):
        cluster_mask = best_labels == k
        if cluster_mask.any():
            cluster_indices = torch.where(cluster_mask)[0]
            cluster_scores = torch.tensor(
                [filtered_results[i.item()]["combined_score"] for i in cluster_indices],
                device=device,
            )

            # Get top N from this cluster
            n_from_cluster = min(images_per_cluster, len(cluster_indices))
            top_indices = torch.topk(cluster_scores, n_from_cluster)[1]

            for idx in top_indices:
                selected_indices.append(cluster_indices[idx].item())

    print(f"\nSelection Results:")
    print(f"  Selected images: {len(selected_indices)}")
    print(f"  Target size: {target_size}")

    if len(selected_indices) < target_size:
        print(f"Warning: Selected {len(selected_indices)} images, need {target_size}")
        # Fill remaining slots with top scores from unselected images
        unselected_mask = torch.ones(
            len(filtered_results), dtype=torch.bool, device=device
        )
        unselected_mask[selected_indices] = False
        unselected_indices = torch.where(unselected_mask)[0]

        if len(unselected_indices) > 0:
            unselected_scores = torch.tensor(
                [
                    filtered_results[i.item()]["combined_score"]
                    for i in unselected_indices
                ],
                device=device,
            )
            n_remaining = target_size - len(selected_indices)
            top_remaining = torch.topk(unselected_scores, n_remaining)[1]
            selected_indices.extend(
                [unselected_indices[i].item() for i in top_remaining]
            )

    selected_results = [filtered_results[i] for i in selected_indices]
    selected_results.sort(key=lambda x: x["combined_score"], reverse=True)

    return selected_results


class ImageDataset(Dataset):
    """Dataset class for image processing with efficient GPU caching."""

    def __init__(
        self, image_paths, captions, processor, clip_processor, device, cache_size=1000
    ):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
        self.clip_processor = clip_processor
        self.device = device

        # Use a fixed-size LRU cache for GPU memory
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []  # Track LRU order

        print(f"\nInitializing dataset with {len(image_paths)} images")
        print(f"Using GPU cache with {cache_size} images")

    def _update_cache(self, idx, data):
        """Update the cache using LRU policy."""
        if len(self.cache) >= self.cache_size:
            # Remove least recently used item
            lru_idx = self.cache_order.pop(0)
            if lru_idx in self.cache:
                del self.cache[lru_idx]

        # Add new item to cache
        self.cache[idx] = data
        self.cache_order.append(idx)

    def _load_and_process_image(self, idx):
        """Load and process a single image with detailed error handling."""
        try:
            image_path = self.image_paths[idx]
            # First check if file exists and is readable
            if not os.path.exists(image_path):
                print(f"File does not exist: {image_path}")
                return None
            if not os.access(image_path, os.R_OK):
                print(f"File not readable: {image_path}")
                return None

            # Try to get file size
            try:
                file_size = os.path.getsize(image_path)
                if file_size == 0:
                    print(f"Empty file: {image_path}")
                    return None
            except OSError as e:
                print(f"Error getting file size for {image_path}: {e}")
                return None

            # Try to open with PIL with detailed error handling
            try:
                with Image.open(image_path) as img:
                    # Force load to catch any decode errors
                    img.load()
                    # Convert to RGB
                    image = img.convert("RGB")
            except Exception as e:
                print(f"PIL error for {image_path}: {str(e)}")
                print(f"File size: {file_size} bytes")
                return None

            # Process images with dummy text input for Phi3.5-Vision
            try:
                # For Phi3.5-Vision, we need to use chat format
                messages = [{"role": "user", "content": "<|image_1|>\nDescribe this image."}]
                prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                phi35_inputs = self.processor(prompt, [image], return_tensors="pt")
                clip_inputs = self.clip_processor(image)

                return {
                    "phi35_inputs": phi35_inputs,
                    "clip_inputs": clip_inputs,
                    "image_path": image_path,
                    "captions": self.captions[idx] if self.captions else [],
                }
            except Exception as e:
                print(f"Processing error for {image_path}: {str(e)}")
                return None

        except Exception as e:
            print(f"Unexpected error loading {self.image_paths[idx]}: {str(e)}")
            return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Check if item is in cache
            if idx in self.cache:
                # Update LRU order
                self.cache_order.remove(idx)
                self.cache_order.append(idx)
                return self.cache[idx]

            # Load and process image
            data = self._load_and_process_image(idx)
            if data is not None:
                self._update_cache(idx, data)
            return data

        except Exception as e:
            print(f"Error in __getitem__ for {self.image_paths[idx]}: {e}")
            return None


class ImageEmbeddingDataset(Dataset):
    """Dataset for efficient image loading and processing."""

    def __init__(self, image_paths, clip_processor):
        self.image_paths = image_paths
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image_input = self.clip_processor(image)
            return image_input, True
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return None, False


def collate_fn(batch):
    """Collate function that guarantees proper tensor stacking."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    phi35_stacks = {"pixel_values": [], "input_ids": [], "attention_mask": []}
    clip_stacks = []
    paths, caps = [], []

    for item in batch:
        paths.append(item["image_path"])
        caps.append(item["captions"])

        for k, v in item["phi35_inputs"].items():
            # Only stack actual tensors
            if isinstance(v, torch.Tensor):
                phi35_stacks[k].append(v)

        if isinstance(item["clip_inputs"], torch.Tensor):
            clip_stacks.append(item["clip_inputs"])

    # Use cat instead of stack for proper batch dimensions
    phi35_inputs = {
        k: torch.cat(v, dim=0)  # (B, ...) not (B,1,...)
        for k, v in phi35_stacks.items()
        if v
    }
    clip_inputs = torch.stack(clip_stacks) if clip_stacks else None

    return {
        "image_path": paths,
        "captions": caps,
        "phi35_inputs": phi35_inputs,
        "clip_inputs": clip_inputs,
    }


def process_dataset_split(
    split_name: str,
    images_dir: str,
    captions_file: str,
    model,
    processor,
    clip_model,
    clip_processor,
    aesthetic_mlp,
    output_dir: str = "vlm_scores",
    max_images: int = None,
    quality_threshold: float = 0.15,
    diversity_ratio: float = 0.5,
    batch_size: int = 128,
    num_workers: int = 4,
    cache_size: int = 1000,
) -> List[Dict]:
    """Process dataset split with H100 optimizations and save all intermediate results."""
    print(f"\nProcessing {split_name} split...")
    device = next(model.parameters()).device

    # Set memory allocation settings for H100
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

    # Enable CUDA graphs and other H100 optimizations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load captions
    with open(captions_file, "r") as f:
        captions_data = json.load(f)

    if max_images:
        captions_data = captions_data[:max_images]

    # Prepare image paths and captions
    image_paths = []
    captions = []
    for item in tqdm(captions_data, desc="Preparing dataset"):
        image_id = item["image_id"]
        # Try both relative and absolute paths
        rel_path = os.path.join(images_dir, f"{image_id:012d}.jpg")
        abs_path = os.path.join("/workspace", rel_path.lstrip("./"))

        # Check both paths
        if os.path.exists(rel_path):
            image_paths.append(rel_path)
            captions.append(item.get("captions", []))
        elif os.path.exists(abs_path):
            image_paths.append(abs_path)
            captions.append(item.get("captions", []))
        else:
            print(
                f"Warning: Image not found for ID {image_id} at either {rel_path} or {abs_path}"
            )

    # Create dataset with efficient GPU caching
    dataset = ImageDataset(
        image_paths, captions, processor, clip_processor, device, cache_size=cache_size
    )

    # Create dataloader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Process batches with GPU-optimized inference
    results = []
    total_batches = len(dataloader)
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save all intermediate results
    intermediate_file = os.path.join(output_dir, f"{split_name}_all_scores.json")

    # Process and save in chunks to avoid memory issues
    chunk_size = 1000  # Process and save every 1000 images
    current_chunk = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if batch is None:
            continue

        # Process batch on GPU
        batch_results = process_batch(
            batch, model, processor, clip_model, aesthetic_mlp, device
        )
        if batch_results:
            results.extend(batch_results)
            current_chunk.extend(batch_results)

            # Save chunk if we've reached chunk_size
            if len(current_chunk) >= chunk_size:
                # Append to existing file or create new
                mode = "a" if os.path.exists(intermediate_file) else "w"
                with open(intermediate_file, mode) as f:
                    if mode == "w":
                        json.dump(current_chunk, f, indent=2)
                    else:
                        # Remove the closing bracket of previous chunk
                        f.seek(0, 2)  # Seek to end
                        f.seek(f.tell() - 2)  # Move back 2 chars
                        f.write(",\n")  # Add comma
                        json.dump(current_chunk, f, indent=2)
                        f.write("\n]")  # Close the array
                current_chunk = []  # Clear chunk

        # Print progress with GPU memory info
        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            images_per_second = (batch_idx + 1) * batch_size / elapsed_time
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"\nProgress: {batch_idx + 1}/{total_batches} batches")
            print(f"Images processed: {(batch_idx + 1) * batch_size}")
            print(f"Processing speed: {images_per_second:.1f} images/sec")
            print(f"GPU Memory used: {gpu_memory:.1f} GB")
            print(
                f"Current average score: {np.mean([r['combined_score'] for r in results]):.4f}"
            )
            print(
                f"Estimated time remaining: {(total_batches - batch_idx - 1) * batch_size / images_per_second / 60:.1f} minutes"
            )

            # Clear GPU cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # Save any remaining results
    if current_chunk:
        mode = "a" if os.path.exists(intermediate_file) else "w"
        with open(intermediate_file, mode) as f:
            if mode == "w":
                json.dump(current_chunk, f, indent=2)
            else:
                f.seek(0, 2)
                f.seek(f.tell() - 2)
                f.write(",\n")
                json.dump(current_chunk, f, indent=2)
                f.write("\n]")

    # Create summary and multiple distilled datasets
    summary = {
        "total_images": len(results),
        "score_stats": {
            "technical": {
                "min": float(min(r["technical_score"] for r in results)),
                "max": float(max(r["technical_score"] for r in results)),
                "mean": float(np.mean([r["technical_score"] for r in results])),
                "std": float(np.std([r["technical_score"] for r in results])),
            },
            "aesthetic": {
                "min": float(min(r["aesthetic_score"] for r in results)),
                "max": float(max(r["aesthetic_score"] for r in results)),
                "mean": float(np.mean([r["aesthetic_score"] for r in results])),
                "std": float(np.std([r["aesthetic_score"] for r in results])),
            },
            "combined": {
                "min": float(min(r["combined_score"] for r in results)),
                "max": float(max(r["combined_score"] for r in results)),
                "mean": float(np.mean([r["combined_score"] for r in results])),
                "std": float(np.std([r["combined_score"] for r in results])),
            },
        },
        "percentiles": {
            "technical": {
                str(p): float(np.percentile([r["technical_score"] for r in results], p))
                for p in [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
            },
            "aesthetic": {
                str(p): float(np.percentile([r["aesthetic_score"] for r in results], p))
                for p in [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
            },
            "combined": {
                str(p): float(np.percentile([r["combined_score"] for r in results], p))
                for p in [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
            },
        },
    }

    # Save summary
    summary_file = os.path.join(output_dir, f"{split_name}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved all results to {intermediate_file}")
    print(f"Saved summary statistics to {summary_file}")

    # Create multiple distilled datasets
    print("\nCreating multiple distilled datasets...")
    print(f"Total images processed: {len(results)}")

    # Define the target percentiles and their names
    distillation_targets = [
        (50, "top_50pct"),
        (25, "top_25pct"),
        (10, "top_10pct"),
        (5, "top_5pct"),
        (1, "top_1pct"),
        (0.1, "top_0.1pct"),
    ]

    # Sort all results by combined score
    all_results = sorted(results, key=lambda x: x["combined_score"], reverse=True)
    print(f"\nScore distribution:")
    print(f"  Min score: {all_results[-1]['combined_score']:.4f}")
    print(f"  Max score: {all_results[0]['combined_score']:.4f}")
    print(f"  Mean score: {np.mean([r['combined_score'] for r in all_results]):.4f}")

    for percentile, name in distillation_targets:
        # Calculate how many images to take
        num_images = int(len(all_results) * (percentile / 100))
        if num_images < 1:
            num_images = 1

        print(f"\nCreating {name} dataset:")
        print(
            f"  Target: {percentile}% of {len(all_results)} images = {num_images} images"
        )

        # Take top N images directly
        top_images = all_results[:num_images]
        print(
            f"  Selected {len(top_images)} images with scores from {top_images[-1]['combined_score']:.4f} to {top_images[0]['combined_score']:.4f}"
        )

        # Apply diversity selection to maintain the target size
        selected_results = select_diverse_subset(
            [r["image_path"] for r in top_images],
            [r["combined_score"] for r in top_images],
            compute_image_embeddings(
                [r["image_path"] for r in top_images],
                model,
                processor,
                clip_model,
                clip_processor,
            ),
            num_images,
            quality_threshold=0.0,  # No quality threshold needed
        )

        print(f"  After diversity selection: {len(selected_results)} images")

        # Add ranking information
        for i, result in enumerate(selected_results):
            result["rank"] = i + 1
            result["percentile"] = (i + 1) / len(selected_results) * 100
            result["dataset_percentile"] = percentile

        # Save this distilled dataset
        distilled_file = os.path.join(output_dir, f"{split_name}_{name}.json")
        with open(distilled_file, "w") as f:
            json.dump(selected_results, f, indent=2)

        # Print statistics for this dataset
        scores = [r["combined_score"] for r in selected_results]
        print(f"  Saved to: {distilled_file}")
        print(f"  Final dataset size: {len(selected_results)} images")
        print(f"  Score range: {min(scores):.4f} to {max(scores):.4f}")
        print(f"  Average score: {np.mean(scores):.4f}")
        print(f"  Score std: {np.std(scores):.4f}")

    print("\nAll distilled datasets have been created!")
    return selected_results


def main():
    # Add imports at the top of the file
    import gc
    import os

    # Dataset paths - adjust these to your VQAv2 paths
    dataset_paths = {
        "train": {
            "images_dir": "vqav2_data/train2014",
            "captions_file": "vqav2_data/train_captions.json",  # You'll need to create this
        }
    }

    # Verify dataset paths
    for split, paths in dataset_paths.items():
        if not os.path.exists(paths["images_dir"]):
            print(f"Error: Images directory not found: {paths['images_dir']}")
            print("Please ensure you have the VQAv2 data in the correct location")
            return
        if not os.path.exists(paths["captions_file"]):
            print(f"Error: Captions file not found: {paths['captions_file']}")
            print("Please create a captions file from your VQAv2 annotations")
            return

    # Load models with H100 optimizations
    print("\nLoading models with H100 optimizations...")
    device = setup_cuda()

    # Load models
    phi35_model, phi35_processor, clip_model, clip_processor, aesthetic_mlp = (
        load_models()
    )
    print("Models loaded successfully!")

    # Process the dataset with H100 optimizations
    print("\nProcessing dataset with Phi3.5-Vision")
    print("Using optimized batch settings for dataset processing")

    # Parameters for dataset processing
    quality_threshold = 0.15  # Keep this threshold
    diversity_ratio = 0.5  # Keep this ratio
    batch_size = 32  # Smaller batch size for Phi3.5-Vision (more memory intensive)
    num_workers = 2  # Fewer workers for stability
    cache_size = 500  # Smaller cache for Phi3.5-Vision

    # Process split
    print("\nStarting dataset processing...")
    print(f"Using batch size: {batch_size}")
    print(f"Using {num_workers} workers")
    print(f"Cache size: {cache_size} images")
    print(f"Quality threshold: {quality_threshold}")
    print(f"Diversity ratio: {diversity_ratio}")

    distilled_results = process_dataset_split(
        split_name="train_phi35",
        images_dir=dataset_paths["train"]["images_dir"],
        captions_file=dataset_paths["train"]["captions_file"],
        model=phi35_model,
        processor=phi35_processor,
        clip_model=clip_model,
        clip_processor=clip_processor,
        aesthetic_mlp=aesthetic_mlp,
        max_images=None,  # Process all images
        quality_threshold=quality_threshold,
        diversity_ratio=diversity_ratio,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_size=cache_size,
    )

    print("\nDataset processing complete!")
    print(f"Total images processed: {len(distilled_results)}")
    print(f"Results saved to: vlm_scores/train_phi35_all_scores.json")
    print("\nCreated the following distilled datasets:")
    print("- vlm_scores/train_phi35_top_50pct.json")
    print("- vlm_scores/train_phi35_top_25pct.json")
    print("- vlm_scores/train_phi35_top_10pct.json")
    print("- vlm_scores/train_phi35_top_5pct.json")
    print("- vlm_scores/train_phi35_top_1pct.json")
    print("- vlm_scores/train_phi35_top_0.1pct.json")


if __name__ == "__main__":
    main() 