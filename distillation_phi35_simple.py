import os
import json
import sys
import torch
from PIL import Image
from tqdm import tqdm
import logging
import numpy as np
from typing import List, Tuple, Dict
import time
import gc
from torch.utils.data import Dataset, DataLoader

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


def get_vlm_relevance_probability(
    model, processor, image_path, question: str = "Is this image good quality and suitable for VLM training?"
) -> Tuple[float, Dict[str, float]]:
    """
    Get the probability that the VLM thinks this datapoint is relevant for training.
    This is the core of ASK-VLM - asking the model directly.

    Args:
        model: The Phi3.5-Vision model
        processor: The model's processor
        image_path: Path to the image
        question: The ASK-VLM question to ask

    Returns:
        tuple: (yes_probability, debug_info)
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare messages in Phi3.5-Vision format with ASK-VLM question
        messages = [{"role": "user", "content": f"<|image_1|>\n{question} Answer yes or no."}]
        
        # Apply chat template
        prompt_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process inputs
        inputs = processor(prompt_text, [image], return_tensors="pt").to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
            next_token_logits = outputs.logits[0, -1]

            # Get token IDs for "yes" and "no"
            yes_token_id = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
            no_token_id = processor.tokenizer.encode("no", add_special_tokens=False)[0]

            # Get logits for yes/no tokens
            yes_logit = next_token_logits[yes_token_id].item()
            no_logit = next_token_logits[no_token_id].item()

            # Get probabilities
            probs = torch.softmax(
                next_token_logits[[yes_token_id, no_token_id]], dim=-1
            )
            yes_prob = probs[0].item()
            no_prob = probs[1].item()

            debug_info = {
                "yes_logit": yes_logit,
                "no_logit": no_logit,
                "yes_prob": yes_prob,
                "no_prob": no_prob,
                "question": question
            }

            return yes_prob, debug_info

    except Exception as e:
        print(f"Error in VLM relevance scoring for {image_path}: {str(e)}")
        return 0.0, {"error": str(e)}


def process_batch_simple(batch, model, processor, ask_question, device):
    """Process batch with simple ASK-VLM scoring using Phi3.5-Vision."""
    if batch is None:
        return []

    try:
        results = []
        
        # Process each image in the batch individually for now (can be optimized later)
        for i, image_path in enumerate(batch["image_path"]):
            # Get ASK-VLM relevance probability
            relevance_prob, debug_info = get_vlm_relevance_probability(
                model, processor, image_path, ask_question
            )
            
            result = {
                "image_path": image_path,
                "relevance_probability": float(relevance_prob),
                "relevance_score": float(relevance_prob),  # Same as probability for simplicity
                "captions": batch["captions"][i] if batch["captions"] else [],
                "debug_info": debug_info
            }
            results.append(result)
            
            # Print progress for first few
            if i < 3:
                print(f"  Image {i}: {image_path}")
                print(f"    Relevance probability: {relevance_prob:.4f}")
                print(f"    Question: {ask_question}")

        print(f"\nSimple ASK-VLM Batch Results:")
        relevance_probs = [r["relevance_probability"] for r in results]
        print(f"  Relevance probabilities: {min(relevance_probs):.3f} to {max(relevance_probs):.3f}")
        print(f"  Average relevance: {np.mean(relevance_probs):.3f}")
        print(f"  Std: {np.std(relevance_probs):.3f}")

        return results

    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return []


class SimpleImageDataset(Dataset):
    """Simple dataset class for image processing without CLIP."""

    def __init__(self, image_paths, captions):
        self.image_paths = image_paths
        self.captions = captions

        print(f"\nInitializing simple dataset with {len(image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            
            # Just return the path and captions - no preprocessing needed
            return {
                "image_path": image_path,
                "captions": self.captions[idx] if self.captions else [],
            }

        except Exception as e:
            print(f"Error in __getitem__ for {self.image_paths[idx]}: {e}")
            return None


def simple_collate_fn(batch):
    """Simple collate function that just groups paths and captions."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    paths = [item["image_path"] for item in batch]
    caps = [item["captions"] for item in batch]

    return {
        "image_path": paths,
        "captions": caps,
    }


def process_dataset_simple(
    split_name: str,
    images_dir: str,
    captions_file: str,
    model,
    processor,
    ask_question: str = "Is this image good quality and suitable for VLM training?",
    output_dir: str = "ask_vlm_scores",
    max_images: int = None,
    batch_size: int = 16,  # Smaller since we're doing individual inference
    num_workers: int = 2,
) -> List[Dict]:
    """Process dataset with simple ASK-VLM scoring."""
    print(f"\nProcessing {split_name} split with ASK-VLM...")
    print(f"ASK Question: '{ask_question}'")
    
    device = next(model.parameters()).device

    # Enable optimizations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = True

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
        # Fix: Use correct COCO filename format
        rel_path = os.path.join(images_dir, f"COCO_train2014_{image_id:012d}.jpg")
        
        # Check if image exists
        if os.path.exists(rel_path):
            image_paths.append(rel_path)
            captions.append(item.get("captions", []))
        else:
            print(f"Warning: Image not found for ID {image_id} at {rel_path}")

    print(f"Found {len(image_paths)} valid images")

    # Create simple dataset
    dataset = SimpleImageDataset(image_paths, captions)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,  # Not needed for simple case
        num_workers=num_workers,
        collate_fn=simple_collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Process batches
    results = []
    total_batches = len(dataloader)
    start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {total_batches} batches...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        if batch is None:
            continue

        # Process batch
        batch_results = process_batch_simple(
            batch, model, processor, ask_question, device
        )
        if batch_results:
            results.extend(batch_results)

        # Print progress
        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            images_per_second = (batch_idx + 1) * batch_size / elapsed_time
            gpu_memory = torch.cuda.memory_allocated() / 1e9
            print(f"\nProgress: {batch_idx + 1}/{total_batches} batches")
            print(f"Images processed: {len(results)}")
            print(f"Processing speed: {images_per_second:.1f} images/sec")
            print(f"GPU Memory used: {gpu_memory:.1f} GB")
            if results:
                avg_relevance = np.mean([r['relevance_probability'] for r in results])
                print(f"Current average relevance: {avg_relevance:.4f}")

            # Clear GPU cache periodically
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    # Create summary
    if results:
        relevance_scores = [r["relevance_probability"] for r in results]
        summary = {
            "total_images": len(results),
            "ask_question": ask_question,
            "relevance_stats": {
                "min": float(min(relevance_scores)),
                "max": float(max(relevance_scores)),
                "mean": float(np.mean(relevance_scores)),
                "std": float(np.std(relevance_scores)),
            },
            "percentiles": {
                str(p): float(np.percentile(relevance_scores, p))
                for p in [1, 5, 10, 20, 25, 50, 75, 80, 90, 95, 99]
            },
        }

        # Save all results
        all_results_file = os.path.join(output_dir, f"{split_name}_ask_vlm_results.json")
        with open(all_results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_file = os.path.join(output_dir, f"{split_name}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved all results to {all_results_file}")
        print(f"Saved summary to {summary_file}")

        # Print summary
        print(f"\nASK-VLM Results Summary:")
        print(f"Total images processed: {summary['total_images']}")
        print(f"Question asked: '{ask_question}'")
        print(f"Relevance probability range: {summary['relevance_stats']['min']:.4f} to {summary['relevance_stats']['max']:.4f}")
        print(f"Average relevance: {summary['relevance_stats']['mean']:.4f}")
        print(f"Standard deviation: {summary['relevance_stats']['std']:.4f}")

        # Create distilled datasets based on relevance probability thresholds
        print(f"\nCreating distilled datasets based on relevance probability...")
        
        # Sort by relevance probability
        sorted_results = sorted(results, key=lambda x: x["relevance_probability"], reverse=True)
        
        # Define probability thresholds
        thresholds = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        for threshold in thresholds:
            filtered_results = [r for r in sorted_results if r["relevance_probability"] >= threshold]
            
            if filtered_results:
                # Add ranking information
                for i, result in enumerate(filtered_results):
                    result["rank"] = i + 1
                    result["percentile"] = (i + 1) / len(filtered_results) * 100
                    result["threshold"] = threshold

                # Save distilled dataset
                distilled_file = os.path.join(output_dir, f"{split_name}_threshold_{threshold:.1f}.json")
                with open(distilled_file, "w") as f:
                    json.dump(filtered_results, f, indent=2)

                print(f"  Threshold {threshold:.1f}: {len(filtered_results)} images saved to {distilled_file}")
            else:
                print(f"  Threshold {threshold:.1f}: No images meet this threshold")

        # Also create top-k datasets
        top_k_values = [1000, 5000, 10000, 25000, 50000]
        
        for k in top_k_values:
            if k <= len(sorted_results):
                top_k_results = sorted_results[:k]
                
                # Add ranking information
                for i, result in enumerate(top_k_results):
                    result["rank"] = i + 1
                    result["percentile"] = (i + 1) / len(top_k_results) * 100
                    result["top_k"] = k

                # Save top-k dataset
                topk_file = os.path.join(output_dir, f"{split_name}_top_{k}.json")
                with open(topk_file, "w") as f:
                    json.dump(top_k_results, f, indent=2)

                min_prob = top_k_results[-1]["relevance_probability"]
                avg_prob = np.mean([r["relevance_probability"] for r in top_k_results])
                print(f"  Top {k}: min_prob={min_prob:.4f}, avg_prob={avg_prob:.4f}, saved to {topk_file}")

    return results


def main():
    # Dataset paths - adjust these to your VQAv2 paths
    dataset_paths = {
        "train": {
            "images_dir": "vqav2_data/train2014",
            "captions_file": "vqav2_data/train_captions.json",  # You'll need to create this
        }
    }

    # ASK-VLM Questions - experiment with different questions
    ask_questions = [
        "Is this image good quality and suitable for VLM training?",
        "Is this image clear, well-composed, and informative?",
        "Would this image be useful for training a vision-language model?",
        "Is this a high-quality, educational image?",
    ]

    # Choose which question to use
    selected_question = ask_questions[0]  # Change index to try different questions

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

    # Load Phi3.5-Vision model
    print("\nLoading Phi3.5-Vision model...")
    phi35_model, phi35_processor = load_phi35_model()
    print("Model loaded successfully!")

    # Process the dataset with ASK-VLM
    print(f"\nStarting ASK-VLM processing...")
    print(f"Question: '{selected_question}'")

    # Parameters for processing
    batch_size = 8   # Small batch size for individual inference
    num_workers = 1  # Keep simple for debugging
    max_images = 1000  # Start small for testing

    results = process_dataset_simple(
        split_name="train_ask_vlm",
        images_dir=dataset_paths["train"]["images_dir"],
        captions_file=dataset_paths["train"]["captions_file"],
        model=phi35_model,
        processor=phi35_processor,
        ask_question=selected_question,
        max_images=max_images,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"\nASK-VLM processing complete!")
    print(f"Total images processed: {len(results)}")
    print(f"Results saved to: ask_vlm_scores/")
    
    if results:
        avg_relevance = np.mean([r['relevance_probability'] for r in results])
        print(f"Average relevance probability: {avg_relevance:.4f}")
        
        # Show some examples
        sorted_results = sorted(results, key=lambda x: x["relevance_probability"], reverse=True)
        print(f"\nTop 5 most relevant images:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['image_path']}: {result['relevance_probability']:.4f}")
        
        print(f"\nBottom 5 least relevant images:")
        for i, result in enumerate(sorted_results[-5:]):
            print(f"  {i+1}. {result['image_path']}: {result['relevance_probability']:.4f}")


if __name__ == "__main__":
    main() 