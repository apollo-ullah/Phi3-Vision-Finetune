#!/usr/bin/env python3
"""
VQA Baseline Evaluation Script for Phi3.5-Vision
Evaluates model performance on VQAv2 validation set with proper metrics.
"""

import json
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import re
from tqdm import tqdm
from PIL import Image
import torch
import warnings

# Import the repository's utilities
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

warnings.filterwarnings("ignore")

class VQAEvaluator:
    def __init__(self):
        """Initialize VQA evaluator with standard VQA accuracy computation."""
        pass
    
    def evaluate_answer(self, predicted_answer, ground_truth_answers):
        """
        Compute VQA accuracy for a single question.
        VQA accuracy: min(# humans that said the answer / 3, 1)
        """
        predicted_answer = self.normalize_answer(predicted_answer)
        
        # Count how many annotators gave each answer
        gt_answer_counts = Counter()
        for ans_info in ground_truth_answers:
            answer = self.normalize_answer(ans_info['answer'])
            gt_answer_counts[answer] += 1
        
        # VQA accuracy: min(count/3, 1) where count is number of annotators who gave this answer
        if predicted_answer in gt_answer_counts:
            accuracy = min(gt_answer_counts[predicted_answer] / 3.0, 1.0)
        else:
            accuracy = 0.0
        
        return accuracy, predicted_answer, dict(gt_answer_counts)
    
    def normalize_answer(self, answer):
        """Normalize answer according to VQA evaluation protocol with better extraction."""
        answer = answer.lower().strip()
        
        # Extract key answer from verbose responses
        answer = self._extract_key_answer(answer)
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def _extract_key_answer(self, answer):
        """Extract key answer from verbose model responses."""
        answer = answer.lower().strip()
        
        # Yes/No questions - look for yes/no at the beginning
        if answer.startswith('yes'):
            return 'yes'
        elif answer.startswith('no'):
            return 'no'
        elif 'yes,' in answer or 'yes.' in answer:
            return 'yes'
        elif 'no,' in answer or 'no.' in answer:
            return 'no'
        
        # Number extraction - look for standalone numbers
        number_match = re.search(r'\b(\d+)\b', answer)
        if number_match:
            return number_match.group(1)
        
        # Color extraction - common VQA answers
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange', 'pink', 'purple', 'gray', 'grey']
        for color in colors:
            if color in answer:
                return color
        
        # Common single-word answers that might be buried in text
        common_answers = [
            'cat', 'dog', 'bird', 'car', 'truck', 'bus', 'train', 'plane', 'boat',
            'man', 'woman', 'person', 'people', 'child', 'boy', 'girl',
            'chair', 'table', 'bed', 'sofa', 'tv', 'computer', 'phone',
            'food', 'pizza', 'sandwich', 'cake', 'apple', 'banana',
            'left', 'right', 'center', 'middle', 'top', 'bottom',
            'inside', 'outside', 'near', 'far', 'close',
            'christmas', 'birthday', 'halloween', 'easter',
            'sunny', 'cloudy', 'rainy', 'snowy',
            'morning', 'afternoon', 'evening', 'night'
        ]
        
        # Check if any common answer appears in the response
        words = answer.split()
        for word in words:
            if word in common_answers:
                return word
        
        # If no specific extraction, return first few words (up to 3)
        words = answer.split()[:3]
        return ' '.join(words)
    
    def compute_overall_accuracy(self, results):
        """Compute overall VQA accuracy across all questions."""
        if not results:
            return 0.0
        
        total_accuracy = sum(result['accuracy'] for result in results)
        return total_accuracy / len(results)
    
    def compute_answer_type_accuracy(self, results):
        """Compute accuracy by answer type (yes/no, number, other)."""
        answer_type_results = defaultdict(list)
        
        for result in results:
            # Classify answer type based on ground truth
            gt_answers = [ans['answer'].lower().strip() for ans in result['ground_truth']]
            
            if any(ans in ['yes', 'no'] for ans in gt_answers):
                answer_type = 'yes/no'
            elif any(ans.isdigit() or self._is_number(ans) for ans in gt_answers):
                answer_type = 'number'
            else:
                answer_type = 'other'
            
            answer_type_results[answer_type].append(result['accuracy'])
        
        # Compute accuracy for each type
        type_accuracies = {}
        for answer_type, accuracies in answer_type_results.items():
            type_accuracies[answer_type] = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return type_accuracies
    
    def _is_number(self, text):
        """Check if text represents a number."""
        try:
            float(text)
            return True
        except ValueError:
            # Check for written numbers
            number_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
            return text.lower() in number_words


def load_vqa_data(questions_file, annotations_file, images_dir, max_samples=None):
    """Load VQA questions, annotations, and verify image paths."""
    print(f"Loading questions from {questions_file}")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    print(f"Loading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    
    # Create mapping from question_id to annotation
    annotations_map = {ann['question_id']: ann for ann in annotations_data['annotations']}
    
    # Combine questions with annotations
    samples = []
    for question in questions_data['questions']:
        question_id = question['question_id']
        if question_id in annotations_map:
            # Construct image path
            image_filename = f"COCO_val2014_{question['image_id']:012d}.jpg"
            image_path = os.path.join(images_dir, image_filename)
            
            if os.path.exists(image_path):
                sample = {
                    'question_id': question_id,
                    'image_id': question['image_id'],
                    'question': question['question'],
                    'image_path': image_path,
                    'ground_truth': annotations_map[question_id]['answers']
                }
                samples.append(sample)
            else:
                print(f"Warning: Image not found {image_path}")
    
    print(f"Loaded {len(samples)} samples with valid images and annotations")
    
    # Sample subset if requested
    if max_samples and max_samples < len(samples):
        samples = random.sample(samples, max_samples)
        print(f"Using random subset of {max_samples} samples")
    
    return samples


def run_inference(model, processor, image_path, question, device='cuda', max_new_tokens=50):
    """Run inference on a single VQA sample."""
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
                max_new_tokens=20,  # Reduced for shorter answers
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phi3.5-Vision on VQAv2')
    parser.add_argument('--model-path', type=str, default='microsoft/Phi-3.5-vision-instruct')
    parser.add_argument('--questions-file', type=str, default='vqav2_data/v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--annotations-file', type=str, default='vqav2_data/v2_mscoco_val2014_annotations.json')
    parser.add_argument('--images-dir', type=str, default='vqav2_data/val2014')
    parser.add_argument('--max-samples', type=int, default=500, help='Maximum number of samples to evaluate (default: 500)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--disable-flash-attention', action='store_true')
    parser.add_argument('--output-file', type=str, default='baseline_results.json')
    parser.add_argument('--seed', type=int, default=42)
    
    # Handle Jupyter notebook environment
    import sys
    if any('ipykernel_launcher' in str(arg) for arg in sys.argv):
        # Running in Jupyter - filter out Jupyter-specific arguments
        filtered_args = [arg for arg in sys.argv if not arg.startswith('-f') and 'kernel' not in arg]
        # If only script name remains, use defaults
        if len(filtered_args) <= 1:
            args = parser.parse_args([])
        else:
            args = parser.parse_args(filtered_args[1:])
    else:
        args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("PHI3.5-VISION VQA BASELINE EVALUATION")
    print("="*60)
    
    # Load model
    print("Loading model...")
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    use_flash_attn = not args.disable_flash_attention
    
    processor, model = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device_map=args.device,
        load_4bit=False,
        load_8bit=False,
        device=args.device,
        use_flash_attn=use_flash_attn
    )
    
    print(f"✓ Model loaded: {model_name}")
    
    # Load VQA data
    samples = load_vqa_data(args.questions_file, args.annotations_file, args.images_dir, args.max_samples)
    
    # Initialize evaluator
    evaluator = VQAEvaluator()
    
    # Run evaluation
    results = []
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for i, sample in enumerate(tqdm(samples)):
        # Run inference
        predicted_answer = run_inference(
            model, processor, sample['image_path'], sample['question'], args.device
        )
        
        # Evaluate accuracy
        accuracy, normalized_pred, gt_counts = evaluator.evaluate_answer(
            predicted_answer, sample['ground_truth']
        )
        
        result = {
            'question_id': sample['question_id'],
            'question': sample['question'],
            'predicted_answer': predicted_answer,
            'normalized_prediction': normalized_pred,
            'ground_truth': sample['ground_truth'],
            'ground_truth_counts': gt_counts,
            'accuracy': accuracy
        }
        results.append(result)
        
        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            current_acc = evaluator.compute_overall_accuracy(results)
            print(f"Progress: {i+1}/{len(samples)}, Current accuracy: {current_acc:.3f}")
    
    # Compute final metrics
    overall_accuracy = evaluator.compute_overall_accuracy(results)
    answer_type_accuracies = evaluator.compute_answer_type_accuracy(results)
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE EVALUATION RESULTS")
    print("="*60)
    print(f"Overall VQA Accuracy: {overall_accuracy:.3f}")
    print(f"Total samples evaluated: {len(results)}")
    print("\nAccuracy by answer type:")
    for answer_type, accuracy in answer_type_accuracies.items():
        count = sum(1 for r in results if answer_type in str(r))  # Simple count
        print(f"  {answer_type}: {accuracy:.3f}")
    
    # Save detailed results
    detailed_results = {
        'model_path': args.model_path,
        'overall_accuracy': overall_accuracy,
        'answer_type_accuracies': answer_type_accuracies,
        'total_samples': len(results),
        'evaluation_settings': vars(args),
        'detailed_results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {args.output_file}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i, result in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground truth: {[ans['answer'] for ans in result['ground_truth']]}")
        print(f"Accuracy: {result['accuracy']:.3f}")


def evaluate_baseline_vqa(
    model_path='microsoft/Phi-3.5-vision-instruct',
    questions_file='vqav2_data/v2_OpenEnded_mscoco_val2014_questions.json',
    annotations_file='vqav2_data/v2_mscoco_val2014_annotations.json',
    images_dir='vqav2_data/val2014',
    max_samples=500,
    device='cuda',
    disable_flash_attention=False,
    output_file='baseline_results.json',
    seed=42
):
    """
    Function-based interface for VQA evaluation (Jupyter-friendly).
    
    Args:
        model_path: Path to the model
        questions_file: Path to VQA questions file
        annotations_file: Path to VQA annotations file
        images_dir: Directory containing images
        max_samples: Maximum number of samples to evaluate
        device: Device to use for inference
        disable_flash_attention: Whether to disable flash attention
        output_file: Path to save results
        seed: Random seed for reproducibility
    
    Returns:
        dict: Evaluation results
    """
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    print("="*60)
    print("PHI3.5-VISION VQA BASELINE EVALUATION")
    print("="*60)
    
    # Load model
    print("Loading model...")
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
    
    print(f"✓ Model loaded: {model_name}")
    
    # Load VQA data
    samples = load_vqa_data(questions_file, annotations_file, images_dir, max_samples)
    
    # Initialize evaluator
    evaluator = VQAEvaluator()
    
    # Run evaluation
    results = []
    print(f"\nEvaluating on {len(samples)} samples...")
    
    for i, sample in enumerate(tqdm(samples)):
        # Run inference
        predicted_answer = run_inference(
            model, processor, sample['image_path'], sample['question'], device
        )
        
        # Evaluate accuracy
        accuracy, normalized_pred, gt_counts = evaluator.evaluate_answer(
            predicted_answer, sample['ground_truth']
        )
        
        result = {
            'question_id': sample['question_id'],
            'question': sample['question'],
            'predicted_answer': predicted_answer,
            'normalized_prediction': normalized_pred,
            'ground_truth': sample['ground_truth'],
            'ground_truth_counts': gt_counts,
            'accuracy': accuracy
        }
        results.append(result)
        
        # Print progress every 50 samples
        if (i + 1) % 50 == 0:
            current_acc = evaluator.compute_overall_accuracy(results)
            print(f"Progress: {i+1}/{len(samples)}, Current accuracy: {current_acc:.3f}")
    
    # Compute final metrics
    overall_accuracy = evaluator.compute_overall_accuracy(results)
    answer_type_accuracies = evaluator.compute_answer_type_accuracy(results)
    
    # Print results
    print("\n" + "="*60)
    print("BASELINE EVALUATION RESULTS")
    print("="*60)
    print(f"Overall VQA Accuracy: {overall_accuracy:.3f}")
    print(f"Total samples evaluated: {len(results)}")
    print("\nAccuracy by answer type:")
    for answer_type, accuracy in answer_type_accuracies.items():
        count = sum(1 for r in results if answer_type in str(r))  # Simple count
        print(f"  {answer_type}: {accuracy:.3f}")
    
    # Save detailed results
    detailed_results = {
        'model_path': model_path,
        'overall_accuracy': overall_accuracy,
        'answer_type_accuracies': answer_type_accuracies,
        'total_samples': len(results),
        'evaluation_settings': {
            'model_path': model_path,
            'max_samples': max_samples,
            'device': device,
            'disable_flash_attention': disable_flash_attention,
            'seed': seed
        },
        'detailed_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {output_file}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i, result in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Predicted: {result['predicted_answer']}")
        print(f"Ground truth: {[ans['answer'] for ans in result['ground_truth']]}")
        print(f"Accuracy: {result['accuracy']:.3f}")
    
    return detailed_results


if __name__ == "__main__":
    main() 