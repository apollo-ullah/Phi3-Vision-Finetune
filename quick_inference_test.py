#!/usr/bin/env python3
"""
Quick inference test using the repository's own CLI approach.
This should work reliably without generation compatibility issues.
"""

import os
import sys
import subprocess
import json
import random

def test_with_cli_inference():
    """Test using the repository's CLI inference tool"""
    
    # Test data paths
    vqa_data_dir = "./vqav2_data"
    train_questions_file = f"{vqa_data_dir}/v2_OpenEnded_mscoco_train2014_questions.json"
    train_images_dir = f"{vqa_data_dir}/train2014"
    
    if not os.path.exists(train_questions_file):
        print(f"✗ VQA questions file not found: {train_questions_file}")
        return False
        
    if not os.path.exists(train_images_dir):
        print(f"✗ Images directory not found: {train_images_dir}")
        return False
    
    # Load VQA questions
    with open(train_questions_file, 'r') as f:
        vqa_data = json.load(f)
    
    questions = vqa_data['questions']
    print(f"✓ Loaded {len(questions)} questions")
    
    # Test with one sample using CLI
    sample = random.choice(questions)
    image_id = sample['image_id']
    question = sample['question']
    
    # Construct image path
    image_filename = f"COCO_train2014_{image_id:012d}.jpg"
    image_path = os.path.join(train_images_dir, image_filename)
    
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        return False
    
    print(f"\n--- Testing CLI Inference ---")
    print(f"Image: {image_filename}")
    print(f"Question: {question}")
    
    try:
        # Use the repository's CLI inference
        cmd = [
            sys.executable, "-m", "src.serve.cli",
            "--model-path", "microsoft/Phi-3.5-vision-instruct",
            "--image-file", image_path,
            "--question", question
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print(f"✓ CLI inference successful!")
            print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"✗ CLI inference failed:")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ CLI inference timed out")
        return False
    except Exception as e:
        print(f"✗ Error running CLI inference: {e}")
        return False

def test_basic_model_loading():
    """Just test that the model loads without running generation"""
    print("\n--- Testing Model Loading Only ---")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        model_name = "microsoft/Phi-3.5-vision-instruct"
        print(f"Loading {model_name}...")
        
        # Just load the model without doing inference
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print("✓ Model and processor loaded successfully")
        print(f"✓ Model device: {next(model.parameters()).device}")
        print(f"✓ Model dtype: {next(model.parameters()).dtype}")
        
        # Clean up
        del model
        del processor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

if __name__ == "__main__":
    print("Quick Phi3.5-Vision Test")
    print("="*40)
    
    # Test 1: Basic model loading
    loading_success = test_basic_model_loading()
    
    # Test 2: Try CLI inference if available
    if loading_success:
        print("\n" + "="*40)
        cli_success = test_with_cli_inference()
        
        if cli_success:
            print("\n✅ All tests passed! Ready for ASK-VLM implementation.")
        else:
            print("\n⚠️  Model loads but CLI inference needs setup. Proceeding to training test.")
    else:
        print("\n❌ Model loading failed. Check environment.")
    
    print("\nNext steps:")
    print("1. Test training pipeline: bash test_train_vqa.sh")
    print("2. Implement ASK-VLM data distillation") 