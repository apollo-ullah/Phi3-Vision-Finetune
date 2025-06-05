#!/usr/bin/env python3
"""
Convert VQAv2 samples to LLaVA format for training/testing.
Creates a small subset for initial validation of the training pipeline.
"""

import os
import json
import random
from tqdm import tqdm

def convert_vqa_to_llava_sample(num_samples=100):
    """Convert a small subset of VQAv2 to LLaVA format"""
    
    vqa_data_dir = "./vqav2_data"
    
    # Load questions and annotations
    train_questions_file = f"{vqa_data_dir}/v2_OpenEnded_mscoco_train2014_questions.json"
    train_annotations_file = f"{vqa_data_dir}/v2_mscoco_train2014_annotations.json"
    
    print("Loading VQAv2 data...")
    
    # Load questions
    with open(train_questions_file, 'r') as f:
        questions_data = json.load(f)
    questions = questions_data['questions']
    
    # Load annotations
    with open(train_annotations_file, 'r') as f:
        annotations_data = json.load(f)
    annotations = annotations_data['annotations']
    
    print(f"Loaded {len(questions)} questions and {len(annotations)} annotations")
    
    # Create question-answer mapping
    qa_mapping = {}
    for ann in annotations:
        qa_mapping[ann['question_id']] = {
            'answers': [ans['answer'] for ans in ann['answers']],
            'multiple_choice_answer': ann['multiple_choice_answer']
        }
    
    # Sample questions with valid answers
    valid_questions = [q for q in questions if q['question_id'] in qa_mapping]
    sample_questions = random.sample(valid_questions, min(num_samples, len(valid_questions)))
    
    print(f"Converting {len(sample_questions)} samples to LLaVA format...")
    
    # Convert to LLaVA format
    llava_format = []
    
    for q in tqdm(sample_questions, desc="Converting"):
        image_id = q['image_id']
        question_id = q['question_id']
        question_text = q['question']
        
        # COCO image filename format
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        
        # Get answer
        if question_id in qa_mapping:
            answer = qa_mapping[question_id]['multiple_choice_answer']
            
            entry = {
                "id": str(question_id),
                "image": image_filename,  # Relative to image folder
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question_text}"
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            llava_format.append(entry)
    
    # Save sample dataset
    output_file = f"{vqa_data_dir}/vqa_llava_sample_{num_samples}.json"
    with open(output_file, 'w') as f:
        json.dump(llava_format, f, indent=2)
    
    print(f"✓ Saved {len(llava_format)} samples to {output_file}")
    
    # Show some examples
    print("\n--- Sample Entries ---")
    for i, entry in enumerate(llava_format[:3], 1):
        print(f"\nSample {i}:")
        print(f"ID: {entry['id']}")
        print(f"Image: {entry['image']}")
        print(f"Question: {entry['conversations'][0]['value']}")
        print(f"Answer: {entry['conversations'][1]['value']}")
    
    return output_file

def create_test_training_script(sample_file):
    """Create a test training script using the sample data"""
    
    script_content = f"""#!/bin/bash

# Test training script for Phi3.5-Vision with VQAv2 sample
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \\
    --deepspeed scripts/zero2.json \\
    --model_id $MODEL_NAME \\
    --data_path {sample_file} \\
    --image_folder ./vqav2_data/train2014 \\
    --tune_img_projector True \\
    --freeze_vision_tower False \\
    --freeze_llm False \\
    --bf16 True \\
    --fp16 False \\
    --disable_flash_attn2 False \\
    --output_dir output/vqa_test_train \\
    --num_crops 4 \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 2 \\
    --gradient_accumulation_steps 2 \\
    --learning_rate 1e-5 \\
    --projector_lr 1e-5 \\
    --vision_lr 1e-6 \\
    --weight_decay 0. \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type "cosine" \\
    --logging_steps 1 \\
    --tf32 True \\
    --gradient_checkpointing True \\
    --report_to tensorboard \\
    --logging_dir ./logs/vqa_test \\
    --lazy_preprocess True \\
    --dataloader_num_workers 2 \\
    --save_steps 50 \\
    --eval_steps 50
"""
    
    script_path = "test_train_vqa.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✓ Created test training script: {script_path}")
    return script_path

if __name__ == "__main__":
    print("VQAv2 to LLaVA Format Converter")
    print("="*40)
    
    # Check if VQA data exists
    if not os.path.exists("./vqav2_data"):
        print("❌ VQAv2 data not found. Please run dataset.py first.")
        exit(1)
    
    # Convert sample
    sample_file = convert_vqa_to_llava_sample(num_samples=100)
    
    # Create test training script
    test_script = create_test_training_script(sample_file)
    
    print(f"\n✅ Ready for testing!")
    print(f"1. Run inference test: python test_phi35v_inference.py")
    print(f"2. Run training test: bash {test_script}")
    print(f"3. After validating these work, implement ASK-VLM data distillation") 

    # %%