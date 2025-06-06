#!/usr/bin/env python3
"""
Convert VQAv2 questions and annotations to captions format for ASK-VLM distillation.
This script combines VQA questions with their corresponding image IDs to create
a captions file in the format expected by the distillation script.
"""

import json
import os
from collections import defaultdict
from tqdm import tqdm

def create_captions_from_vqa():
    """Convert VQA data to captions format for distillation."""
    
    # Input files
    vqa_dir = "./vqav2_data"
    questions_file = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json")
    annotations_file = os.path.join(vqa_dir, "v2_mscoco_train2014_annotations.json")
    
    # Output file
    output_file = os.path.join(vqa_dir, "train_captions.json")
    
    print("Loading VQA questions...")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    questions = questions_data['questions']
    print(f"Loaded {len(questions)} questions")
    
    print("Loading VQA annotations...")
    with open(annotations_file, 'r') as f:
        annotations_data = json.load(f)
    annotations = annotations_data['annotations']
    print(f"Loaded {len(annotations)} annotations")
    
    # Group questions by image_id
    image_questions = defaultdict(list)
    question_map = {}
    
    for question in tqdm(questions, desc="Processing questions"):
        image_id = question['image_id']
        question_id = question['question_id']
        question_text = question['question']
        
        image_questions[image_id].append(question_text)
        question_map[question_id] = {
            'image_id': image_id,
            'question': question_text
        }
    
    # Add answer information from annotations
    print("Adding answer information...")
    for annotation in tqdm(annotations, desc="Processing annotations"):
        question_id = annotation['question_id']
        if question_id in question_map:
            # Get the most common answer
            answers = [ans['answer'] for ans in annotation['answers']]
            most_common_answer = max(set(answers), key=answers.count)
            question_map[question_id]['answer'] = most_common_answer
            question_map[question_id]['multiple_choice_answer'] = annotation.get('multiple_choice_answer', '')
    
    # Create captions format
    captions_data = []
    
    print("Creating captions format...")
    for image_id, questions_list in tqdm(image_questions.items(), desc="Creating captions"):
        # Create entry in the format expected by distillation script
        entry = {
            "image_id": image_id,
            "captions": questions_list,  # Use questions as captions
            "num_questions": len(questions_list)
        }
        captions_data.append(entry)
    
    # Sort by image_id for consistency
    captions_data.sort(key=lambda x: x['image_id'])
    
    # Save captions file
    print(f"Saving captions to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"✅ Created {output_file}")
    print(f"   Total images: {len(captions_data)}")
    print(f"   Total questions used as captions: {sum(entry['num_questions'] for entry in captions_data)}")
    
    # Show some examples
    print("\n--- Sample Entries ---")
    for i, entry in enumerate(captions_data[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Image ID: {entry['image_id']}")
        print(f"  Number of captions: {entry['num_questions']}")
        print(f"  Sample captions:")
        for j, caption in enumerate(entry['captions'][:3]):  # Show first 3 captions
            print(f"    {j+1}. {caption}")
        if len(entry['captions']) > 3:
            print(f"    ... and {len(entry['captions']) - 3} more")
    
    return output_file

def verify_captions_file(captions_file):
    """Verify the created captions file has the expected format."""
    print(f"\nVerifying {captions_file}...")
    
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ File loaded successfully")
    print(f"   Total entries: {len(data)}")
    
    # Check format
    required_keys = {'image_id', 'captions'}
    for i, entry in enumerate(data[:5]):  # Check first 5 entries
        missing_keys = required_keys - set(entry.keys())
        if missing_keys:
            print(f"❌ Entry {i} missing keys: {missing_keys}")
            return False
        
        if not isinstance(entry['captions'], list):
            print(f"❌ Entry {i} captions is not a list")
            return False
    
    print("✅ Format verification passed")
    
    # Check image file existence for a few samples
    images_dir = "./vqav2_data/train2014"
    print(f"\nChecking if corresponding images exist in {images_dir}...")
    
    missing_images = 0
    for entry in data[:10]:  # Check first 10 images
        image_id = entry['image_id']
        image_path = os.path.join(images_dir, f"COCO_train2014_{image_id:012d}.jpg")
        if not os.path.exists(image_path):
            missing_images += 1
            print(f"❌ Missing: {image_path}")
    
    if missing_images == 0:
        print("✅ All sample images found")
    else:
        print(f"⚠️  {missing_images}/10 sample images not found")
    
    return True

if __name__ == "__main__":
    print("VQA to Captions Converter for ASK-VLM Distillation")
    print("=" * 55)
    
    # Create captions file
    captions_file = create_captions_from_vqa()
    
    # Verify the created file
    if verify_captions_file(captions_file):
        print(f"\n🎉 SUCCESS! Ready to run ASK-VLM distillation!")
        print(f"   The distillation script can now find: {captions_file}")
        print(f"\nNext steps:")
        print(f"   1. Run: python distillation_phi35_simple.py")
        print(f"   2. The script will use your VQA questions as 'captions' for ASK-VLM evaluation")
    else:
        print(f"\n❌ Verification failed. Please check the output.") 