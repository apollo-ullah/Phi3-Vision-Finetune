#!/usr/bin/env python3
"""
Test Hugging Face authentication and basic model functionality.
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os

def test_hf_auth():
    """Test if Hugging Face authentication works."""
    print("🔐 Testing Hugging Face Authentication...")
    
    try:
        # Try to load model without trust_remote_code first
        print("Loading model with trust_remote_code=False...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct",
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model loaded without trust_remote_code!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Trying with trust_remote_code=True...")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3.5-vision-instruct",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("✅ Model loaded with trust_remote_code=True!")
            
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return None, None
    
    try:
        processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct",
            trust_remote_code=True
        )
        print("✅ Processor loaded!")
        
    except Exception as e:
        print(f"❌ Processor failed: {e}")
        return model, None
    
    return model, processor

def test_simple_generation(model, processor):
    """Test basic text generation without images."""
    print("\n📝 Testing Simple Text Generation...")
    
    try:
        # Simple text prompt
        text = "What is the capital of France?"
        inputs = processor.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🎯 Model response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Text generation failed: {e}")
        return False

def test_image_processing(processor):
    """Test if processor can handle images."""
    print("\n🖼️ Testing Image Processing...")
    
    # Create a dummy image
    try:
        from PIL import Image
        import numpy as np
        
        # Create 224x224 RGB image
        dummy_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        # Test basic image processing
        inputs = processor(images=dummy_image, return_tensors="pt")
        print(f"✅ Image processed! Keys: {list(inputs.keys())}")
        
        if 'pixel_values' in inputs:
            print(f"   Pixel values shape: {inputs['pixel_values'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def test_chat_template(processor):
    """Test chat template functionality."""
    print("\n💬 Testing Chat Template...")
    
    try:
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print(f"✅ Chat template works!")
        print(f"   Generated prompt: {repr(prompt[:100])}...")
        return True
        
    except Exception as e:
        print(f"❌ Chat template failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Hugging Face Authentication & Model Test")
    print("=" * 50)
    
    # Test 1: Authentication
    model, processor = test_hf_auth()
    if model is None or processor is None:
        print("\n❌ Critical failure - cannot proceed")
        return
    
    # Test 2: Simple text generation
    text_works = test_simple_generation(model, processor)
    
    # Test 3: Image processing
    image_works = test_image_processing(processor)
    
    # Test 4: Chat template
    chat_works = test_chat_template(processor)
    
    # Summary
    print("\n📊 Test Results Summary:")
    print(f"   🔐 HF Authentication: ✅")
    print(f"   📝 Text Generation: {'✅' if text_works else '❌'}")
    print(f"   🖼️ Image Processing: {'✅' if image_works else '❌'}")
    print(f"   💬 Chat Template: {'✅' if chat_works else '❌'}")
    
    if all([text_works, image_works, chat_works]):
        print("\n🎉 All tests passed! Model should work for vision tasks.")
    else:
        print("\n⚠️ Some tests failed. Check individual components.")

if __name__ == "__main__":
    main() 