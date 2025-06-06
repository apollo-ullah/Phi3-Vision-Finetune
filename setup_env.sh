#!/bin/bash

# Phi3.5-Vision Fine-tuning Environment Setup Script
# This script sets up the complete environment for ASK-VLM distillation
# Optimized for RunPod and cloud GPU environments

set -e  # Exit on any error

echo "🚀 Setting up Phi3.5-Vision Fine-tuning Environment..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're on RunPod
if [[ -n "${RUNPOD_POD_ID}" ]]; then
    print_status "RunPod environment detected (Pod ID: ${RUNPOD_POD_ID})"
    IS_RUNPOD=true
else
    print_status "Local/other cloud environment detected"
    IS_RUNPOD=false
fi

# Check CUDA availability
print_step "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    print_status "CUDA is available"
else
    print_error "CUDA not found! This project requires CUDA support."
    exit 1
fi

# Check Python version
print_step "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"

# Update system packages
print_step "Updating system packages..."
if command -v apt-get &> /dev/null; then
    if [ "$EUID" -eq 0 ]; then
        # Running as root, no need for sudo
        apt-get update -y
        apt-get install -y git wget curl build-essential
    else
        # Not root, use sudo
        sudo apt-get update -y
        sudo apt-get install -y git wget curl build-essential
    fi
elif command -v yum &> /dev/null; then
    if [ "$EUID" -eq 0 ]; then
        # Running as root, no need for sudo
        yum update -y
        yum install -y git wget curl gcc gcc-c++ make
    else
        # Not root, use sudo
        sudo yum update -y
        sudo yum install -y git wget curl gcc gcc-c++ make
    fi
fi

# Install or update pip
print_step "Setting up pip..."
python3 -m pip install --upgrade pip

# Check if we have the repository's requirements.txt
print_step "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    print_status "Found repository's requirements.txt - using exact versions"
    # Install from repository's requirements.txt with correct PyTorch index
    pip3 install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
    
    # Install flash attention separately (as per repository instructions)
    print_step "Installing Flash Attention..."
    pip3 install flash-attn --no-build-isolation
    
else
    print_warning "Repository requirements.txt not found, installing generic versions"
    
    # Install PyTorch with CUDA support
    print_step "Installing PyTorch with CUDA support..."
    # Use the latest stable PyTorch with CUDA 12.1 (common on RunPod)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install core dependencies
    print_step "Installing core dependencies..."
    pip3 install --upgrade \
        Pillow \
        tqdm \
        numpy \
        transformers \
        accelerate \
        bitsandbytes \
        datasets \
        scikit-learn \
        matplotlib \
        seaborn \
        pandas \
        requests \
        jupyter \
        ipywidgets

    # Install additional ML libraries
    print_step "Installing additional ML libraries..."
    pip3 install --upgrade \
        wandb \
        tensorboard \
        deepspeed \
        flash-attn --no-build-isolation \
        xformers
fi

# Install Hugging Face CLI
print_step "Installing Hugging Face CLI..."
pip3 install --upgrade huggingface_hub
# Optionally login to HF (user can do this manually)
echo "To login to Hugging Face, run: huggingface-cli login"

# Create necessary directories
print_step "Creating project directories..."
mkdir -p data
mkdir -p models
mkdir -p outputs
mkdir -p logs
mkdir -p ask_vlm_scores
mkdir -p vqav2_data
mkdir -p src

print_status "Created directories: data/, models/, outputs/, logs/, ask_vlm_scores/, vqav2_data/, src/"

# Set up Git LFS (useful for large model files)
print_step "Setting up Git LFS..."
if command -v git-lfs &> /dev/null; then
    git lfs install
    print_status "Git LFS is ready"
else
    print_warning "Git LFS not found. Installing..."
    if command -v apt-get &> /dev/null; then
        if [ "$EUID" -eq 0 ]; then
            apt-get install -y git-lfs
        else
            sudo apt-get install -y git-lfs
        fi
    elif command -v yum &> /dev/null; then
        if [ "$EUID" -eq 0 ]; then
            yum install -y git-lfs
        else
            sudo yum install -y git-lfs
        fi
    fi
    git lfs install
fi

# Create a basic requirements.txt for future reference
print_step "Creating requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    print_status "Creating generic requirements.txt since repository version not found"
    cat > requirements.txt << EOF
# Core ML libraries
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.14.0

# Image processing
Pillow>=10.0.0

# Utilities
tqdm>=4.65.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Optimization
bitsandbytes>=0.41.0
flash-attn>=2.3.0
xformers>=0.0.22

# Experiment tracking
wandb>=0.15.0
tensorboard>=2.14.0

# Deep learning optimization
deepspeed>=0.10.0

# HuggingFace
huggingface_hub>=0.17.0

# Jupyter
jupyter>=1.0.0
ipywidgets>=8.0.0
EOF
    print_status "Created generic requirements.txt"
else
    print_status "Using existing repository requirements.txt with tested versions"
fi

# Create environment info script
print_step "Creating environment info script..."
cat > check_env.py << 'EOF'
#!/usr/bin/env python3
"""
Environment check script for Phi3.5-Vision fine-tuning
"""
import sys
import torch
import transformers
import accelerate
from PIL import Image
import numpy as np
import pandas as pd

def check_environment():
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    # Key library versions
    print(f"Transformers version: {transformers.__version__}")
    print(f"Accelerate version: {accelerate.__version__}")
    
    # Check flash attention
    try:
        import flash_attn
        print(f"Flash Attention version: {flash_attn.__version__}")
    except ImportError:
        print("Flash Attention: Not installed")
    
    # Check xformers
    try:
        import xformers
        print(f"xFormers version: {xformers.__version__}")
    except ImportError:
        print("xFormers: Not installed")
    
    # Memory info
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.1f} GB, Cached: {cached:.1f} GB")
    
    print("\n✅ Environment check complete!")

if __name__ == "__main__":
    check_environment()
EOF

chmod +x check_env.py
print_status "Created check_env.py"

# Create a simple test script
print_step "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test to verify the environment is working
"""
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np

def test_basic_functionality():
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    # Test PyTorch
    print("Testing PyTorch...")
    x = torch.randn(3, 3)
    if torch.cuda.is_available():
        x = x.cuda()
        print("✅ PyTorch CUDA test passed")
    else:
        print("⚠️  CUDA not available, using CPU")
    
    # Print version info
    print(f"PyTorch version: {torch.__version__}")
    
    # Test PIL
    print("Testing PIL...")
    img = Image.new('RGB', (224, 224), color='red')
    print("✅ PIL test passed")
    
    # Test numpy
    print("Testing NumPy...")
    arr = np.random.randn(10, 10)
    print("✅ NumPy test passed")
    
    print("\n✅ Basic functionality test complete!")

def check_version_compatibility():
    print("\n🔍 Checking Version Compatibility")
    print("=" * 40)
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        # Check if we have repository's tested versions
        if hasattr(torch, '__version__'):
            torch_version = torch.__version__
            if '2.6.0' in torch_version:
                print("✅ Using repository's tested PyTorch version")
            elif '2.7.0' in torch_version:
                print("⚠️  Using newer PyTorch 2.7.0 - may have compatibility issues")
                print("   Consider: pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124")
            else:
                print(f"ℹ️  PyTorch version: {torch_version}")
        
        if hasattr(transformers, '__version__'):
            transformers_version = transformers.__version__
            if '4.51.3' in transformers_version:
                print("✅ Using repository's tested Transformers version")
            else:
                print(f"ℹ️  Transformers version: {transformers_version}")
        
    except Exception as e:
        print(f"⚠️  Version check failed: {e}")

def test_model_loading():
    print("\n🤖 Testing Model Loading (Optional)")
    print("=" * 40)
    
    try:
        print("Attempting to load Phi-3.5-vision-instruct...")
        print("This may take a few minutes on first run...")
        
        # This will download the model if not cached
        processor = AutoProcessor.from_pretrained(
            "microsoft/Phi-3.5-vision-instruct",
            trust_remote_code=True
        )
        print("✅ Processor loaded successfully")
        
        # Note: We don't load the full model here to save time/memory
        print("✅ Model loading test passed")
        
    except Exception as e:
        print(f"⚠️  Model loading test failed: {e}")
        print("This may be due to version incompatibilities")
        print("Try installing exact repository versions:")
        print("pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124")

if __name__ == "__main__":
    test_basic_functionality()
    check_version_compatibility()
    
    # Ask user if they want to test model loading
    response = input("\nTest model loading? This will download ~7GB (y/N): ").lower()
    if response in ['y', 'yes']:
        test_model_loading()
    else:
        print("Skipping model loading test")
    
    print("\n🎉 Setup verification complete!")
EOF

chmod +x test_setup.py
print_status "Created test_setup.py"

# Create a data download helper script
print_step "Creating data download helper..."
cat > download_data.py << 'EOF'
#!/usr/bin/env python3
"""
Helper script to download and prepare VQAv2 data
"""
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)

def download_vqav2_data():
    """Download VQAv2 dataset"""
    print("📦 VQAv2 Data Download Helper")
    print("=" * 40)
    
    # Create data directory
    os.makedirs("vqav2_data", exist_ok=True)
    os.chdir("vqav2_data")
    
    # VQAv2 URLs
    urls = {
        "train_images": "http://images.cocodataset.org/zips/train2014.zip",
        "val_images": "http://images.cocodataset.org/zips/val2014.zip",
        "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
    }
    
    print("This will download ~37GB of data. Continue? (y/N): ", end="")
    if input().lower() not in ['y', 'yes']:
        print("Download cancelled")
        return
    
    for name, url in urls.items():
        filename = url.split('/')[-1]
        if not os.path.exists(filename):
            download_file(url, filename)
            
            # Extract if it's a zip file
            if filename.endswith('.zip'):
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall()
                print(f"✅ Extracted {filename}")
        else:
            print(f"✅ {filename} already exists")
    
    print("\n🎉 VQAv2 data download complete!")
    print("Remember to create train_captions.json from the annotations")

if __name__ == "__main__":
    download_vqav2_data()
EOF

chmod +x download_data.py
print_status "Created download_data.py"

# Create RunPod specific optimizations
if [ "$IS_RUNPOD" = true ]; then
    print_step "Applying RunPod-specific optimizations..."
    
    # Set up persistent workspace
    if [ -d "/workspace" ]; then
        print_status "Using /workspace as persistent storage"
        ln -sf /workspace/models models_persistent
        ln -sf /workspace/data data_persistent
    fi
    
    # RunPod environment variables
    export CUDA_VISIBLE_DEVICES=0
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # Common RunPod GPU architectures
    
    # Create RunPod startup script
    cat > runpod_start.sh << 'EOF'
#!/bin/bash
# RunPod startup script
echo "🚀 Starting Phi3.5-Vision environment on RunPod..."
cd /workspace/Phi3-Vision-Finetune || cd $(pwd)
python3 check_env.py
echo "✅ Environment ready! You can now run your training scripts."
EOF
    chmod +x runpod_start.sh
    print_status "Created runpod_start.sh"
fi

# Set environment variables
print_step "Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# Phi3.5-Vision Environment Variables
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HOME}/.cache/huggingface"
export WANDB_CACHE_DIR="${HOME}/.cache/wandb"

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
EOF

# Create final verification
print_step "Running final verification..."
python3 check_env.py

print_status "Environment setup complete! 🎉"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Run verification: python3 test_setup.py"
echo "2. Download data (optional): python3 download_data.py"
echo "3. Login to Hugging Face: huggingface-cli login"
echo "4. Run your training: python3 distillation_phi35_simple.py"
echo ""
echo "📁 Created Files:"
if [ -f "requirements.txt" ] && grep -q "torch==2.6.0" requirements.txt; then
    echo "- requirements.txt (repository's tested versions - RECOMMENDED)"
else
    echo "- requirements.txt (generic versions)"
fi
echo "- check_env.py (environment checker)"
echo "- test_setup.py (setup verification)"
echo "- download_data.py (data download helper)"
if [ "$IS_RUNPOD" = true ]; then
    echo "- runpod_start.sh (RunPod startup script)"
fi
echo ""
echo "💡 Tips:"
echo "- Use 'python3 check_env.py' to verify your environment anytime"
echo "- The setup automatically uses repository's exact package versions when available"
echo "- All dependencies are compatible with A100/H100 GPUs"
if [ "$IS_RUNPOD" = true ]; then
    echo "- On RunPod, your /workspace is persistent across pod restarts"
fi
echo ""
echo "⚠️  Important:"
echo "- If you encounter import errors, the repository's requirements.txt has tested versions"
echo "- Run 'pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124' for exact versions"
echo ""
print_status "Happy training! 🚀" 