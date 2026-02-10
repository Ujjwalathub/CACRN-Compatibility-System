# Fix GPU Detection - Python 3.12 Issue

## Problem Identified ✅

Your system has:
- ✅ NVIDIA RTX 3050 GPU (working)
- ✅ NVIDIA Driver 591.86 (working)
- ❌ Python 3.12.8 (too new for TensorFlow 2.10)
- ❌ TensorFlow 2.20.0 (no Windows GPU support)

**Root Cause**: TensorFlow 2.10 (last version with Windows GPU support) doesn't support Python 3.12.

---

## Solutions

### Option 1: Quick CPU Training (Temporary)

If you just want to train now on CPU while you set up GPU properly:

```bash
# Reinstall TensorFlow (any version will work for CPU)
pip install tensorflow
```

Then train with smaller batch size:
```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 32
```

**Time**: ~30-40 minutes training time

---

### Option 2: Setup GPU with Python 3.11 (Recommended)

For GPU acceleration, you need to create a Python 3.11 environment.

#### Step 1: Install Python 3.11

1. Download Python 3.11.x from: https://www.python.org/downloads/
2. Run installer:
   - ✅ Check "Add Python 3.11 to PATH"
   - ✅ Install for all users
   - Choose custom installation location (e.g., `C:\Python311`)

#### Step 2: Create Virtual Environment

```powershell
# Navigate to your project
cd E:\Model

# Create virtual environment with Python 3.11
C:\Python311\python.exe -m venv venv_gpu

# Activate it
.\venv_gpu\Scripts\Activate.ps1

# Verify Python version
python --version  # Should show Python 3.11.x
```

#### Step 3: Install Dependencies

```powershell
# Install TensorFlow 2.10 (GPU support)
pip install tensorflow==2.10.0

# Install other requirements
pip install scikit-learn pandas numpy sentence-transformers openpyxl matplotlib seaborn
```

#### Step 4: Install CUDA 11.2 and cuDNN 8.1

Since you have CUDA 13.1 (newer), you need CUDA 11.2 for TensorFlow 2.10:

1. **Download CUDA 11.2**:
   - Visit: https://developer.nvidia.com/cuda-11.2.0-download-archive
   - Select: Windows → x86_64 → 10/11 → exe (local)
   - Install alongside your existing CUDA

2. **Download cuDNN 8.1**:
   - Visit: https://developer.nvidia.com/cudnn
   - Download: cuDNN v8.1 for CUDA 11.2
   - Extract and copy files to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`

3. **Update PATH** to use CUDA 11.2:
   ```powershell
   # Add these to system PATH (first in order)
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
   ```

#### Step 5: Verify GPU Setup

```powershell
# Activate environment
.\venv_gpu\Scripts\Activate.ps1

# Verify GPU detection
python verify_gpu.py
```

#### Step 6: Train with GPU

```powershell
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128
```

**Time**: ~5-10 minutes training time (6-8x faster!)

---

### Option 3: Use WSL2 with Ubuntu (Advanced)

If you're comfortable with Linux, WSL2 + Ubuntu supports newer TensorFlow with GPU:

```bash
# In PowerShell (Admin)
wsl --install

# Inside Ubuntu
sudo apt update
sudo apt install python3-pip
pip3 install tensorflow[and-cuda]

# Copy project files and train
```

---

## Quick Decision Guide

| Your Priority | Recommended Option | Setup Time | Training Time |
|---------------|-------------------|------------|---------------|
| **Train NOW (CPU is fine)** | Option 1 | 2 minutes | 30-40 min |
| **Want GPU speed** | Option 2 | 30-60 minutes | 5-10 min |
| **Linux comfortable** | Option 3 | 20-30 minutes | 5-10 min |

---

## Immediate Action (Train on CPU Now)

```powershell
# Reinstall any TensorFlow version for CPU
pip install tensorflow

# Train with CPU-optimized settings
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 32 --epochs 50
```

---

## Permanent Fix Commands (Python 3.11 + GPU)

Copy-paste these commands:

```powershell
# Step 1: Download and install Python 3.11 from python.org
# Then:

cd E:\Model

# Step 2: Create virtual environment
C:\Python311\python.exe -m venv venv_gpu

# Step 3: Activate
.\venv_gpu\Scripts\Activate.ps1

# Step 4: Install TensorFlow 2.10
pip install tensorflow==2.10.0

# Step 5: Install other dependencies
pip install scikit-learn pandas numpy sentence-transformers openpyxl matplotlib seaborn

# Step 6: Install CUDA 11.2 + cuDNN 8.1 (see guide above)

# Step 7: Verify
python verify_gpu.py

# Step 8: Train!
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128
```

---

## Summary

**Current Issue**: Python 3.12 → TensorFlow can't use GPU  
**Quick Fix**: Train on CPU (30-40 min)  
**Proper Fix**: Python 3.11 + TensorFlow 2.10 + CUDA 11.2 → GPU works (5-10 min)  

**Your Choice!** 🚀
