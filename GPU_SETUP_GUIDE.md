# GPU Acceleration Setup Guide

## 🚀 Complete Guide for NVIDIA RTX 3050 (6GB VRAM)

This guide will help you set up GPU acceleration for the CACRN model, achieving **10-20x faster training** compared to CPU.

---

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3050 (6GB VRAM) or better
- **System RAM**: 8GB+ recommended
- **Storage**: 5GB free space (for CUDA/cuDNN)

### Software Requirements
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.8-3.11 (Python 3.12 not recommended yet)
- **NVIDIA Driver**: Version 450.80.02 or higher

---

## 🔧 Installation Steps

### Step 1: Update NVIDIA Drivers

1. **Check current driver version**:
   ```bash
   nvidia-smi
   ```
   - If this command fails, proceed to step 2
   - If it works, note the "Driver Version" number

2. **Update drivers**:
   - Visit: https://www.nvidia.com/drivers
   - Select: GeForce → RTX 30 Series → RTX 3050
   - Download and install the latest Game Ready Driver
   - **Restart your computer** after installation

3. **Verify**:
   ```bash
   nvidia-smi
   ```
   You should see your GPU information displayed

---

### Step 2: Install Compatible TensorFlow (CRITICAL for Windows)

**Important**: TensorFlow 2.11+ dropped native GPU support for Windows. You **must** use TensorFlow 2.10 or lower.

```bash
# Uninstall any existing TensorFlow
pip uninstall tensorflow tensorflow-gpu

# Install TensorFlow 2.10 (last version with Windows GPU support)
pip install "tensorflow<2.11"

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Expected output**: `TensorFlow version: 2.10.x`

---

### Step 3: Install CUDA Toolkit 11.2

**Note**: TensorFlow 2.10 requires CUDA 11.2 specifically.

1. **Download CUDA 11.2**:
   - Visit: https://developer.nvidia.com/cuda-11.2.0-download-archive
   - Select: Windows → x86_64 → 10 (or 11) → exe (local)
   - Download the installer (~3GB)

2. **Install CUDA**:
   - Run the installer as Administrator
   - Choose "Custom" installation
   - **Select these components**:
     - CUDA → Runtime → Libraries
     - CUDA → Development → Headers
     - CUDA → Development → Libraries
     - Integration → Visual Studio Integration (if you have Visual Studio)
   - Default installation path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`

3. **Verify installation**:
   ```bash
   nvcc --version
   ```
   Should display CUDA 11.2 information

---

### Step 4: Install cuDNN 8.1

**Note**: cuDNN 8.1 is compatible with CUDA 11.2 and TensorFlow 2.10.

1. **Download cuDNN**:
   - Visit: https://developer.nvidia.com/cudnn
   - You need to create a free NVIDIA Developer account
   - Download: cuDNN v8.1.x for CUDA 11.2
   - File will be named like: `cudnn-11.2-windows-x64-v8.1.x.zip`

2. **Install cuDNN** (manual extraction):
   ```
   1. Extract the downloaded zip file
   2. You'll see three folders: bin, include, lib
   3. Copy contents of each folder to the corresponding CUDA folder:
   
   From: cudnn-11.2-windows-x64-v8.1.x\bin\*
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin\
   
   From: cudnn-11.2-windows-x64-v8.1.x\include\*
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include\
   
   From: cudnn-11.2-windows-x64-v8.1.x\lib\x64\*
   To:   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64\
   ```

3. **Add to System PATH** (if not already added):
   - Press `Win + X` → System → Advanced system settings
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Add these paths (if not present):
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
     ```
   - Click OK and **restart your computer**

---

### Step 5: Verify GPU Setup

Run the verification script:

```bash
python verify_gpu.py
```

**Expected output for successful setup**:

```
============================================================
[SYSTEM] INITIALIZING GPU ACCELERATION
============================================================
   ✅ SUCCESS: Hardware Accelerator Found: 1 GPU(s)
      - Device Details: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
      - VRAM Management: Dynamic (Memory Growth Enabled)
      - Compute Policy: Mixed Precision (float16) ACTIVE
      - Performance Boost: EXPECTED
```

**If you see warnings**:
- ⚠️ No GPU detected → Check TensorFlow version and CUDA installation
- ❌ CUDA/cuDNN errors → Verify DLL files are in correct locations

---

## 🎮 Using GPU Acceleration

### Automatic GPU Detection

The model **automatically detects and configures GPU** when you run:

```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv
```

You'll see the GPU initialization message at the start.

### Optimized Batch Sizes for RTX 3050 (6GB)

| Batch Size | VRAM Usage | Training Speed | Recommended For |
|------------|------------|----------------|-----------------|
| 32         | ~1GB       | Moderate       | Conservative / Small models |
| 64         | ~2GB       | Good           | Balanced performance |
| **128**    | **~3GB**   | **Excellent**  | **Recommended default** |
| 256        | ~5GB       | Best (if fits) | Maximum performance |

**Start with 128**:
```bash
python main.py --mode train --batch-size 128 --train-profiles data/train.xlsx --target data/target.csv
```

**If you get OOM (Out of Memory) errors**, reduce batch size:
```bash
python main.py --mode train --batch-size 64 --train-profiles data/train.xlsx --target data/target.csv
```

---

## 📊 Performance Monitoring

### Monitor GPU Usage During Training

Open a **separate terminal** and run:

```bash
nvidia-smi -l 1
```

This refreshes every second and shows:
- **GPU Utilization**: Should be 80-100% during training
- **Memory Usage**: 2-4GB with batch_size=128
- **Temperature**: Should stay under 80°C
- **Power Usage**: Varies by load

### Expected Performance Improvements

| Configuration | Time per Epoch | Total Training Time (100 epochs) |
|---------------|----------------|----------------------------------|
| **CPU Only** | 15-20 seconds | ~30-40 minutes |
| **RTX 3050 GPU** | 1-3 seconds | **5-10 minutes** |
| **Speedup** | **10-15x faster** | **6x faster** |

---

## 🧪 Quick Start with GPU

### Option 1: Automated Script (Windows)

```bash
quick_start_gpu.bat
```

This will:
1. Verify GPU setup
2. Run training with GPU-optimized settings
3. Display results

### Option 2: Manual Command

```bash
# Verify GPU
python verify_gpu.py

# Train with GPU acceleration
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128 --epochs 100
```

---

## ⚙️ Advanced Configuration

### Mixed Precision Training

**Automatically enabled** for RTX GPUs. This feature:
- Uses float16 for computations (float32 for outputs)
- **Reduces memory usage by ~40%**
- **Speeds up training by 20-30%**
- Leverages RTX Tensor Cores

You'll see this in the output:
```
- Compute Policy: Mixed Precision (float16) ACTIVE
```

### Dynamic Memory Growth

**Automatically enabled** to prevent TensorFlow from allocating all 6GB VRAM at once. This allows:
- Other applications to use GPU simultaneously
- Gradual VRAM allocation as needed
- Prevention of OOM errors

### Force CPU Mode (for testing)

If you want to temporarily disable GPU:

```python
# Add to top of main.py before GPU setup
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

Or use environment variable:
```bash
set CUDA_VISIBLE_DEVICES=-1
python main.py --mode train ...
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No GPU detected"

**Diagnosis**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**If output is `[]`**, check:
- ✅ TensorFlow version is <2.11 for Windows
- ✅ CUDA 11.2 is installed
- ✅ cuDNN 8.1 is installed
- ✅ NVIDIA drivers are up to date
- ✅ System PATH includes CUDA bin directory

**Solution**:
```bash
pip uninstall tensorflow
pip install "tensorflow==2.10.0"
```

#### 2. "Out of Memory" (OOM) Error

**Symptoms**:
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:
1. **Reduce batch size**:
   ```bash
   python main.py --mode train --batch-size 64 ...
   ```

2. **Close GPU-intensive applications**:
   - Chrome/Firefox (GPU rendering)
   - Games
   - Other ML applications
   
3. **Check current GPU usage**:
   ```bash
   nvidia-smi
   ```

4. **Restart Python and try again** (clears GPU memory)

#### 3. "cudart64_110.dll not found" or similar DLL errors

**Cause**: CUDA DLL files not in PATH or missing

**Solutions**:

1. **Verify CUDA bin in PATH**:
   ```bash
   echo %PATH%
   ```
   Should contain: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`

2. **Download specific DLL** (temporary fix):
   - Search for the DLL file online
   - Place in: `C:\Windows\System32\`

3. **Reinstall CUDA Toolkit** (permanent fix)

#### 4. Training is slow despite GPU detected

**Possible causes**:

1. **Batch size too small**:
   - Solution: Increase to 128 or 256
   
2. **Data loading bottleneck**:
   - Check if CPU usage is maxed out
   - Model already loads all data in memory, so should be fine

3. **GPU not fully utilized**:
   - Run `nvidia-smi -l 1` to check GPU utilization
   - Should be 80-100% during training epochs

#### 5. Mixed Precision Warnings

**Warning messages like**:
```
Mixed precision compatibility warning...
```

**Solution**: These are usually informational. If training loss becomes `nan`:
- Reduce learning rate: `--learning-rate 0.0001`
- Or disable mixed precision (edit main.py)

---

## 📚 Additional Resources

### Official Documentation
- **NVIDIA CUDA**: https://docs.nvidia.com/cuda/
- **cuDNN**: https://docs.nvidia.com/deeplearning/cudnn/
- **TensorFlow GPU**: https://www.tensorflow.org/install/gpu

### Compatibility Matrix

| Component | Required Version |
|-----------|-----------------|
| NVIDIA Driver | 450.80.02+ |
| CUDA Toolkit | 11.2 |
| cuDNN | 8.1.x for CUDA 11.2 |
| TensorFlow | 2.10.x (Windows) |
| Python | 3.8 - 3.11 |

### Community Support
- **TensorFlow Forum**: https://discuss.tensorflow.org/
- **NVIDIA Developer Forums**: https://forums.developer.nvidia.com/

---

## ✅ Setup Checklist

Before running training, verify:

- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] TensorFlow 2.10 installed (`pip list | grep tensorflow`)
- [ ] CUDA 11.2 installed (`nvcc --version`)
- [ ] cuDNN 8.1 files copied to CUDA directory
- [ ] CUDA bin directory in system PATH
- [ ] GPU detected by TensorFlow (`python verify_gpu.py`)
- [ ] Training data in `data/` directory
- [ ] Batch size set appropriately (128 for RTX 3050)

---

## 🎯 Quick Reference

### Essential Commands

```bash
# Verify GPU setup
python verify_gpu.py

# Train with GPU (recommended settings)
python main.py --mode train --batch-size 128 --train-profiles data/train.xlsx --target data/target.csv

# Monitor GPU during training
nvidia-smi -l 1

# Check TensorFlow GPU
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"

# Check CUDA version
nvcc --version
```

### Batch Size Quick Guide

- **RTX 3050 (6GB)**: Start with **128**, try 256 if memory allows
- **Got OOM error?**: Reduce to **64** or **32**
- **Want max speed?**: Try **256** (monitor for OOM)

---

## 🚀 Ready to Train!

Your system is now configured for GPU acceleration. Run:

```bash
python verify_gpu.py          # Final verification
quick_start_gpu.bat           # Automated training (Windows)
```

Or manually:

```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128
```

**Expected result**: Training completes in **5-10 minutes** instead of 30-40 minutes on CPU! 🎉

---

**Need help?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed error solutions.
