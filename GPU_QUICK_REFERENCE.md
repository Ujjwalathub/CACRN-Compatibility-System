# GPU Acceleration - Quick Reference Card

## 🚀 Essential Commands

### Verify GPU Setup
```bash
python verify_gpu.py
```

### Train with GPU (Recommended)
```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128
```

### Monitor GPU Usage
```bash
nvidia-smi -l 1
```

---

## 📊 Batch Size Guide (RTX 3050)

| Batch Size | When to Use |
|------------|-------------|
| 32 | Conservative / Troubleshooting |
| 64 | If OOM error with 128 |
| **128** | **Recommended default** |
| 256 | Maximum performance (if no OOM) |

---

## ⚡ Expected Performance

| Metric | CPU | GPU (RTX 3050) |
|--------|-----|----------------|
| Time per Epoch | 15-20s | 1-3s |
| Total Training | 30-40 min | **5-10 min** |
| Speedup | 1x | **10-15x** |

---

## ✅ Success Indicators

### Startup Message
```
✅ SUCCESS: Hardware Accelerator Found: 1 GPU(s)
- VRAM Management: Dynamic (Memory Growth Enabled)
- Compute Policy: Mixed Precision (float16) ACTIVE
```

### During Training
- GPU Utilization: 80-100%
- VRAM Usage: 2-4GB
- Temperature: <80°C

---

## 🐛 Quick Troubleshooting

### "No GPU detected"
```bash
pip uninstall tensorflow
pip install "tensorflow<2.11"  # Windows requirement
```

### OOM Error
```bash
# Reduce batch size
python main.py --mode train --batch-size 64 ...
```

### Check GPU Status
```bash
nvidia-smi                    # GPU info
nvcc --version                # CUDA version
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📋 Required Versions (Windows)

| Component | Version |
|-----------|---------|
| TensorFlow | **<2.11** |
| CUDA | 11.2 |
| cuDNN | 8.1 |
| NVIDIA Driver | 450.80+ |
| Python | 3.8-3.11 |

---

## 🎯 One-Click Training

```bash
quick_start_gpu.bat
```

---

## 📚 Full Documentation

- **Setup Guide**: GPU_SETUP_GUIDE.md
- **Troubleshooting**: TROUBLESHOOTING.md  
- **Complete Summary**: GPU_INTEGRATION_SUMMARY.md

---

**Status**: ✅ Ready to use  
**Target**: NVIDIA RTX 3050 (6GB VRAM)  
**Speedup**: 10-20x faster training
