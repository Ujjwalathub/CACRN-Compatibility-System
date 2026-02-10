# Troubleshooting Guide

## Common Errors and Solutions

### GPU Acceleration Issues (NEW)

#### Warning: `No GPU detected. Training will fall back to CPU`

**Cause**: TensorFlow cannot detect your NVIDIA GPU

**Solutions**:

1. **Check GPU Driver**:
   ```bash
   nvidia-smi
   ```
   - If this command fails, update your NVIDIA drivers from https://www.nvidia.com/drivers
   - Ensure driver version is 450.80.02+ for CUDA 11.2

2. **Verify TensorFlow GPU Support**:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   - If empty list `[]`, TensorFlow cannot see GPU

3. **Windows-Specific: TensorFlow Version Requirement**:
   - **Critical**: TensorFlow 2.11+ dropped native Windows GPU support
   - **Solution**: Install TensorFlow 2.10 or lower
   ```bash
   pip uninstall tensorflow
   pip install "tensorflow<2.11"
   ```

4. **Install CUDA Toolkit and cuDNN**:
   - Download CUDA 11.2 from NVIDIA website
   - Download cuDNN 8.1 for CUDA 11.2
   - Add to system PATH:
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
     - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp`

5. **Verify Installation**:
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
   ```

#### Error: `OOM (Out of Memory)` when running with GPU

**Cause**: Batch size too large for your GPU's VRAM (6GB on RTX 3050)

**Solutions**:

1. **Reduce Batch Size Gradually**:
   ```bash
   # Try 128 first (recommended for RTX 3050)
   python main.py --mode train --batch-size 128 ...
   
   # If OOM persists, reduce to 64
   python main.py --mode train --batch-size 64 ...
   
   # For severe cases, use 32
   python main.py --mode train --batch-size 32 ...
   ```

2. **Memory Growth Already Enabled**: The GPU setup automatically enables memory growth, but if you need manual control:
   ```python
   # In main.py, the setup_gpu_environment() function handles this
   tf.config.experimental.set_memory_growth(gpu, True)
   ```

3. **Close GPU-Intensive Applications**:
   - Close Chrome/Firefox (they use GPU for rendering)
   - Close games, video editors, or other ML applications
   - Check GPU usage: `nvidia-smi`

#### Error: `cudart64_110.dll not found` or similar DLL errors

**Cause**: Missing CUDA libraries

**Solutions**:

1. **Download Missing DLL**:
   - Search for the specific DLL file online
   - Place in `C:\Windows\System32\` or CUDA bin directory

2. **Reinstall CUDA Toolkit**:
   - Uninstall existing CUDA versions
   - Clean install CUDA 11.2 from NVIDIA

3. **Add CUDA to PATH** (if not already):
   ```powershell
   # Open PowerShell as Administrator
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin", "Machine")
   ```

#### Error: Mixed Precision - `loss scaling` warnings

**Cause**: Normal behavior with mixed precision training (float16)

**Solution**: These warnings are informational, not errors. Mixed precision is working correctly. If loss becomes `nan`:
1. Disable mixed precision by commenting out in main.py:
   ```python
   # policy = mixed_precision.Policy('mixed_float16')
   # mixed_precision.set_global_policy(policy)
   ```

#### GPU not being utilized (low GPU usage in `nvidia-smi`)

**Causes & Solutions**:

1. **Batch size too small**:
   - GPU spends more time on data transfer than computation
   - **Solution**: Increase batch size to 128 or 256

2. **Data loading bottleneck**:
   - CPU preprocessing is the bottleneck
   - **Solution**: The model already loads all data in memory, so this shouldn't be an issue

3. **Small model**:
   - If your model is very small, GPU might be underutilized
   - **Check**: Monitor `nvidia-smi` during training to see GPU utilization %

#### Performance not improving with GPU

**Diagnostic Steps**:

1. **Compare Training Times**:
   - CPU: ~10-20 seconds per epoch (typical)
   - GPU: ~1-2 seconds per epoch (expected with RTX 3050)
   
2. **Check GPU Memory Usage**:
   ```bash
   # Run this in a separate terminal while training
   nvidia-smi -l 1  # Updates every second
   ```
   - Should see 2-4GB VRAM usage with batch_size=128

3. **Verify Mixed Precision is Active**:
   - Check console output during startup:
   ```
   - Compute Policy: Mixed Precision (float16) ACTIVE
   ```

---

### Installation Issues

#### Error: `pip: command not found` or `python: command not found`

**Cause**: Python is not installed or not in PATH

**Solution**:
1. Install Python 3.8+ from https://www.python.org/
2. During installation, check "Add Python to PATH"
3. Restart your terminal/command prompt
4. Verify: `python --version`

#### Error: `No module named 'tensorflow'`

**Cause**: Dependencies not installed

**Solution**:
```bash
pip install -r requirements.txt
```

If this fails:
```bash
# Install packages one by one
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install numpy
pip install sentence-transformers
pip install openpyxl
```

#### Error: `Could not find a version that satisfies the requirement tensorflow`

**Cause**: Python version incompatibility (too old or too new)

**Solution**:
- Check Python version: `python --version`
- TensorFlow requires Python 3.8-3.11
- If using Python 3.12+, downgrade to 3.11 or use a virtual environment

---

### Training Errors

#### Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data/train.xlsx'`

**Cause**: Data files not in correct location

**Solution**:
1. Create `data/` directory if it doesn't exist
2. Place your data files there:
   ```
   data/
   ├── train.xlsx
   └── target.csv
   ```
3. Verify file names match exactly (case-sensitive on Linux/Mac)

#### Error: `MemoryError` or `OOM (Out of Memory)`

**Cause**: Not enough RAM

**Solutions**:

1. **Reduce batch size**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 16
   ```

2. **Disable sentence embeddings**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --no-embeddings
   ```

3. **Use a machine with more RAM** or **cloud platform**

4. **Close other applications** to free memory

#### Error: `ValueError: Dimension mismatch`

**Cause**: Inconsistent features between train and test, or corrupted preprocessor

**Solution**:
1. Delete `models/preprocessor.pkl`
2. Retrain from scratch
3. Ensure train and test data have same columns

#### Error: Training loss is `nan` or very high

**Causes & Solutions**:

1. **Learning rate too high**:
   ```python
   # Edit src/model.py, change learning rate
   optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
   ```

2. **Data scaling issues**:
   - Check for infinite or very large values in data
   - Ensure `Age` and `Company_Size_Employees` are reasonable numbers

3. **Missing data**:
   - Check for NaN values in your data
   - Use `df.fillna()` or drop rows with missing values

#### Error: `Validation loss not decreasing`

**Causes & Solutions**:

1. **Underfitting**:
   - Increase epochs: `--epochs 200`
   - Add more layers in `src/model.py`
   - Reduce dropout rates

2. **Bad hyperparameters**:
   - Try different learning rates
   - Adjust batch size
   - Change model architecture

3. **Data quality issues**:
   - Check if features are informative
   - Verify target scores are correct
   - Look for data leakage

---

### Prediction Errors

#### Error: `Model not found: models/best_model.h5`

**Cause**: Haven't trained the model yet

**Solution**:
```bash
# Train first
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv

# Then predict
python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv
```

#### Error: `Mismatch in feature dimensions`

**Cause**: Test data preprocessed differently than training data

**Solution**:
1. Ensure `preprocessor.pkl` exists and is from the same training run
2. Use the same `--no-embeddings` flag for both train and predict
3. Retrain if you changed preprocessing settings

#### Error: `All predictions are the same value`

**Causes & Solutions**:

1. **Model not trained properly**:
   - Check training loss - should decrease significantly
   - Train for more epochs
   - Verify validation metrics

2. **Bad initialization**:
   - Delete model files and retrain
   - Try different random seed: `--seed 123`

3. **Feature scaling issues**:
   - Check if features have variance
   - Verify normalization is working

---

### Data Format Issues

#### Error: `KeyError: 'Profile_ID'` or `KeyError: 'compatibility_score'`

**Cause**: Column names don't match expected format

**Solution**:
1. Check your CSV/Excel headers
2. Required columns for profiles:
   - `Profile_ID`
   - `Age`
   - `Role`
   - `Seniority_Level`
   - `Industry`
   - `Location_City`
   - `Company_Size_Employees`
   - `Business_Interests`
   - `Business_Objectives`
   - `Constraints`

3. Required columns for target:
   - `src_user_id`
   - `dst_user_id`
   - `compatibility_score`

#### Error: `ValueError: could not convert string to float`

**Cause**: Non-numeric data in numeric columns

**Solution**:
1. Check `Age` column - should be integers
2. Check `Company_Size_Employees` - should be integers
3. Check `compatibility_score` - should be floats between 0 and 1
4. Remove any text or special characters from numeric columns

#### Error: Excel file parsing errors

**Cause**: File format or encoding issues

**Solutions**:
1. Save Excel file as CSV and use that instead
2. Ensure Excel file is `.xlsx` format (not `.xls`)
3. Check for special characters in data
4. Try opening and re-saving the file in Excel

---

### Performance Issues

#### Problem: Training is very slow

**Causes & Solutions**:

1. **First-time setup**:
   - Downloading sentence transformer model (~100MB)
   - This happens once, wait for it to complete

2. **CPU-only training**:
   - Normal for laptops without GPU
   - Reduce batch size to speed up: `--batch-size 32`
   - Reduce epochs for testing: `--epochs 10`

3. **Large dataset**:
   - Increase batch size: `--batch-size 128`
   - Use GPU if available

4. **Sentence embeddings**:
   - Try `--no-embeddings` for faster training
   - Use smaller model: `--embedding-model all-MiniLM-L6-v2`

#### Problem: Prediction takes too long

**Solutions**:
1. Increase batch size: `--batch-size 256`
2. Process in chunks if you have many test pairs
3. Use GPU if available

---

### Submission Issues

#### Error: Submission file has wrong number of rows

**Cause**: Test pairs not generated correctly

**Solution**:
1. Check if `test_pairs.csv` is provided by competition
2. Verify number of rows matches competition requirements
3. Check for duplicate pairs
4. Ensure all pairs are included

#### Error: Submission file format rejected

**Cause**: Wrong CSV format

**Solution**:
1. Check delimiter (should be comma)
2. Verify column names: `pair_id`, `compatibility_score`
3. Ensure no extra columns or missing values
4. Check `pair_id` format: `{src_user_id}_{dst_user_id}`
5. Verify scores are between 0 and 1

Example correct format:
```csv
pair_id,compatibility_score
100_101,0.8542
100_102,0.3219
```

#### Problem: Low competition score

**Strategies to improve**:

1. **More training**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 200 --patience 25
   ```

2. **Feature engineering**:
   - Add domain-specific features in `src/feature_engineering.py`
   - Customize complementary role pairs
   - Add business logic rules

3. **Hyperparameter tuning**:
   - Try different batch sizes
   - Adjust learning rate
   - Modify architecture in `src/model.py`

4. **Ensemble**:
   - Train multiple models with different seeds
   - Average their predictions

5. **Data quality**:
   - Check for outliers in training data
   - Verify label quality
   - Handle missing data better

---

### Environment Issues

#### Error: `ImportError: DLL load failed` (Windows)

**Cause**: Missing Visual C++ Redistributables

**Solution**:
1. Install Visual C++ Redistributables from Microsoft
2. Reinstall TensorFlow: `pip install --force-reinstall tensorflow`

#### Error: `Illegal instruction` (Linux/Mac)

**Cause**: TensorFlow compiled for newer CPU instructions

**Solution**:
```bash
pip install --upgrade tensorflow
# Or use CPU-only version
pip install tensorflow-cpu
```

#### Error: Permission denied when saving files

**Cause**: Insufficient permissions

**Solution**:
1. Run terminal as administrator (Windows) or use `sudo` (Linux/Mac)
2. Check folder permissions
3. Save to a different directory

---

### Debug Mode

To get more detailed error information, run with verbose output:

```python
# Edit src/train.py or src/predict.py
# Add at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use Python's verbose mode:
```bash
python -v main.py --mode train --train-profiles data/train.xlsx --target data/target.csv
```

---

### Still Having Issues?

1. **Check logs**: Look for detailed error messages in terminal output
2. **Verify setup**: Run `python verify_setup.py`
3. **Test with small data**: Try with a subset of your data first
4. **Check file formats**: Open files in Excel/text editor to verify format
5. **Update packages**: `pip install --upgrade -r requirements.txt`

---

### Getting Help

When asking for help, include:
1. **Exact error message** (full traceback)
2. **Command you ran**
3. **Python version**: `python --version`
4. **Package versions**: `pip list`
5. **Operating system**
6. **Data format** (sample of CSV headers)
7. **Steps to reproduce**

Example good help request:
```
I'm getting this error when training:
[paste full error traceback]

Command: python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv
Python: 3.10.5
OS: Windows 11
Data: train.xlsx has 1000 rows, target.csv has 5000 pairs
```

This helps diagnose the issue quickly!
