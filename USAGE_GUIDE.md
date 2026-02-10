# CACRN Implementation Guide

## 📋 Step-by-Step Implementation

### Step 1: Environment Setup

1. **Verify Setup**
   ```bash
   python verify_setup.py
   ```
   
   This will check:
   - Python version (3.8+ recommended)
   - Required packages
   - Directory structure
   - Source files

2. **Install Dependencies** (if needed)
   ```bash
   pip install -r requirements.txt
   ```
   
   This may take a few minutes as it downloads:
   - TensorFlow (~500MB)
   - Sentence transformers models (~100MB on first use)

### Step 2: Prepare Your Data

1. **Place files in `data/` directory:**
   ```
   data/
   ├── train.xlsx        # Training user profiles
   ├── target.csv        # Training pairs with scores
   ├── test.xlsx         # Test user profiles
   └── test_pairs.csv    # Test pairs (if provided)
   ```

2. **Data Format Requirements:**

   **train.xlsx / test.xlsx** should have columns:
   - `Profile_ID`: Unique user identifier
   - `Age`: User age (numeric)
   - `Role`: User role (e.g., "Investor", "Startup Founder")
   - `Seniority_Level`: Experience level
   - `Industry`: Industry sector
   - `Location_City`: City location
   - `Company_Size_Employees`: Company size (numeric)
   - `Business_Interests`: Semicolon-separated interests
   - `Business_Objectives`: Semicolon-separated objectives
   - `Constraints`: Semicolon-separated constraints

   **target.csv** format:
   ```csv
   src_user_id,dst_user_id,compatibility_score
   1,2,0.85
   1,3,0.42
   ...
   ```

   **test_pairs.csv** format (if provided):
   ```csv
   src_user_id,dst_user_id
   100,101
   100,102
   ...
   ```

### Step 3: Train the Model

1. **Basic Training** (recommended for first run):
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv
   ```

2. **With Custom Settings**:
   ```bash
   python main.py --mode train \
       --train-profiles data/train.xlsx \
       --target data/target.csv \
       --epochs 50 \
       --batch-size 128 \
       --val-size 0.15 \
       --patience 15
   ```

3. **Training Process**:
   - Loads and preprocesses data (~2-5 minutes)
   - Downloads sentence transformer model on first run (~100MB)
   - Trains neural network (time varies by epochs)
   - Saves best model automatically
   - Generates training plots

4. **Expected Output**:
   ```
   models/
   ├── best_model.h5              # Trained model
   ├── preprocessor.pkl           # Fitted preprocessor
   ├── evaluation_results.json   # Metrics
   └── training_history.png      # Training curves
   ```

### Step 4: Generate Predictions

1. **Basic Prediction**:
   ```bash
   python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv
   ```

2. **Prediction Process**:
   - Loads trained model and preprocessor
   - Processes test data
   - Generates predictions
   - Creates submission file

3. **Expected Output**:
   ```
   output/
   ├── submission.csv       # Final submission
   └── submission_debug.csv # Detailed predictions
   ```

### Step 5: Validate Submission

1. **Check the submission file**:
   ```bash
   # On Windows PowerShell
   Get-Content output/submission.csv -Head 10
   ```

2. **Verify format**:
   - Should have columns: `pair_id`, `compatibility_score`
   - `pair_id` format: `{src_user_id}_{dst_user_id}`
   - Scores should be between 0 and 1
   - Number of rows should match test requirements (e.g., 16,000)

## 🎯 Quick Reference Commands

### Training Variations

```bash
# Fast training (for testing)
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 10 --batch-size 128

# CPU-optimized (less memory)
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 32 --no-embeddings

# High-accuracy training
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 200 --patience 20
```

### Prediction Variations

```bash
# With custom model
python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv --model-path models/custom_model.h5

# Different output location
python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv --submission output/final_submission.csv
```

## 🐛 Common Issues & Solutions

### Issue 1: "Out of Memory" Error

**Solution 1**: Reduce batch size
```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 32
```

**Solution 2**: Disable sentence embeddings
```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --no-embeddings
```

### Issue 2: "Model not found" during prediction

**Solution**: Ensure you've trained first
```bash
# Train first
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv

# Then predict
python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv
```

### Issue 3: Slow training

**Causes & Solutions**:
- **First run**: Downloading sentence transformer model (one-time)
- **Too many epochs**: Reduce `--epochs` to 50 or use early stopping
- **Large dataset**: Increase `--batch-size` to 128 or 256

### Issue 4: Poor validation scores

**Solutions**:
1. **Increase training**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 150
   ```

2. **Adjust architecture**: Edit `src/model.py` to add more layers or units

3. **Add features**: Edit `src/feature_engineering.py` to add domain-specific features

### Issue 5: Missing test pairs file

**Solution**: If competition doesn't provide test pairs, you need to generate them. Edit `src/dataset_builder.py` method `generate_test_pairs()` to create all possible pairs or use competition-specific logic.

## 📊 Understanding the Output

### Training Output Files

1. **best_model.h5**
   - Neural network weights
   - Automatically loaded for prediction
   - Can be loaded manually: `tf.keras.models.load_model('models/best_model.h5')`

2. **preprocessor.pkl**
   - Fitted scalers and encodings
   - Required for prediction
   - Ensures consistent feature transformation

3. **evaluation_results.json**
   ```json
   {
     "val_loss": 0.0234,
     "val_mse": 0.0234,
     "val_mae": 0.1123,
     "val_rmse": 0.1529,
     "val_r2": 0.8234
   }
   ```
   - Lower is better for loss, MSE, MAE, RMSE
   - Higher is better for R² (max 1.0)

4. **training_history.png**
   - Visualizes training progress
   - Check for overfitting (train/val divergence)

### Prediction Output Files

1. **submission.csv**
   ```csv
   pair_id,compatibility_score
   user1_user2,0.8542
   user1_user3,0.3219
   ```
   - Ready for submission
   - Scores clipped to [0, 1]

2. **submission_debug.csv**
   - Includes src_user_id and dst_user_id separately
   - Useful for manual inspection

## 🚀 Optimization Tips

### For Speed

1. **Use smaller embedding model**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --embedding-model all-MiniLM-L6-v2
   ```

2. **Increase batch size** (if you have RAM):
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 256
   ```

### For Accuracy

1. **More epochs with patience**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 200 --patience 25
   ```

2. **Larger validation set**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --val-size 0.25
   ```

3. **Custom features**: Add domain knowledge to `src/feature_engineering.py`

### For Low Resources

1. **Disable embeddings**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --no-embeddings
   ```

2. **Small batch + fewer epochs**:
   ```bash
   python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 16 --epochs 30
   ```

## 📚 Next Steps

1. **Baseline**: Run with defaults first to establish baseline
2. **Iterate**: Experiment with hyperparameters
3. **Customize**: Add domain-specific features and logic
4. **Validate**: Check submission format carefully
5. **Submit**: Upload to competition platform

## 🤝 Need Help?

Check inline documentation in source files:
- `src/preprocessing.py` - Data transformation details
- `src/feature_engineering.py` - Feature logic
- `src/model.py` - Architecture details
- `src/train.py` - Training process
- `src/predict.py` - Prediction pipeline

---

**Good luck! 🎯**
