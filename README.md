# Context-Aware Compatibility Regression Network (CACRN)

A neural network-based solution for predicting user compatibility in business networking contexts, built for the Enigma Hackathon.

## 🎯 Overview

CACRN uses a **Siamese-style Neural Network with Cross-Interaction Head** to predict compatibility scores between users based on their business profiles. The system learns complementary relationships (e.g., "Provider" matches "Seeker") rather than just similarity.

### Key Features

- **Dual Encoding**: Separate encoders for each user with shared weights
- **Interaction Layers**: Explicit modeling of relationships through concatenation, difference, product, and cosine similarity
- **Logic Features**: Hand-crafted features capturing business logic (constraint violations, objective-interest overlap, etc.)
- **Semantic Embeddings**: Uses sentence transformers for text field encoding
- **Low Resource**: Trains efficiently on CPU/laptop without massive LLMs

## 📁 Project Structure

```
Model/
├── data/                      # Data directory (place your data here)
│   ├── train.xlsx            # Training profiles
│   ├── target.csv            # Training pairs with scores
│   ├── test.xlsx             # Test profiles
│   └── test_pairs.csv        # Test pairs (if provided)
├── src/                      # Source code
│   ├── preprocessing.py      # Data preprocessing module
│   ├── feature_engineering.py # Logic feature engineering
│   ├── dataset_builder.py    # Dataset construction utilities
│   ├── model.py              # Neural network architecture
│   ├── train.py              # Training pipeline
│   └── predict.py            # Prediction pipeline
├── models/                   # Saved models and preprocessors
├── output/                   # Prediction outputs
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Windows Users with NVIDIA GPU**: For GPU acceleration, ensure TensorFlow version is <2.11:
```bash
pip uninstall tensorflow
pip install "tensorflow<2.11"
```

### 2. Verify GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU, verify your system is configured for GPU acceleration:

```bash
python verify_gpu.py
```

This will check:
- ✅ NVIDIA driver installation
- ✅ TensorFlow GPU support
- ✅ CUDA/cuDNN compatibility
- ✅ Mixed precision support

**Expected output for successful setup:**
```
✅ SUCCESS: Hardware Accelerator Found: 1 GPU(s)
- VRAM Management: Dynamic (Memory Growth Enabled)
- Compute Policy: Mixed Precision (float16) ACTIVE
- Performance Boost: EXPECTED
```

### 3. Prepare Data

Place your data files in the `data/` directory:
- `train.xlsx` - Training profiles
- `target.csv` - Training pairs with compatibility scores
- `test.xlsx` - Test profiles
- `test_pairs.csv` - Test pairs (if provided by competition)

### 4. Train the Model

```bash
# GPU-accelerated training (RTX 3050 - 10x faster!)
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 128

# CPU training (use smaller batch size)
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --batch-size 32

# Custom training settings
python main.py --mode train \
    --train-profiles data/train.xlsx \
    --target data/target.csv \
    --epochs 50 \
    --batch-size 128 \
    --patience 15
```

**Batch Size Recommendations:**
- **RTX 3050 (6GB)**: 128-256
- **RTX 3060 (8GB+)**: 256-512
- **CPU Only**: 32-64

Training will:
- 🚀 Automatically detect and configure GPU if available
- Preprocess features and generate embeddings
- Engineer specialized logic features
- Train the neural network with validation split
- Save the best model and preprocessor
- Generate training plots and evaluation metrics

**Performance Expectations:**
- **With GPU (RTX 3050)**: 1-3 seconds per epoch, ~5 minutes total
- **CPU Only**: 10-20 seconds per epoch, ~30-60 minutes total

### 5. Generate Predictions

```bash
# Basic prediction
python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv

# Custom prediction settings
python main.py --mode predict \
    --test-profiles data/test.xlsx \
    --test-pairs data/test_pairs.csv \
    --model-path models/best_model.h5 \
    --submission output/submission.csv
```

## 📊 Model Architecture

### Phase 1: Data Preprocessing

1. **Semicolon-Separated Lists** (Business_Interests, Business_Objectives, Constraints)
   - Uses sentence embeddings (`all-MiniLM-L6-v2`) to encode semantic meaning
   - Alternative: Multi-hot encoding for lightweight approach

2. **Categorical Encoding** (Role, Seniority_Level, Industry, Location_City)
   - Entity embeddings learned during training
   - Maps categories to dense vectors

3. **Numerical Normalization** (Age, Company_Size_Employees)
   - MinMaxScaler to normalize to [0, 1] range

### Phase 2: Feature Engineering

Specialized logic features capture compatibility patterns:

- **Constraint Violation**: Binary flag for incompatible profiles
- **Objective-Interest Overlap**: Intersection between objectives and interests
- **Seniority Gap**: Difference in seniority levels
- **Role Complementarity**: Matches complementary roles (e.g., Investor-Startup)
- **Age Compatibility**: Normalized age difference
- **Company Size Ratio**: Size compatibility metric
- **Location Match**: Same city indicator
- **Industry Match**: Same industry indicator

### Phase 3: Neural Network

```
Input A (Source User) ────┐
                         │
                    [Shared Encoder]
                         │
                    [Encoded A]
                         │
                         ├─── Concatenation ───┐
                         │                      │
                         ├─── Difference ───────┤
                         │                      │
Input B (Dest User) ─────┤                      │
                         ├─── Product ──────────┤
                         │                      │
                         └─── Cosine Sim ───────┤
                                                 │
Logic Features ──────────────────────────────────┤
                                                 │
                                        [Deep Layers]
                                                 │
                                          [Output: 0-1]
```

**Key Components:**

- **Shared Encoder**: 128 → 64 → 32 dimensions with BatchNorm and Dropout
- **Interaction Layers**: 4 types of interactions between user embeddings
- **Deep Layers**: 128 → 64 → 32 → 16 dimensions
- **Output**: Sigmoid activation for [0, 1] compatibility score

## 🎛️ Configuration Options

### Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--val-size` | 0.2 | Validation set fraction |
| `--patience` | 10 | Early stopping patience |
| `--seed` | 42 | Random seed for reproducibility |
| `--no-embeddings` | False | Use multi-hot instead of embeddings |
| `--embedding-model` | all-MiniLM-L6-v2 | Sentence transformer model |

### Prediction Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | models/best_model.h5 | Path to trained model |
| `--preprocessor-path` | models/preprocessor.pkl | Path to preprocessor |
| `--submission` | output/submission.csv | Output submission file |

## 📈 Output Files

After training:
- `models/best_model.h5` - Best model weights
- `models/preprocessor.pkl` - Fitted preprocessor
- `models/evaluation_results.json` - Validation metrics
- `models/training_history.png` - Training curves

After prediction:
- `output/submission.csv` - Final submission file
- `output/submission_debug.csv` - Detailed predictions for debugging

## 🔍 Evaluation Metrics

The model optimizes for **Mean Squared Error (MSE)** and reports:
- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **R² Score**: Coefficient of determination

## 💡 Tips for Better Performance

1. **Data Quality**
   - Ensure clean semicolon-separated lists
   - Handle missing values appropriately
   - Verify profile IDs match between files

2. **Hyperparameter Tuning**
   - Increase `--epochs` if validation loss still decreasing
   - Adjust `--batch-size` based on memory (32, 64, 128)
   - Tune `--patience` for early stopping

3. **Feature Engineering**
   - Add domain-specific features in `feature_engineering.py`
   - Experiment with different complementary role pairs
   - Adjust seniority level mappings for your domain

4. **Model Architecture**
   - Modify encoder depth in `model.py`
   - Add/remove interaction types
   - Adjust dropout rates for regularization

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce `--batch-size` (try 32 or 16)
- Use `--no-embeddings` for lower memory footprint

### Poor Predictions
- Check data preprocessing logs for missing values
- Verify feature distributions in validation set
- Increase `--epochs` if underfitting
- Add dropout if overfitting

### File Not Found Errors
- Ensure data files are in correct locations
- Check file extensions (.xlsx vs .csv)
- Verify model and preprocessor exist before prediction

## 📚 Dependencies

- **TensorFlow**: Neural network framework
- **scikit-learn**: Preprocessing utilities
- **pandas**: Data manipulation
- **sentence-transformers**: Text embeddings
- **numpy**: Numerical operations

## 🏆 Competition Submission

The output file `output/submission.csv` is formatted as:
```csv
pair_id,compatibility_score
user1_user2,0.8542
user3_user4,0.3219
...
```

- `pair_id`: `{src_user_id}_{dst_user_id}`
- `compatibility_score`: Float in [0, 1]

## 🤝 Contributing

To extend this solution:

1. **Add new features**: Edit `feature_engineering.py`
2. **Modify architecture**: Edit `model.py`
3. **Custom preprocessing**: Edit `preprocessing.py`
4. **Change training logic**: Edit `train.py`

## 📄 License

This project is open-source for educational and competition purposes.

## 🙏 Acknowledgments

Built following best practices for:
- Siamese neural networks
- Feature engineering for compatibility matching
- Production-ready ML pipelines

---

**Good luck with your hackathon! 🚀**

For questions or issues, refer to the inline documentation in each module.
