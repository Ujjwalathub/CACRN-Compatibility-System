import pandas as pd
import pickle
import sys
sys.path.insert(0, 'src')

from preprocessing import DataPreprocessor, create_feature_vector

# Load preprocessor
print("Loading preprocessor...")
preprocessor = DataPreprocessor.load('models/preprocessor.pkl')

# Load test data
print("Loading test data...")
test_df = pd.read_excel('data/test.xlsx')
print(f"Test data shape: {test_df.shape}")
print(f"Test columns: {list(test_df.columns)}")

# Transform one sample
print("\nTransforming first test sample...")
sample = test_df.head(1)
transformed = preprocessor.transform(sample)
print(f"Transformed type: {type(transformed)}")
if isinstance(transformed, dict):
    print(f"Transformed keys: {transformed.keys()}")
    for key, val in transformed.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}")

# Create feature vector
print("\nCreating feature vector...")
feature_vec = create_feature_vector(transformed, use_embeddings=True)
print(f"Feature vector shape: {feature_vec.shape}")
print(f"Expected: (1, 1154), Got: {feature_vec.shape}")

# Now check with training data
print("\n" + "="*80)
print("Checking with training data...")
train_df = pd.read_excel('data/train.xlsx')
print(f"Train data shape: {train_df.shape}")

sample_train = train_df.head(1)
transformed_train = preprocessor.transform(sample_train)
print(f"Transformed train type: {type(transformed_train)}")
if isinstance(transformed_train, dict):
    for key, val in transformed_train.items():
        if hasattr(val, 'shape'):
            print(f"  {key}: shape={val.shape}")

feature_vec_train = create_feature_vector(transformed_train, use_embeddings=True)
print(f"Feature vector train shape: {feature_vec_train.shape}")

# Check column differences
print("\n" + "="*80)
print("Checking for column differences after transformation...")
if isinstance(transformed, dict) and isinstance(transformed_train, dict):
    test_keys = set(transformed.keys())
    train_keys = set(transformed_train.keys())
    
    missing_in_test = train_keys - test_keys
    missing_in_train = test_keys - train_keys
    
    if missing_in_test:
        print(f"\nKeys in train but not in test: {missing_in_test}")
    if missing_in_train:
        print(f"\nKeys in test but not in train: {missing_in_train}")
        
    if not (missing_in_test or missing_in_train):
        print("\n✓ All keys present in both datasets")
