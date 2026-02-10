"""
Main Execution Script for CACRN
Provides a unified interface for training and prediction
"""

import os
import sys
import argparse
from datetime import datetime

# ==============================================================================
# GPU ACCELERATION SETUP - MUST BE FIRST!
# ==============================================================================
# Add CUDA libraries to PATH BEFORE importing TensorFlow

import site
# site.getsitepackages() returns the venv root, we need Lib/site-packages
site_packages = os.path.join(site.getsitepackages()[0], 'Lib', 'site-packages')
nvidia_dir = os.path.join(site_packages, 'nvidia')

if os.path.exists(nvidia_dir):
    # Add all CUDA library bin paths
    for lib_name in ['cudnn', 'cublas', 'cuda_runtime']:
        bin_path = os.path.join(nvidia_dir, lib_name, 'bin')
        if os.path.exists(bin_path):
            try:
                os.add_dll_directory(bin_path)
            except Exception as e:
                pass

# NOW import TensorFlow
import tensorflow as tf
from tensorflow.keras import mixed_precision


def setup_gpu_environment():
    """
    Configures TensorFlow to utilize NVIDIA GPU resources effectively.
    - Enables Memory Growth: Prevents TF from allocating all VRAM at launch.
    - Enables Mixed Precision: Uses float16 for faster math on RTX cards.
    
    Returns:
        bool: True if GPU is successfully configured, False if falling back to CPU
    """
    print("="*60)
    print("[SYSTEM] INITIALIZING GPU ACCELERATION")
    print("="*60)
    
    # 1. Detect Physical Devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("   ⚠️  WARNING: No GPU detected. Training will fall back to CPU.")
        print("       (Check CUDA installation or TensorFlow version compatibility)")
        return False

    try:
        for gpu in gpus:
            # 2. Enable Memory Growth
            # Critical for 6GB VRAM. Allows TF to allocate memory dynamically
            # rather than crashing by trying to grab the full 6GB instantly.
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"   ✅ SUCCESS: Hardware Accelerator Found: {len(gpus)} GPU(s)")
        print(f"      - Device Details: {gpus}")
        print("      - VRAM Management: Dynamic (Memory Growth Enabled)")
        
        # 3. Enable Mixed Precision (FP16)
        # RTX 3000 series Tensor Cores are optimized for FP16 operations.
        # This reduces memory usage by ~40% and speeds up matrix multiplication.
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("      - Compute Policy: Mixed Precision (float16) ACTIVE")
        print("      - Performance Boost: EXPECTED")
        
        return True

    except RuntimeError as e:
        print(f"   ❌ CRITICAL ERROR during GPU Setup: {e}")
        return False


# ==============================================================================
# EXECUTE SETUP IMMEDIATELY
# ==============================================================================
# This must be the first thing your script does.
using_gpu = setup_gpu_environment()
print("="*60 + "\n")


def train_model(args):
    """
    Train the CACRN model
    """
    # Import here to avoid loading unnecessary modules
    import sys
    sys.path.insert(0, 'src')
    from train import Trainer
    
    config = {
        # Data paths
        'train_profiles_path': args.train_profiles,
        'target_path': args.target,
        
        # Output settings
        'output_dir': args.output_dir,
        
        # Preprocessing settings
        'use_embeddings': not args.no_embeddings,
        'embedding_model': args.embedding_model,
        
        # Training settings
        'val_size': args.val_size,
        'random_state': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
    }
    
    # Validate paths
    if not os.path.exists(config['train_profiles_path']):
        print(f"ERROR: Training profiles file not found: {config['train_profiles_path']}")
        sys.exit(1)
    
    if not os.path.exists(config['target_path']):
        print(f"ERROR: Target file not found: {config['target_path']}")
        sys.exit(1)
    
    # Run training
    trainer = Trainer(config)
    trainer.run()


def predict_test(args):
    """
    Generate predictions on test data
    """
    # Import here to avoid loading unnecessary modules
    import sys
    sys.path.insert(0, 'src')
    from predict import Predictor
    
    config = {
        # Trained resources paths
        'model_path': args.model_path,
        'preprocessor_path': args.preprocessor_path,
        
        # Test data paths
        'test_profiles_path': args.test_profiles,
        'test_pairs_path': args.test_pairs,
        
        # Output settings
        'submission_path': args.submission,
        
        # Preprocessing settings
        'use_embeddings': not args.no_embeddings,
        
        # Prediction settings
        'batch_size': args.batch_size,
    }
    
    # Validate paths
    required_files = [
        config['model_path'],
        config['preprocessor_path'],
        config['test_profiles_path']
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure you have:")
        print("  1. Trained the model (run with --mode train)")
        print("  2. Placed test data in the correct location")
        sys.exit(1)
    
    # Run prediction
    predictor = Predictor(config)
    predictor.run()


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description='Context-Aware Compatibility Regression Network (CACRN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv
  
  # Generate predictions
  python main.py --mode predict --test-profiles data/test.xlsx --test-pairs data/test_pairs.csv
  
  # Train with custom settings
  python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 50 --batch-size 128
        """
    )
    
    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'predict'],
        help='Mode: train or predict'
    )
    
    # Training arguments
    parser.add_argument(
        '--train-profiles',
        type=str,
        default='data/train.xlsx',
        help='Path to training profiles file (default: data/train.xlsx)'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='data/target.csv',
        help='Path to target pairs file (default: data/target.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    # Prediction arguments
    parser.add_argument(
        '--test-profiles',
        type=str,
        default='data/test.xlsx',
        help='Path to test profiles file (default: data/test.xlsx)'
    )
    parser.add_argument(
        '--test-pairs',
        type=str,
        default='data/test_pairs.csv',
        help='Path to test pairs file (default: data/test_pairs.csv)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/best_model.h5',
        help='Path to trained model (default: models/best_model.h5)'
    )
    parser.add_argument(
        '--preprocessor-path',
        type=str,
        default='models/preprocessor.pkl',
        help='Path to fitted preprocessor (default: models/preprocessor.pkl)'
    )
    parser.add_argument(
        '--submission',
        type=str,
        default='output/submission.csv',
        help='Path to save submission file (default: output/submission.csv)'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size (default: 128 for GPU, reduce to 32-64 for CPU)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Validation set size (default: 0.2)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    # Feature engineering options
    parser.add_argument(
        '--no-embeddings',
        action='store_true',
        help='Use multi-hot encoding instead of sentence embeddings'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*80)
    print(" "*20 + "CACRN - Compatibility Prediction System")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Route to appropriate function
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'predict':
        predict_test(args)


if __name__ == '__main__':
    main()
