"""
Configuration Template for CACRN
Copy and modify this file to customize your training and prediction settings
"""

# ==============================================================================
# DATA PATHS
# ==============================================================================

DATA_CONFIG = {
    # Training data
    'train_profiles': 'data/train.xlsx',
    'target_pairs': 'data/target.csv',
    
    # Test data
    'test_profiles': 'data/test.xlsx',
    'test_pairs': 'data/test_pairs.csv',
    
    # Output directories
    'output_dir': 'models',
    'submission_dir': 'output',
}


# ==============================================================================
# PREPROCESSING SETTINGS
# ==============================================================================

PREPROCESSING_CONFIG = {
    # Use sentence embeddings (True) or multi-hot encoding (False)
    'use_embeddings': True,
    
    # Sentence transformer model
    # Options: 'all-MiniLM-L6-v2' (fast, 384-dim)
    #          'all-mpnet-base-v2' (better quality, 768-dim, slower)
    #          'paraphrase-MiniLM-L6-v2' (good for paraphrases)
    'embedding_model': 'all-MiniLM-L6-v2',
    
    # Normalization method for numerical features
    # Options: 'minmax', 'standard'
    'normalization_method': 'minmax',
}


# ==============================================================================
# FEATURE ENGINEERING SETTINGS
# ==============================================================================

FEATURE_ENGINEERING_CONFIG = {
    # Enable/disable specific feature groups
    'use_constraint_features': True,
    'use_overlap_features': True,
    'use_seniority_features': True,
    'use_role_complementarity': True,
    'use_demographic_features': True,
    'use_location_features': True,
    'use_industry_features': True,
    
    # Custom seniority level mappings (adjust for your domain)
    'seniority_mapping': {
        'intern': 1,
        'junior': 2,
        'mid-level': 3,
        'mid level': 3,
        'senior': 4,
        'lead': 5,
        'manager': 6,
        'director': 7,
        'vp': 8,
        'vice president': 8,
        'c-level': 9,
        'executive': 9,
        'ceo': 9,
        'cto': 9,
        'cfo': 9,
    },
    
    # Complementary role pairs (customize for your domain)
    'complementary_roles': [
        ('provider', 'seeker'),
        ('investor', 'startup'),
        ('investor', 'entrepreneur'),
        ('mentor', 'mentee'),
        ('advisor', 'founder'),
        ('buyer', 'seller'),
        ('client', 'service provider'),
        ('recruiter', 'job seeker'),
        ('employer', 'job seeker'),
        ('partner', 'partner'),
    ],
}


# ==============================================================================
# MODEL ARCHITECTURE SETTINGS
# ==============================================================================

MODEL_CONFIG = {
    # Encoder architecture
    'encoder_layers': [128, 64, 32],  # Layer sizes
    'encoder_activation': 'relu',
    'encoder_dropout': [0.3, 0.2],    # Dropout after each layer
    'use_batch_norm': True,
    
    # Deep layers architecture (after interaction)
    'deep_layers': [128, 64, 32, 16],
    'deep_activation': 'relu',
    'deep_dropout': [0.3, 0.3, 0.2],
    
    # Interaction types to use
    'use_concatenation': True,
    'use_difference': True,
    'use_product': True,
    'use_cosine_similarity': True,
    
    # Categorical embeddings
    'use_categorical_embeddings': False,  # Set to True for advanced usage
    'embedding_dims': {
        'role': 8,
        'seniority': 4,
        'industry': 8,
        'location': 8,
    },
}


# ==============================================================================
# TRAINING SETTINGS
# ==============================================================================

TRAINING_CONFIG = {
    # Data splitting
    'val_size': 0.2,         # Validation set fraction
    'random_state': 42,      # Random seed for reproducibility
    
    # Training hyperparameters
    'epochs': 100,           # Maximum number of epochs
    'batch_size': 128,       # Batch size (GPU optimized: 128-256, CPU: 32-64)
    'learning_rate': 0.001,  # Initial learning rate
    
    # Optimizer settings
    'optimizer': 'adam',     # Options: 'adam', 'sgd', 'rmsprop'
    
    # Loss function
    'loss': 'mse',          # Options: 'mse', 'mae', 'huber'
    
    # Callbacks
    'early_stopping_patience': 10,  # Epochs to wait before stopping
    'reduce_lr_patience': 5,        # Epochs to wait before reducing LR
    'reduce_lr_factor': 0.5,        # Factor to reduce LR by
    'min_learning_rate': 1e-6,      # Minimum learning rate
    
    # Model checkpointing
    'save_best_only': True,
    'monitor_metric': 'val_loss',   # Metric to monitor for best model
    
    # TensorBoard
    'use_tensorboard': True,
    'tensorboard_log_dir': 'logs',
}


# ==============================================================================
# PREDICTION SETTINGS
# ==============================================================================

PREDICTION_CONFIG = {
    # Batch size for prediction
    'batch_size': 64,
    
    # Post-processing
    'clip_predictions': True,  # Ensure predictions are in [0, 1]
    'round_predictions': False,  # Round to specific decimal places
    'round_decimals': 4,
    
    # Output format
    'include_debug_info': True,  # Save additional debug file
}


# ==============================================================================
# ADVANCED SETTINGS
# ==============================================================================

ADVANCED_CONFIG = {
    # Data augmentation (experimental)
    'use_data_augmentation': False,
    'augmentation_factor': 2,  # Generate N copies of each pair
    
    # Ensemble settings (experimental)
    'use_ensemble': False,
    'n_models': 5,
    
    # Multi-GPU training (if available)
    'use_multi_gpu': False,
    
    # Mixed precision training (faster on modern GPUs)
    # NOTE: Mixed precision is now enabled automatically by setup_gpu_environment() in main.py
    # This setting is deprecated but kept for backward compatibility
    'use_mixed_precision': True,
    
    # Verbose logging
    'verbose': 1,  # 0: silent, 1: progress bar, 2: one line per epoch
}


# ==============================================================================
# GPU ACCELERATION SETTINGS
# ==============================================================================

GPU_CONFIG = {
    # Batch size recommendations by hardware
    'batch_size_recommendations': {
        'cpu': 32,              # CPU-only systems
        'gpu_4gb': 64,          # Entry-level GPUs (GTX 1650, etc.)
        'gpu_6gb': 128,         # RTX 3050, RTX 2060
        'gpu_8gb_plus': 256,    # RTX 3060, RTX 3070, etc.
    },
    
    # Memory management
    'gpu_memory_growth': True,      # Enabled automatically by main.py
    'gpu_memory_fraction': None,    # None = dynamic allocation, 0.8 = use 80% max
    
    # Mixed precision training
    'mixed_precision_enabled': True,  # Enabled automatically for RTX GPUs
    
    # Troubleshooting
    'force_cpu': False,  # Set to True to disable GPU even if available
    
    # CUDA settings
    'cuda_visible_devices': None,  # None = all GPUs, '0' = first GPU only
}


# ==============================================================================
# HELPER FUNCTION
# ==============================================================================

def get_full_config():
    """
    Get complete configuration by merging all config dictionaries
    """
    full_config = {}
    full_config.update(DATA_CONFIG)
    full_config.update(PREPROCESSING_CONFIG)
    full_config.update(FEATURE_ENGINEERING_CONFIG)
    full_config.update(MODEL_CONFIG)
    full_config.update(TRAINING_CONFIG)
    full_config.update(PREDICTION_CONFIG)
    full_config.update(ADVANCED_CONFIG)
    full_config.update(GPU_CONFIG)
    
    return full_config


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == '__main__':
    """
    Example: Print current configuration
    """
    import json
    
    config = get_full_config()
    
    print("="*80)
    print("Current CACRN Configuration")
    print("="*80)
    print(json.dumps(config, indent=2))
    print("="*80)
