"""
Prediction and Submission Script for CACRN
Generates predictions on test data and creates submission file
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

from preprocessing import DataPreprocessor, create_feature_vector
from feature_engineering import FeatureEngineer
from dataset_builder import DatasetBuilder, create_submission_file


class Predictor:
    """
    Manages the prediction pipeline for test data
    """
    
    def __init__(self, config):
        """
        Initialize the predictor
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.preprocessor = None
        self.feature_engineer = None
        self.model = None
        
    def load_resources(self):
        """
        Load the trained model and preprocessor
        """
        print("="*80)
        print("LOADING TRAINED RESOURCES")
        print("="*80)
        
        # Load preprocessor
        preprocessor_path = self.config['preprocessor_path']
        print(f"\nLoading preprocessor from {preprocessor_path}")
        self.preprocessor = DataPreprocessor.load(preprocessor_path)
        
        # Load model
        model_path = self.config['model_path']
        print(f"Loading model from {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Error loading model with default method: {e}")
            print("\nAttempting alternative loading method...")
            print("Reconstructing model architecture and loading weights...")
            
            # Import the actual function from model.py
            from model import build_model
            
            # --- FIX STARTS HERE ---
            # FORCE the base_dim to 1154 to match the saved best_model.h5 file.
            # Do NOT use self.preprocessor.input_dim or similar variables here.
            FIXED_BASE_DIM = 1154
            logic_dim = 12  # Updated logic feature dimension
            
            # Rebuild model architecture (must match training exactly)
            print(f"Rebuilding model with base_dim={FIXED_BASE_DIM}, logic_dim={logic_dim}")
            self.model = build_model(base_dim=FIXED_BASE_DIM, logic_dim=logic_dim)
            # --- FIX ENDS HERE ---
            
            # Load weights only (version-independent!)
            self.model.load_weights(model_path)
            print("✅ Successfully reconstructed model and loaded weights!")
        
        print("\nModel architecture:")
        self.model.summary()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        print("\nResources loaded successfully!")
        
    def prepare_test_data(self):
        """
        Load and prepare test data
        """
        print("\n" + "="*80)
        print("PREPARING TEST DATA")
        print("="*80)
        
        # Initialize dataset builder for test data
        dataset_builder = DatasetBuilder(
            profiles_path=self.config['test_profiles_path'],
            target_path=None  # No target for test data
        )
        
        # Generate test pairs
        # Check if test pairs file is provided
        test_pairs_path = self.config.get('test_pairs_path', None)
        
        if test_pairs_path:
            print(f"\nUsing provided test pairs from {test_pairs_path}")
        else:
            print("\nNo test pairs file provided.")
            print("Note: You may need to generate pairs based on competition requirements.")
            print("For now, we'll assume test pairs are provided.")
        
        src_df, dst_df, pair_ids_df = dataset_builder.generate_test_pairs(test_pairs_path)
        
        print("\n" + "="*80)
        print("PREPROCESSING TEST FEATURES")
        print("="*80)
        
        # Transform test data using fitted preprocessor
        print("\nTransforming source users...")
        src_features = self.preprocessor.transform(src_df)
        
        print("\nTransforming destination users...")
        dst_features = self.preprocessor.transform(dst_df)
        
        # Create feature vectors
        X_src_test = create_feature_vector(src_features,
                                          use_embeddings=self.config.get('use_embeddings', True))
        X_dst_test = create_feature_vector(dst_features,
                                          use_embeddings=self.config.get('use_embeddings', True))
        
        print(f"\nBase feature dimensions: {X_src_test.shape[1]}")
        
        print("\n" + "="*80)
        print("ENGINEERING LOGIC FEATURES FOR TEST DATA")
        print("="*80)
        
        # Generate logic features for test data
        logic_test = self.feature_engineer.generate_pair_features(src_df, dst_df)
        
        print(f"\nLogic feature dimensions: {logic_test.shape[1]}")
        
        # Save test data
        self.test_data = {
            'X_src': X_src_test,
            'X_dst': X_dst_test,
            'logic': logic_test.values.astype(np.float32),
            'pair_ids': pair_ids_df
        }
        
        print(f"\nTest set size: {len(X_src_test)}")
        
    def predict(self):
        """
        Generate predictions on test data
        """
        print("\n" + "="*80)
        print("GENERATING PREDICTIONS")
        print("="*80)
        
        # Make predictions
        print("\nRunning model inference...")
        predictions = self.model.predict(
            [self.test_data['X_src'], self.test_data['X_dst'], self.test_data['logic']],
            batch_size=self.config.get('batch_size', 64),
            verbose=1
        ).flatten()
        
        # Clip predictions to [0, 1] range (safety measure)
        predictions = np.clip(predictions, 0, 1)
        
        print("\nPrediction Statistics:")
        print(f"  Min: {predictions.min():.4f}")
        print(f"  Max: {predictions.max():.4f}")
        print(f"  Mean: {predictions.mean():.4f}")
        print(f"  Median: {np.median(predictions):.4f}")
        print(f"  Std: {predictions.std():.4f}")
        
        # Save predictions
        self.predictions = predictions
        
    def create_submission(self):
        """
        Create submission file in the required format
        """
        print("\n" + "="*80)
        print("CREATING SUBMISSION FILE")
        print("="*80)
        
        submission_path = self.config['submission_path']
        
        create_submission_file(
            pair_ids_df=self.test_data['pair_ids'],
            predictions=self.predictions,
            output_path=submission_path
        )
        
        print(f"\n✓ Submission file created successfully!")
        print(f"  Location: {submission_path}")
        
        # Also save predictions with additional details for debugging
        debug_submission = self.test_data['pair_ids'].copy()
        debug_submission['compatibility_score'] = self.predictions
        
        debug_path = submission_path.replace('.csv', '_debug.csv')
        debug_submission.to_csv(debug_path, index=False)
        print(f"  Debug file: {debug_path}")
        
    def run(self):
        """
        Run the complete prediction pipeline
        """
        start_time = datetime.now()
        print("\n" + "="*80)
        print("STARTING CACRN PREDICTION PIPELINE")
        print("="*80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directory
        os.makedirs(os.path.dirname(self.config['submission_path']), exist_ok=True)
        
        # Run pipeline steps
        self.load_resources()
        self.prepare_test_data()
        self.predict()
        self.create_submission()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("PREDICTION PIPELINE COMPLETE!")
        print("="*80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")


def main():
    """
    Main prediction function
    """
    # Configuration
    config = {
        # Trained resources paths
        'model_path': 'models/best_model.h5',
        'preprocessor_path': 'models/preprocessor.pkl',
        
        # Test data paths
        'test_profiles_path': 'data/test.xlsx',  # or 'data/test.csv'
        'test_pairs_path': 'data/test_pairs.csv',  # CSV with src_user_id, dst_user_id
        
        # Output settings
        'submission_path': 'output/submission.csv',
        
        # Preprocessing settings (must match training)
        'use_embeddings': True,
        
        # Prediction settings
        'batch_size': 64,
    }
    
    # Check if required files exist
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
        print("  1. Trained the model (run train.py)")
        print("  2. Placed test data in the correct location")
        sys.exit(1)
    
    # Initialize and run predictor
    predictor = Predictor(config)
    predictor.run()


if __name__ == '__main__':
    main()
