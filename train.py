"""
Training Pipeline for CACRN
Orchestrates the full training process
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from preprocessing import DataPreprocessor, create_feature_vector
from feature_engineering import FeatureEngineer
from dataset_builder import DatasetBuilder
from model import build_simple_model, create_callbacks


class Trainer:
    """
    Manages the end-to-end training pipeline
    """
    
    def __init__(self, config):
        """
        Initialize the trainer
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.preprocessor = None
        self.feature_engineer = None
        self.model = None
        self.history = None
        
    def prepare_data(self):
        """
        Load and prepare the training data
        """
        print("="*80)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*80)
        
        # Initialize dataset builder
        dataset_builder = DatasetBuilder(
            profiles_path=self.config['train_profiles_path'],
            target_path=self.config['target_path']
        )
        
        # Build pairs dataset
        src_df, dst_df, targets = dataset_builder.build_pairs_dataset()
        
        # Split into train and validation
        src_train, src_val, dst_train, dst_val, y_train, y_val = \
            dataset_builder.split_train_val(
                src_df, dst_df, targets,
                val_size=self.config.get('val_size', 0.2),
                random_state=self.config.get('random_state', 42)
            )
        
        print("\n" + "="*80)
        print("STEP 2: PREPROCESSING FEATURES")
        print("="*80)
        
        # Initialize and fit preprocessor on training data
        self.preprocessor = DataPreprocessor(
            use_sentence_embeddings=self.config.get('use_embeddings', True),
            embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Fit on all unique profiles (train + val)
        all_profiles = pd.concat([src_train, dst_train, src_val, dst_val]).drop_duplicates()
        self.preprocessor.fit(all_profiles)
        
        # Transform training data
        print("\nTransforming training data...")
        src_train_features = self.preprocessor.transform(src_train)
        dst_train_features = self.preprocessor.transform(dst_train)
        
        # Transform validation data
        print("\nTransforming validation data...")
        src_val_features = self.preprocessor.transform(src_val)
        dst_val_features = self.preprocessor.transform(dst_val)
        
        # Create feature vectors
        X_src_train = create_feature_vector(src_train_features, 
                                           use_embeddings=self.config.get('use_embeddings', True))
        X_dst_train = create_feature_vector(dst_train_features,
                                           use_embeddings=self.config.get('use_embeddings', True))
        
        X_src_val = create_feature_vector(src_val_features,
                                         use_embeddings=self.config.get('use_embeddings', True))
        X_dst_val = create_feature_vector(dst_val_features,
                                         use_embeddings=self.config.get('use_embeddings', True))
        
        print(f"\nBase feature dimensions: {X_src_train.shape[1]}")
        
        print("\n" + "="*80)
        print("STEP 3: ENGINEERING LOGIC FEATURES")
        print("="*80)
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Generate logic features for training
        logic_train = self.feature_engineer.generate_pair_features(src_train, dst_train)
        
        # Generate logic features for validation
        logic_val = self.feature_engineer.generate_pair_features(src_val, dst_val)
        
        print(f"\nLogic feature dimensions: {logic_train.shape[1]}")
        
        # Save processed data
        self.train_data = {
            'X_src': X_src_train,
            'X_dst': X_dst_train,
            'logic': logic_train.values.astype(np.float32),
            'y': y_train
        }
        
        self.val_data = {
            'X_src': X_src_val,
            'X_dst': X_dst_val,
            'logic': logic_val.values.astype(np.float32),
            'y': y_val
        }
        
        print(f"\nTraining set size: {len(y_train)}")
        print(f"Validation set size: {len(y_val)}")
        
        # Save preprocessor
        preprocessor_path = os.path.join(self.config['output_dir'], 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
    def build_and_train(self):
        """
        Build and train the model
        """
        print("\n" + "="*80)
        print("STEP 4: BUILDING MODEL")
        print("="*80)
        
        # Build model
        input_dim = self.train_data['X_src'].shape[1]
        logic_dim = self.train_data['logic'].shape[1]
        
        self.model = build_simple_model(input_dim, logic_dim)
        
        print("\n" + "="*80)
        print("STEP 5: TRAINING MODEL")
        print("="*80)
        
        # ---------------------------------------------------------
        # REMEDIATION STEP 1: Compute Aggressive Sample Weights
        # ---------------------------------------------------------
        print("\n[FIX] Computing Aggressive Sample Weights...")
        # Initialize all weights to 1.0
        weights = np.ones(len(self.train_data['y']), dtype='float32')
        
        # Apply penalties based on ground truth scores
        # If the target score is a 'match' (>0.1), make it 20x more important
        weights[self.train_data['y'] > 0.1] = 20.0
        # If the target score is a 'strong match' (>0.5), make it 50x more important
        weights[self.train_data['y'] > 0.5] = 50.0
        
        print(f"   - Weights Summary: Min={weights.min()}, Max={weights.max()}, Mean={weights.mean():.2f}")
        print("   - The model is now forced to pay attention to rare matches.")
        
        # Prepare callbacks
        model_save_path = os.path.join(self.config['output_dir'], 'best_model.h5')
        callbacks = create_callbacks(
            model_save_path=model_save_path,
            patience=self.config.get('patience', 10)
        )
        
        # ---------------------------------------------------------
        # REMEDIATION STEP 2: Train with Weights
        # ---------------------------------------------------------
        print("\n[FIX] Starting Weighted Training...")
        self.history = self.model.fit(
            [self.train_data['X_src'], self.train_data['X_dst'], self.train_data['logic']],
            self.train_data['y'],
            sample_weight=weights,  # <--- Crucial: Passes the weights to the loss function
            validation_data=(
                [self.val_data['X_src'], self.val_data['X_dst'], self.val_data['logic']],
                self.val_data['y']
            ),
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 64),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
    def evaluate(self):
        """
        Evaluate the trained model
        """
        print("\n" + "="*80)
        print("STEP 6: EVALUATING MODEL")
        print("="*80)
        
        # Evaluate on validation set
        val_loss, val_mse, val_mae = self.model.evaluate(
            [self.val_data['X_src'], self.val_data['X_dst'], self.val_data['logic']],
            self.val_data['y'],
            verbose=0
        )
        
        print(f"\nValidation Metrics:")
        print(f"  Loss (MSE): {val_loss:.6f}")
        print(f"  MAE: {val_mae:.6f}")
        print(f"  RMSE: {np.sqrt(val_mse):.6f}")
        
        # Make predictions
        val_predictions = self.model.predict(
            [self.val_data['X_src'], self.val_data['X_dst'], self.val_data['logic']],
            verbose=0
        ).flatten()
        
        # Check for NaN values
        nan_count = np.isnan(val_predictions).sum()
        inf_count = np.isinf(val_predictions).sum()
        
        if nan_count > 0 or inf_count > 0:
            print(f"\n⚠️  WARNING: Predictions contain invalid values!")
            print(f"  NaN count: {nan_count}/{len(val_predictions)}")
            print(f"  Inf count: {inf_count}/{len(val_predictions)}")
            print(f"  This indicates numerical instability during training.")
            
            # Replace NaN/Inf with 0 for metric calculation
            val_predictions_clean = np.nan_to_num(val_predictions, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            val_predictions_clean = val_predictions
        
        # Calculate additional metrics
        from sklearn.metrics import r2_score, mean_absolute_percentage_error
        
        try:
            r2 = r2_score(self.val_data['y'], val_predictions_clean)
            print(f"  R² Score: {r2:.6f}")
        except Exception as e:
            print(f"  R² Score: Could not calculate (error: {e})")
        
        # Distribution analysis
        print(f"\nPrediction Statistics:")
        valid_preds = val_predictions_clean[~np.isnan(val_predictions)]
        if len(valid_preds) > 0:
            print(f"  Min: {valid_preds.min():.4f}")
            print(f"  Max: {valid_preds.max():.4f}")
            print(f"  Mean: {valid_preds.mean():.4f}")
            print(f"  Median: {np.median(valid_preds):.4f}")
            print(f"  Std: {valid_preds.std():.4f}")
        else:
            print(f"  All predictions are invalid!")
        
        # ---------------------------------------------------------
        # REMEDIATION VERIFICATION: Model Collapse Check
        # ---------------------------------------------------------
        print(f"\n{'='*80}")
        print("MODEL COLLAPSE VERIFICATION")
        print(f"{'='*80}")
        std_dev = val_predictions.std()
        min_pred = val_predictions.min()
        max_pred = val_predictions.max()
        
        if std_dev > 0.05 and min_pred != max_pred:
            print("✅ PASS: Model is predicting a range of values")
            print(f"   - Std Dev: {std_dev:.4f} (> 0.05)")
            print(f"   - Range: [{min_pred:.4f}, {max_pred:.4f}]")
        else:
            print("❌ FAIL: Model may still be collapsed")
            print(f"   - Std Dev: {std_dev:.4f} (should be > 0.05)")
            print(f"   - Range: [{min_pred:.4f}, {max_pred:.4f}]")
            print("   - Consider increasing weights (try 100x)")
        print(f"{'='*80}")
        
        # Save evaluation results
        eval_results = {
            'val_loss': val_loss,
            'val_mse': val_mse,
            'val_mae': val_mae,
            'val_rmse': np.sqrt(val_mse),
            'val_r2': r2,
            'prediction_stats': {
                'min': float(val_predictions.min()),
                'max': float(val_predictions.max()),
                'mean': float(val_predictions.mean()),
                'median': float(np.median(val_predictions)),
                'std': float(val_predictions.std())
            }
        }
        
        import json
        results_path = os.path.join(self.config['output_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nEvaluation results saved to {results_path}")
        
    def plot_training_history(self):
        """
        Plot training history
        """
        print("\n" + "="*80)
        print("STEP 7: PLOTTING TRAINING HISTORY")
        print("="*80)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MAE
        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Training and Validation MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.config['output_dir'], 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {plot_path}")
        
        plt.close()
        
    def run(self):
        """
        Run the complete training pipeline
        """
        start_time = datetime.now()
        print("\n" + "="*80)
        print("STARTING CACRN TRAINING PIPELINE")
        print("="*80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Run pipeline steps
        self.prepare_data()
        self.build_and_train()
        self.evaluate()
        self.plot_training_history()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*80)
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print(f"\nOutputs saved to: {self.config['output_dir']}")


def main():
    """
    Main training function
    """
    # Configuration
    config = {
        # Data paths
        'train_profiles_path': 'data/train.xlsx',  # or 'data/train.csv'
        'target_path': 'data/target.csv',
        
        # Output settings
        'output_dir': 'models',
        
        # Preprocessing settings
        'use_embeddings': True,  # Use sentence embeddings (recommended)
        'embedding_model': 'all-MiniLM-L6-v2',  # Lightweight and fast
        
        # Training settings
        'val_size': 0.2,
        'random_state': 42,
        'epochs': 100,
        'batch_size': 64,
        'patience': 10,  # Early stopping patience
    }
    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
