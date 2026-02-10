"""
Dataset Builder Module for CACRN
Constructs training pairs from target.csv and merges profile data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DatasetBuilder:
    """
    Builds paired datasets for compatibility prediction
    """
    
    def __init__(self, profiles_path, target_path=None):
        """
        Initialize the dataset builder
        
        Args:
            profiles_path: Path to the profiles CSV/Excel file
            target_path: Path to the target CSV file with pairs and scores (optional for test)
        """
        self.profiles_path = profiles_path
        self.target_path = target_path
        
        # Load data
        print(f"Loading profiles from {profiles_path}")
        if profiles_path.endswith('.xlsx') or profiles_path.endswith('.xls'):
            self.profiles_df = pd.read_excel(profiles_path)
        else:
            self.profiles_df = pd.read_csv(profiles_path)
        
        print(f"Loaded {len(self.profiles_df)} profiles")
        
        if target_path:
            print(f"Loading target pairs from {target_path}")
            self.target_df = pd.read_csv(target_path)
            print(f"Loaded {len(self.target_df)} pairs")
        else:
            self.target_df = None
    
    def build_pairs_dataset(self):
        """
        Build the paired dataset by merging profile data with target pairs
        
        Returns:
            Tuple of (src_profiles_df, dst_profiles_df, targets)
        """
        if self.target_df is None:
            raise ValueError("Target pairs file is required for building pairs dataset")
        
        print("Building pairs dataset...")
        
        # Merge source user profiles
        print("Merging source user profiles...")
        merged_df = self.target_df.merge(
            self.profiles_df,
            left_on='src_user_id',
            right_on='Profile_ID',
            how='left',
            suffixes=('', '_src')
        )
        
        # Merge destination user profiles
        print("Merging destination user profiles...")
        merged_df = merged_df.merge(
            self.profiles_df,
            left_on='dst_user_id',
            right_on='Profile_ID',
            how='left',
            suffixes=('_src', '_dst')
        )
        
        # Separate source and destination profiles
        src_columns = [col for col in merged_df.columns if col.endswith('_src') or 
                      (col in self.profiles_df.columns and not col.endswith('_dst'))]
        
        # Get source features (clean column names)
        src_features_df = pd.DataFrame()
        for col in self.profiles_df.columns:
            if col == 'Profile_ID':
                src_features_df['Profile_ID'] = merged_df['src_user_id']
            elif f"{col}_src" in merged_df.columns:
                src_features_df[col] = merged_df[f"{col}_src"]
            elif col in merged_df.columns and not col.endswith('_dst'):
                src_features_df[col] = merged_df[col]
        
        # Get destination features (clean column names)
        dst_features_df = pd.DataFrame()
        for col in self.profiles_df.columns:
            if col == 'Profile_ID':
                dst_features_df['Profile_ID'] = merged_df['dst_user_id']
            elif f"{col}_dst" in merged_df.columns:
                dst_features_df[col] = merged_df[f"{col}_dst"]
        
        # Get targets
        targets = merged_df['compatibility_score'].values
        
        print(f"Built dataset with {len(src_features_df)} pairs")
        print(f"Source features shape: {src_features_df.shape}")
        print(f"Destination features shape: {dst_features_df.shape}")
        
        return src_features_df, dst_features_df, targets
    
    def generate_test_pairs(self, test_pairs_path=None):
        """
        Generate test pairs for prediction
        
        Args:
            test_pairs_path: Optional path to CSV with test pairs (src_user_id, dst_user_id)
                           If not provided, generates all possible pairs
        
        Returns:
            Tuple of (src_profiles_df, dst_profiles_df, pair_ids_df)
        """
        if test_pairs_path:
            print(f"Loading test pairs from {test_pairs_path}")
            test_pairs_df = pd.read_csv(test_pairs_path)
        else:
            print("Generating all possible test pairs...")
            # Generate all possible pairs (excluding self-pairs)
            user_ids = self.profiles_df['Profile_ID'].unique()
            pairs = []
            for i, src_id in enumerate(user_ids):
                for dst_id in user_ids:
                    if src_id != dst_id:
                        pairs.append({'src_user_id': src_id, 'dst_user_id': dst_id})
            test_pairs_df = pd.DataFrame(pairs)
            print(f"Generated {len(test_pairs_df)} test pairs")
        
        # Merge with profiles
        print("Merging test pairs with profiles...")
        merged_df = test_pairs_df.merge(
            self.profiles_df,
            left_on='src_user_id',
            right_on='Profile_ID',
            how='left',
            suffixes=('', '_src')
        )
        
        merged_df = merged_df.merge(
            self.profiles_df,
            left_on='dst_user_id',
            right_on='Profile_ID',
            how='left',
            suffixes=('_src', '_dst')
        )
        
        # Extract source and destination features
        src_features_df = pd.DataFrame()
        for col in self.profiles_df.columns:
            if col == 'Profile_ID':
                src_features_df['Profile_ID'] = merged_df['src_user_id']
            elif f"{col}_src" in merged_df.columns:
                src_features_df[col] = merged_df[f"{col}_src"]
            elif col in merged_df.columns and not col.endswith('_dst'):
                src_features_df[col] = merged_df[col]
        
        dst_features_df = pd.DataFrame()
        for col in self.profiles_df.columns:
            if col == 'Profile_ID':
                dst_features_df['Profile_ID'] = merged_df['dst_user_id']
            elif f"{col}_dst" in merged_df.columns:
                dst_features_df[col] = merged_df[f"{col}_dst"]
        
        # Keep pair IDs for submission
        pair_ids_df = test_pairs_df[['src_user_id', 'dst_user_id']].copy()
        
        print(f"Test dataset ready with {len(src_features_df)} pairs")
        
        return src_features_df, dst_features_df, pair_ids_df
    
    def split_train_val(self, src_df, dst_df, targets, val_size=0.2, random_state=42):
        """
        Split data into training and validation sets
        
        Args:
            src_df: Source user profiles
            dst_df: Destination user profiles
            targets: Compatibility scores
            val_size: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Tuple of (src_train, src_val, dst_train, dst_val, y_train, y_val)
        """
        indices = np.arange(len(targets))
        
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
        
        src_train = src_df.iloc[train_idx].reset_index(drop=True)
        src_val = src_df.iloc[val_idx].reset_index(drop=True)
        
        dst_train = dst_df.iloc[train_idx].reset_index(drop=True)
        dst_val = dst_df.iloc[val_idx].reset_index(drop=True)
        
        y_train = targets[train_idx]
        y_val = targets[val_idx]
        
        print(f"Train set: {len(y_train)} pairs")
        print(f"Validation set: {len(y_val)} pairs")
        
        return src_train, src_val, dst_train, dst_val, y_train, y_val


def create_submission_file(pair_ids_df, predictions, output_path):
    """
    Create submission file in the required format
    
    Args:
        pair_ids_df: DataFrame with src_user_id and dst_user_id
        predictions: Array of compatibility scores
        output_path: Path to save the submission file
    """
    submission_df = pair_ids_df.copy()
    
    # Create pair ID column
    submission_df['pair_id'] = (
        submission_df['src_user_id'].astype(str) + '_' + 
        submission_df['dst_user_id'].astype(str)
    )
    
    # Add predictions
    submission_df['compatibility_score'] = predictions
    
    # Ensure scores are between 0 and 1
    submission_df['compatibility_score'] = submission_df['compatibility_score'].clip(0, 1)
    
    # Keep only required columns
    submission_df = submission_df[['pair_id', 'compatibility_score']]
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Score range: [{submission_df['compatibility_score'].min():.4f}, "
          f"{submission_df['compatibility_score'].max():.4f}]")
