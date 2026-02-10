"""
Feature Engineering Module for CACRN
Implements specialized logic features to capture compatibility patterns
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Creates advanced features that capture relationship logic between user pairs
    """
    
    def __init__(self):
        """Initialize the feature engineer"""
        pass
    
    def _parse_semicolon_list(self, text):
        """Parse semicolon-separated string into set of cleaned items"""
        if pd.isna(text) or text == '':
            return set()
        items = str(text).split(';')
        return {item.strip().lower() for item in items if item.strip()}
    
    def calculate_constraint_violation(self, src_row, dst_row):
        """
        Check if either user violates the other's constraints
        
        Returns 1 if violation exists, 0 otherwise
        """
        src_constraints = self._parse_semicolon_list(src_row.get('Constraints', ''))
        dst_constraints = self._parse_semicolon_list(dst_row.get('Constraints', ''))
        
        # Check if dst violates src's constraints
        dst_role = str(src_row.get('Role', '')).lower()
        dst_industry = str(src_row.get('Industry', '')).lower()
        
        src_role = str(dst_row.get('Role', '')).lower()
        src_industry = str(dst_row.get('Industry', '')).lower()
        
        # Check if any constraint keywords appear in the other user's profile
        violation = 0
        
        # Check src constraints against dst
        for constraint in src_constraints:
            if constraint in dst_role or constraint in dst_industry:
                violation = 1
                break
        
        # Check dst constraints against src
        if violation == 0:
            for constraint in dst_constraints:
                if constraint in src_role or constraint in src_industry:
                    violation = 1
                    break
        
        return violation
    
    def calculate_objective_interest_overlap(self, src_row, dst_row):
        """
        Calculate the intersection between objectives and interests
        
        Returns:
            - overlap_count: Number of overlapping items
            - overlap_ratio: Normalized by total unique items
        """
        src_objectives = self._parse_semicolon_list(src_row.get('Business_Objectives', ''))
        src_interests = self._parse_semicolon_list(src_row.get('Business_Interests', ''))
        
        dst_objectives = self._parse_semicolon_list(dst_row.get('Business_Objectives', ''))
        dst_interests = self._parse_semicolon_list(dst_row.get('Business_Interests', ''))
        
        # Cross-match: src objectives with dst interests and vice versa
        overlap_src_to_dst = len(src_objectives.intersection(dst_interests))
        overlap_dst_to_src = len(dst_objectives.intersection(src_interests))
        
        total_overlap = overlap_src_to_dst + overlap_dst_to_src
        
        # Also calculate direct interest overlap
        interest_overlap = len(src_interests.intersection(dst_interests))
        
        # Calculate ratio
        total_unique = len(src_objectives | src_interests | dst_objectives | dst_interests)
        overlap_ratio = total_overlap / total_unique if total_unique > 0 else 0
        
        return {
            'obj_interest_overlap_count': total_overlap,
            'obj_interest_overlap_ratio': overlap_ratio,
            'interest_overlap': interest_overlap
        }
    
    def calculate_seniority_gap(self, src_row, dst_row):
        """
        Calculate the difference in seniority levels
        
        Assumes seniority levels can be mapped to numeric scale
        """
        seniority_order = {
            'intern': 1,
            'junior': 2,
            'mid-level': 3,
            'senior': 4,
            'lead': 5,
            'manager': 6,
            'director': 7,
            'vp': 8,
            'c-level': 9,
            'executive': 9
        }
        
        src_sen = str(src_row.get('Seniority_Level', '')).lower()
        dst_sen = str(dst_row.get('Seniority_Level', '')).lower()
        
        src_level = seniority_order.get(src_sen, 0)
        dst_level = seniority_order.get(dst_sen, 0)
        
        gap = abs(src_level - dst_level)
        
        return {
            'seniority_gap': gap,
            'seniority_gap_normalized': gap / 9.0  # Normalize by max gap
        }
    
    def calculate_role_complementarity(self, src_row, dst_row):
        """
        Check if roles are complementary (e.g., Provider-Seeker, Investor-Startup)
        
        Returns a score indicating complementarity
        """
        src_role = str(src_row.get('Role', '')).lower()
        dst_role = str(dst_row.get('Role', '')).lower()
        
        # Define complementary role pairs
        complementary_pairs = [
            ('provider', 'seeker'),
            ('investor', 'startup'),
            ('investor', 'entrepreneur'),
            ('mentor', 'mentee'),
            ('advisor', 'founder'),
            ('buyer', 'seller'),
            ('client', 'service provider'),
            ('recruiter', 'job seeker')
        ]
        
        complementarity_score = 0
        
        for role_a, role_b in complementary_pairs:
            if (role_a in src_role and role_b in dst_role) or \
               (role_b in src_role and role_a in dst_role):
                complementarity_score = 1
                break
        
        return complementarity_score
    
    def calculate_age_compatibility(self, src_row, dst_row):
        """
        Calculate age difference and compatibility score
        """
        src_age = src_row.get('Age', 0)
        dst_age = dst_row.get('Age', 0)
        
        if src_age == 0 or dst_age == 0:
            return {'age_diff': 0, 'age_diff_normalized': 0}
        
        age_diff = abs(src_age - dst_age)
        
        # Normalize assuming max reasonable age diff is 50 years
        age_diff_normalized = min(age_diff / 50.0, 1.0)
        
        return {
            'age_diff': age_diff,
            'age_diff_normalized': age_diff_normalized
        }
    
    def calculate_company_size_compatibility(self, src_row, dst_row):
        """
        Calculate company size difference
        """
        src_size = src_row.get('Company_Size_Employees', 0)
        dst_size = dst_row.get('Company_Size_Employees', 0)
        
        if src_size == 0 or dst_size == 0:
            return {'size_ratio': 1.0}
        
        # Calculate ratio (smaller/larger)
        size_ratio = min(src_size, dst_size) / max(src_size, dst_size)
        
        return {'size_ratio': size_ratio}
    
    def calculate_location_match(self, src_row, dst_row):
        """
        Check if users are in the same location
        """
        src_location = str(src_row.get('Location_City', '')).lower()
        dst_location = str(dst_row.get('Location_City', '')).lower()
        
        same_location = 1 if src_location == dst_location and src_location != '' else 0
        
        return same_location
    
    def calculate_industry_match(self, src_row, dst_row):
        """
        Check if users are in the same industry
        """
        src_industry = str(src_row.get('Industry', '')).lower()
        dst_industry = str(dst_row.get('Industry', '')).lower()
        
        same_industry = 1 if src_industry == dst_industry and src_industry != '' else 0
        
        return same_industry
    
    def generate_pair_features(self, src_df, dst_df):
        """
        Generate all specialized features for user pairs
        
        Args:
            src_df: DataFrame with source user profiles
            dst_df: DataFrame with destination user profiles
            
        Returns:
            DataFrame with engineered features
        """
        print("Generating specialized pair features...")
        
        n_pairs = len(src_df)
        features = []
        
        for idx in range(n_pairs):
            if idx % 1000 == 0:
                print(f"Processing pair {idx}/{n_pairs}")
            
            src_row = src_df.iloc[idx]
            dst_row = dst_df.iloc[idx]
            
            pair_features = {}
            
            # Calculate all features
            pair_features['constraint_violation'] = self.calculate_constraint_violation(src_row, dst_row)
            
            overlap_features = self.calculate_objective_interest_overlap(src_row, dst_row)
            pair_features.update(overlap_features)
            
            seniority_features = self.calculate_seniority_gap(src_row, dst_row)
            pair_features.update(seniority_features)
            
            pair_features['role_complementarity'] = self.calculate_role_complementarity(src_row, dst_row)
            
            age_features = self.calculate_age_compatibility(src_row, dst_row)
            pair_features.update(age_features)
            
            size_features = self.calculate_company_size_compatibility(src_row, dst_row)
            pair_features.update(size_features)
            
            pair_features['same_location'] = self.calculate_location_match(src_row, dst_row)
            pair_features['same_industry'] = self.calculate_industry_match(src_row, dst_row)
            
            features.append(pair_features)
        
        features_df = pd.DataFrame(features)
        print(f"Generated {len(features_df.columns)} specialized features")
        
        return features_df


def combine_features_with_logic(base_features_src, base_features_dst, logic_features):
    """
    Combine base features (from preprocessing) with logic features
    
    Args:
        base_features_src: Feature matrix for source users (n_pairs, n_features)
        base_features_dst: Feature matrix for destination users (n_pairs, n_features)
        logic_features: DataFrame with logic features (n_pairs, n_logic_features)
        
    Returns:
        Tuple of (src_features, dst_features, logic_features_array)
    """
    logic_features_array = logic_features.values.astype(np.float32)
    
    return base_features_src, base_features_dst, logic_features_array
