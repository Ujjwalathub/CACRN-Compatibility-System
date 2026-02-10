"""
Data Preprocessing Module for CACRN
Handles semicolon-separated lists, categorical encoding, and numerical normalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sentence_transformers import SentenceTransformer
import pickle
import os


class DataPreprocessor:
    """
    Comprehensive preprocessor for user profile data
    """
    
    def __init__(self, use_sentence_embeddings=True, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize the preprocessor
        
        Args:
            use_sentence_embeddings: If True, use sentence transformers for text fields
            embedding_model: Name of the sentence transformer model to use
        """
        self.use_sentence_embeddings = use_sentence_embeddings
        self.embedding_model_name = embedding_model
        self.sentence_model = None
        
        # Scalers for numerical features
        self.age_scaler = MinMaxScaler()
        self.company_size_scaler = MinMaxScaler()
        
        # Categorical mappings
        self.role_mapping = {}
        self.seniority_mapping = {}
        self.industry_mapping = {}
        self.location_mapping = {}
        
        # Text field vocabularies (for TF-IDF alternative)
        self.interests_vocab = set()
        self.objectives_vocab = set()
        self.constraints_vocab = set()
        
        self.is_fitted = False
        
    def _load_sentence_model(self):
        """Lazy load the sentence transformer model"""
        if self.sentence_model is None and self.use_sentence_embeddings:
            print(f"Loading sentence transformer model: {self.embedding_model_name}")
            self.sentence_model = SentenceTransformer(self.embedding_model_name)
            
    def _parse_semicolon_list(self, text):
        """Parse semicolon-separated string into list of cleaned items"""
        if pd.isna(text) or text == '':
            return []
        items = str(text).split(';')
        return [item.strip().lower() for item in items if item.strip()]
    
    def _encode_text_field(self, texts, field_name):
        """
        Encode text field using sentence embeddings
        
        Args:
            texts: List of semicolon-separated strings
            field_name: Name of the field (for logging)
            
        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        self._load_sentence_model()
        
        # Parse and join items for each text
        processed_texts = []
        for text in texts:
            items = self._parse_semicolon_list(text)
            if items:
                # Join items with comma for better sentence encoding
                processed_texts.append(", ".join(items))
            else:
                processed_texts.append("none")
        
        print(f"Encoding {field_name} with sentence embeddings...")
        embeddings = self.sentence_model.encode(processed_texts, show_progress_bar=True)
        return embeddings
    
    def _build_vocabulary(self, texts, vocab_set):
        """Build vocabulary from semicolon-separated texts"""
        for text in texts:
            items = self._parse_semicolon_list(text)
            vocab_set.update(items)
    
    def _text_to_multihot(self, text, vocab_list):
        """Convert semicolon-separated text to multi-hot encoding"""
        items = set(self._parse_semicolon_list(text))
        vector = np.zeros(len(vocab_list))
        for idx, vocab_item in enumerate(vocab_list):
            if vocab_item in items:
                vector[idx] = 1
        return vector
    
    def fit(self, df):
        """
        Fit the preprocessor on the training data
        
        Args:
            df: DataFrame with user profile data
        """
        print("Fitting preprocessor on data...")
        
        # Fit numerical scalers
        if 'Age' in df.columns:
            self.age_scaler.fit(df[['Age']].fillna(df['Age'].median()))
        
        if 'Company_Size_Employees' in df.columns:
            self.company_size_scaler.fit(df[['Company_Size_Employees']].fillna(0))
        
        # Build categorical mappings
        if 'Role' in df.columns:
            unique_roles = df['Role'].unique()
            self.role_mapping = {role: idx for idx, role in enumerate(unique_roles)}
        
        if 'Seniority_Level' in df.columns:
            unique_seniority = df['Seniority_Level'].unique()
            self.seniority_mapping = {sen: idx for idx, sen in enumerate(unique_seniority)}
        
        if 'Industry' in df.columns:
            unique_industries = df['Industry'].unique()
            self.industry_mapping = {ind: idx for idx, ind in enumerate(unique_industries)}
        
        if 'Location_City' in df.columns:
            unique_locations = df['Location_City'].unique()
            self.location_mapping = {loc: idx for idx, loc in enumerate(unique_locations)}
        
        # Build vocabularies for text fields (if not using embeddings)
        if not self.use_sentence_embeddings:
            if 'Business_Interests' in df.columns:
                self._build_vocabulary(df['Business_Interests'], self.interests_vocab)
            if 'Business_Objectives' in df.columns:
                self._build_vocabulary(df['Business_Objectives'], self.objectives_vocab)
            if 'Constraints' in df.columns:
                self._build_vocabulary(df['Constraints'], self.constraints_vocab)
        
        self.is_fitted = True
        print("Preprocessing fit complete!")
        
    def transform(self, df):
        """
        Transform the data using fitted preprocessor
        
        Args:
            df: DataFrame with user profile data
            
        Returns:
            Dictionary containing all processed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform!")
        
        print("Transforming data...")
        processed_features = {}
        
        # Numerical features
        if 'Age' in df.columns:
            processed_features['age'] = self.age_scaler.transform(
                df[['Age']].fillna(df['Age'].median())
            ).flatten()
        
        if 'Company_Size_Employees' in df.columns:
            processed_features['company_size'] = self.company_size_scaler.transform(
                df[['Company_Size_Employees']].fillna(0)
            ).flatten()
        
        # Categorical features (as integers for embedding layers)
        if 'Role' in df.columns:
            processed_features['role'] = df['Role'].map(self.role_mapping).fillna(-1).astype(int).values
        
        if 'Seniority_Level' in df.columns:
            processed_features['seniority'] = df['Seniority_Level'].map(
                self.seniority_mapping
            ).fillna(-1).astype(int).values
        
        if 'Industry' in df.columns:
            processed_features['industry'] = df['Industry'].map(
                self.industry_mapping
            ).fillna(-1).astype(int).values
        
        if 'Location_City' in df.columns:
            processed_features['location'] = df['Location_City'].map(
                self.location_mapping
            ).fillna(-1).astype(int).values
        
        # Text features
        if self.use_sentence_embeddings:
            if 'Business_Interests' in df.columns:
                processed_features['interests_emb'] = self._encode_text_field(
                    df['Business_Interests'], 'Business_Interests'
                )
            
            if 'Business_Objectives' in df.columns:
                processed_features['objectives_emb'] = self._encode_text_field(
                    df['Business_Objectives'], 'Business_Objectives'
                )
            
            if 'Constraints' in df.columns:
                processed_features['constraints_emb'] = self._encode_text_field(
                    df['Constraints'], 'Constraints'
                )
        else:
            # Multi-hot encoding alternative
            if 'Business_Interests' in df.columns:
                interests_vocab_list = sorted(list(self.interests_vocab))
                processed_features['interests_multihot'] = np.array([
                    self._text_to_multihot(text, interests_vocab_list)
                    for text in df['Business_Interests']
                ])
            
            if 'Business_Objectives' in df.columns:
                objectives_vocab_list = sorted(list(self.objectives_vocab))
                processed_features['objectives_multihot'] = np.array([
                    self._text_to_multihot(text, objectives_vocab_list)
                    for text in df['Business_Objectives']
                ])
            
            if 'Constraints' in df.columns:
                constraints_vocab_list = sorted(list(self.constraints_vocab))
                processed_features['constraints_multihot'] = np.array([
                    self._text_to_multihot(text, constraints_vocab_list)
                    for text in df['Constraints']
                ])
        
        print("Transformation complete!")
        return processed_features
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)
    
    def save(self, filepath):
        """Save the fitted preprocessor to disk"""
        # Don't save the sentence model (too large), it will be reloaded
        temp_model = self.sentence_model
        self.sentence_model = None
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        self.sentence_model = temp_model
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load a fitted preprocessor from disk"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def create_feature_vector(processed_features, use_embeddings=True):
    """
    Combine all processed features into a single feature matrix
    
    Args:
        processed_features: Dictionary from preprocessor.transform()
        use_embeddings: Whether sentence embeddings were used
        
    Returns:
        numpy array of shape (n_samples, total_features)
    """
    feature_arrays = []
    
    # Numerical features
    if 'age' in processed_features:
        feature_arrays.append(processed_features['age'].reshape(-1, 1))
    
    if 'company_size' in processed_features:
        feature_arrays.append(processed_features['company_size'].reshape(-1, 1))
    
    # Categorical features (as integers - will be embedded in model)
    # Note: These are kept separate for embedding layers
    
    # Text embeddings or multi-hot
    if use_embeddings:
        if 'interests_emb' in processed_features:
            feature_arrays.append(processed_features['interests_emb'])
        if 'objectives_emb' in processed_features:
            feature_arrays.append(processed_features['objectives_emb'])
        if 'constraints_emb' in processed_features:
            feature_arrays.append(processed_features['constraints_emb'])
    else:
        if 'interests_multihot' in processed_features:
            feature_arrays.append(processed_features['interests_multihot'])
        if 'objectives_multihot' in processed_features:
            feature_arrays.append(processed_features['objectives_multihot'])
        if 'constraints_multihot' in processed_features:
            feature_arrays.append(processed_features['constraints_multihot'])
    
    # Concatenate all features
    if feature_arrays:
        return np.hstack(feature_arrays)
    else:
        raise ValueError("No features found to create feature vector!")
