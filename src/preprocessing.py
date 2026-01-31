"""
Data Preprocessing Module for Airline Passenger Satisfaction Prediction

This module contains the DataPreprocessor class that handles all data cleaning,
transformation, and preparation steps required before model training.

Author: ML Course Project Team
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive preprocessing pipeline for airline passenger satisfaction data.
    
    This class handles:
    - Data loading with error handling
    - Missing value imputation
    - Categorical encoding (ordinal and one-hot)
    - Feature scaling
    - Train-test split preparation
    
    Attributes:
        train_path (str): Path to training data CSV file
        test_path (str): Path to test data CSV file
        preprocessor (ColumnTransformer): Fitted sklearn preprocessing pipeline
        label_encoder (LabelEncoder): Encoder for target variable
    """
    
    def __init__(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Initialize the DataPreprocessor with file paths.
        
        Args:
            train_path (str): Path to training CSV file
            test_path (str): Path to test CSV file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """
        Load training and test data from CSV files with comprehensive error handling.
        
        Returns:
            tuple: (train_df, test_df) pandas DataFrames
            
        Raises:
            FileNotFoundError: If CSV files are not found
            Exception: For other data loading errors
        """
        try:
            print(f"Loading training data from {self.train_path}...")
            train_df = pd.read_csv(self.train_path)
            print(f"✓ Training data loaded: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
            
            print(f"\nLoading test data from {self.test_path}...")
            test_df = pd.read_csv(self.test_path)
            print(f"✓ Test data loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
            
            return train_df, test_df
            
        except FileNotFoundError as e:
            print("\n" + "="*70)
            print("ERROR: Dataset files not found!")
            print("="*70)
            print("\nPlease follow these steps:")
            print("1. Visit: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction")
            print("2. Download the dataset (you may need a Kaggle account)")
            print("3. Extract train.csv and test.csv")
            print("4. Place them in the 'data/' folder in your project directory")
            print("\nExpected file structure:")
            print("  ml project/")
            print("  ├── data/")
            print("  │   ├── train.csv  ← REQUIRED")
            print("  │   └── test.csv   ← REQUIRED")
            print("="*70)
            raise
            
        except Exception as e:
            print(f"\n✗ Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Strategy:
        - 'Arrival Delay in Minutes': Median imputation (robust to right-skewed distribution)
        - Other columns: Check and report
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with imputed missing values
        """
        print("\nHandling missing values...")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) == 0:
            print("✓ No missing values found")
            return df
        
        print(f"Found missing values in {len(missing_cols)} column(s):")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} missing ({count/len(df)*100:.2f}%)")
        
        # Impute 'Arrival Delay in Minutes' with median
        # Rationale: Delay data is typically right-skewed with outliers
        # Median is more robust than mean for skewed distributions
        if 'Arrival Delay in Minutes' in df.columns and df['Arrival Delay in Minutes'].isnull().any():
            median_delay = df['Arrival Delay in Minutes'].median()
            df['Arrival Delay in Minutes'].fillna(median_delay, inplace=True)
            print(f"✓ Imputed 'Arrival Delay in Minutes' with median: {median_delay:.2f}")
        
        return df
    
    def prepare_features_and_target(self, train_df, test_df):
        """
        Separate features from target variable and prepare for encoding.
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            test_df (pd.DataFrame): Test dataframe
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print("\nPreparing features and target variable...")
        
        # Drop ID columns if they exist (not useful for prediction)
        cols_to_drop = ['Unnamed: 0', 'id']
        for col in cols_to_drop:
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in test_df.columns:
                test_df = test_df.drop(columns=[col])
        
        # Separate target variable
        if 'satisfaction' not in train_df.columns:
            raise ValueError("Target variable 'satisfaction' not found in training data")
        
        X_train = train_df.drop(columns=['satisfaction'])
        y_train = train_df['satisfaction']
        
        X_test = test_df.drop(columns=['satisfaction'])
        y_test = test_df['satisfaction']
        
        print(f"✓ Features: {X_train.shape[1]} columns")
        print(f"✓ Training samples: {len(X_train)}")
        print(f"✓ Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def create_preprocessing_pipeline(self, X_train):
        """
        Create a sklearn preprocessing pipeline with appropriate transformers.
        
        Feature Engineering Strategy:
        - Ordinal Encoding: For 'Class' (maintains economic hierarchy: Eco < Eco Plus < Business)
        - One-Hot Encoding: For 'Gender', 'Customer Type', 'Type of Travel' (nominal categories)
        - Standard Scaling: For all numerical features (normalize for distance-based models)
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
        """
        print("\nCreating preprocessing pipeline...")
        
        # Define feature categories
        # Ordinal feature: Has a natural order
        ordinal_features = ['Class']
        
        # Nominal categorical features: No inherent order
        nominal_features = ['Gender', 'Customer Type', 'Type of Travel']
        
        # Numerical features: All others (includes rating scales 0-5)
        numerical_features = [col for col in X_train.columns 
                             if col not in ordinal_features + nominal_features]
        
        print(f"  - Ordinal features (1): {ordinal_features}")
        print(f"  - Nominal features ({len(nominal_features)}): {nominal_features}")
        print(f"  - Numerical features ({len(numerical_features)}): {len(numerical_features)} columns")
        
        # Create transformers
        # 1. Ordinal Encoder for Class (Eco < Eco Plus < Business)
        ordinal_transformer = OrdinalEncoder(
            categories=[['Eco', 'Eco Plus', 'Business']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        
        # 2. One-Hot Encoder for nominal categories
        onehot_transformer = OneHotEncoder(
            drop='first',  # Avoid multicollinearity
            sparse_output=False,
            handle_unknown='ignore'
        )
        
        # 3. Standard Scaler for numerical features
        numerical_transformer = StandardScaler()
        
        # Combine all transformers into a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('ordinal', ordinal_transformer, ordinal_features),
                ('onehot', onehot_transformer, nominal_features),
                ('numerical', numerical_transformer, numerical_features)
            ],
            remainder='passthrough'  # Keep any other columns as-is
        )
        
        print("✓ Preprocessing pipeline created")
        return preprocessor
    
    def encode_target(self, y_train, y_test):
        """
        Encode target variable from text to numerical labels.
        
        Encoding:
        - 'satisfied' → 1
        - 'neutral or dissatisfied' → 0
        
        Args:
            y_train (pd.Series): Training target
            y_test (pd.Series): Test target
            
        Returns:
            tuple: (y_train_encoded, y_test_encoded) as numpy arrays
        """
        print("\nEncoding target variable...")
        
        # Fit on training data
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Display encoding mapping
        print("✓ Target encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"  - '{label}' → {i}")
        
        # Show class distribution
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        print("\nTraining set class distribution:")
        for label_idx, count in zip(unique, counts):
            label_name = self.label_encoder.classes_[label_idx]
            print(f"  - {label_name}: {count} ({count/len(y_train_encoded)*100:.2f}%)")
        
        return y_train_encoded, y_test_encoded
    
    def load_and_preprocess(self):
        """
        Execute the complete preprocessing pipeline.
        
        This is the main method that orchestrates all preprocessing steps:
        1. Load data
        2. Handle missing values
        3. Separate features and target
        4. Create and fit preprocessing pipeline
        5. Transform features
        6. Encode target variable
        
        Returns:
            tuple: (X_train_transformed, X_test_transformed, y_train_encoded, y_test_encoded)
                  All as numpy arrays ready for model training
        """
        print("="*70)
        print("AIRLINE PASSENGER SATISFACTION - DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        train_df, test_df = self.load_data()
        
        # Step 2: Handle missing values
        train_df = self.handle_missing_values(train_df)
        test_df = self.handle_missing_values(test_df)
        
        # Step 3: Separate features and target
        X_train, X_test, y_train, y_test = self.prepare_features_and_target(train_df, test_df)
        
        # Step 4: Create preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Step 5: Fit and transform features
        print("\nTransforming features...")
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        print(f"✓ Training features shape: {X_train_transformed.shape}")
        print(f"✓ Test features shape: {X_test_transformed.shape}")
        
        # Step 6: Encode target
        y_train_encoded, y_test_encoded = self.encode_target(y_train, y_test)
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE!")
        print("="*70)
        print(f"Ready for model training with {X_train_transformed.shape[1]} features")
        
        return X_train_transformed, X_test_transformed, y_train_encoded, y_test_encoded
    
    def get_feature_names(self):
        """
        Get the names of features after preprocessing transformation.
        
        Returns:
            list: Feature names after encoding
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet. Run load_and_preprocess() first.")
        
        feature_names = []
        
        # Get feature names from each transformer
        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'ordinal':
                feature_names.extend(features)
            elif name == 'onehot':
                if hasattr(transformer, 'get_feature_names_out'):
                    feature_names.extend(transformer.get_feature_names_out(features))
            elif name == 'numerical':
                feature_names.extend(features)
        
        return feature_names


# Convenience function for quick preprocessing
def preprocess_data(train_path='data/train.csv', test_path='data/test.csv'):
    """
    Convenience function to preprocess data in one line.
    
    Args:
        train_path (str): Path to training CSV
        test_path (str): Path to test CSV
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    dp = DataPreprocessor(train_path, test_path)
    X_train, X_test, y_train, y_test = dp.load_and_preprocess()
    return X_train, X_test, y_train, y_test, dp


# Example usage
if __name__ == "__main__":
    print("Testing DataPreprocessor...")
    print("\nNOTE: Make sure you have downloaded the dataset from Kaggle")
    print("and placed train.csv and test.csv in the data/ folder.\n")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run full preprocessing pipeline
        X_train, X_test, y_train, y_test = preprocessor.load_and_preprocess()
        
        print("\n✓ Preprocessing test successful!")
        print(f"\nData shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
    except Exception as e:
        print(f"\n✗ Preprocessing test failed: {str(e)}")






