import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(test_size=0.2, random_state=42):
    """
    Load and preprocess the breast cancer dataset
    """
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Create DataFrame for better visualization
    feature_names = data.feature_names
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, df, scaler

def get_data_info(df):
    """
    Print dataset information
    """
    print("Dataset Shape:", df.shape)
    print("\nFeatures:", list(df.columns[:-1]))
    print("\nTarget Distribution:")
    print(df['target'].value_counts())
    print(f"\nBenign: {sum(df['target']==0)}")
    print(f"Malignant: {sum(df['target']==1)}")
    
    return df.describe()
