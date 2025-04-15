import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(df):
    """
    Perform preprocessing on the dataset:
    - Outlier Capping
    - One-hot Encoding
    - SMOTE (for training data only)
    - Return processed features (X) and target (y)
    """

    # 1. Outlier Capping at the 99th Percentile
    outlier_cols = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues']
    for col in outlier_cols:
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=upper_limit)
    print("Outliers capped at 99th percentile ✅")

    # 2. Handle missing values (if any)
    # This uses the SimpleImputer to fill missing values with the median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # For categorical columns, we can fill missing values with the most frequent category
    numerical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # You can scale numerical features here if needed
    ])

    categorical_transformer = Pipeline(steps=[ 
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first'))  # One-hot encoding with dropping the first column to avoid collinearity
    ])

    # Combine numerical and categorical transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[ 
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply preprocessing pipeline to the entire dataset (for both training and prediction)
    df_processed = preprocessor.fit_transform(df)

    # 3. Split the features and target
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']

    return X, y, preprocessor

# Save the preprocessor to a specific location
def save_preprocessor(preprocessor, path='C:/Users/BALA/Downloads/Palindrome/ecommerce_purchase_prediction/models/preprocessor.pkl'):
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved at {path}")

# Example usage
if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('data/raw/online_shoppers_intention.csv')
    
    # Preprocess the data
    X, y, preprocessor = preprocess_data(df)
    
    # Save the preprocessing pipeline (can be used during inference)
    save_preprocessor(preprocessor)