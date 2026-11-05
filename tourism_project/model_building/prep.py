import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_data():
    csv_path = 'tourism_project/data/tourism.csv'

    # 1. Load the dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)

    # 2. Remove unnecessary columns
    # 'Unnamed: 0' (if present) and 'CustomerID' are not predictive features
    cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 3. Simple Missing Value Treatment
    # Handle missing values to avoid issues with downstream training
    # Numerical columns: Fill with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns: Fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Split into features and target
    target_col = 'ProdTaken'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Save locally as CSVs (to be grabbed by the GitHub Action workflow as artifacts)
    X_train.to_csv('Xtrain.csv', index=False)
    X_test.to_csv('Xtest.csv', index=False)
    y_train.to_csv('ytrain.csv', index=False, header=True)
    y_test.to_csv('ytest.csv', index=False, header=True)

    print("Data Preparation Succeeded!")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

if __name__ == '__main__':
    prepare_data()
