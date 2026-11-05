import pandas as pd
import os
import sys

def register_data():
    csv_path = 'tourism_project/data/tourism.csv'

    # 1. Check if dataset exists
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        sys.exit(1)

    print(f"Dataset found at: {csv_path}")

    # 2. Load the dataset
    try:
        df = pd.read_csv(csv_path)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    # 3. Define the expected columns
    expected_columns = [
        'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
        'Occupation', 'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
        'MaritalStatus', 'NumberOfTrips', 'Passport', 'OwnCar',
        'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome',
        'PitchSatisfactionScore', 'ProductPitched', 'NumberOfFollowups', 'DurationOfPitch'
    ]

    # 4. Validate columns
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Validation Failed. Missing columns: {missing_cols}")
        sys.exit(1)
    else:
        print("✓ Validation Succeeded! All expected columns are present.")

    # 5. Print a short summary of the data
    print("\n--- Dataset Summary Info ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\n--- Missing Values Per Column ---")
    print(df[expected_columns].isnull().sum())
    print("\n--- First 5 Rows Preview ---")
    print(df.head())

if __name__ == '__main__':
    register_data()
