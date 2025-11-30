# `pip install tabulate` first
import pandas as pd
import numpy as np

# Define file names
RAW_DATA_FILE = 'Final_data.csv'
CLEANED_DATA_FILE = 'Life_Style_Data_Cleaned.csv'


def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: '{file_path}' not found.")
        print("Please run 'dataset.py' first to download the data.")
        return None

    print("DataFrame Info (datatypes and non-null counts):")
    df.info()

    original_columns = df.columns.tolist()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.rename(columns={'name_of_exercise': 'exercise'}, inplace=True)
    new_columns = df.columns.tolist()

    print("Original columns:", original_columns)
    print("New columns:", new_columns)

    duplicates_before = df.duplicated().sum()
    print(f"Found {duplicates_before} duplicate rows.")
    if duplicates_before > 0:
        df.drop_duplicates(inplace=True)
        print(f"Removed duplicates. New shape: {df.shape}")

    missing_values = df.isnull().sum()
    print("Missing values per column:")
    print(missing_values[missing_values > 0])

    print(df.describe(include='all').to_markdown())

    try:
        df.to_csv(CLEANED_DATA_FILE, index=False)
        print(f"\n--- Cleaned data saved to '{CLEANED_DATA_FILE}' ---")
    except Exception as e:
        print(f"\nError saving file: {e}")

    return df


if __name__ == "__main__":
    cleaned_df = load_and_clean_data(RAW_DATA_FILE)
    if cleaned_df is not None:
        print("\nData cleaning and initial exploration complete.")
        print(f"Cleaned DataFrame shape: {cleaned_df.shape}")