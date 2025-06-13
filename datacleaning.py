import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import os

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the dataset CSV into a pandas DataFrame.
    It can load from local file or URL.

    Parameters:
    - file_path: str - Path to the CSV file or URL.

    Returns:
    - df: pandas DataFrame containing the dataset.
    """
    logger.info(f"Loading dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def clean_data(df):
    """
    Clean the dataset:
    - Fill missing 'Age' with median age
    - Fill missing 'Embarked' with mode
    - Fill missing 'Cabin' with 'Unknown'
    - Drop 'Ticket' column
    - Label encode 'Sex' and 'Embarked' columns

    Parameters:
    - df: pandas DataFrame to be cleaned

    Returns:
    - df_clean: cleaned pandas DataFrame
    """
    logger.info("Starting data cleaning...")

    # Fill missing Age with median
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'].fillna(median_age, inplace=True)
        logger.info(f"Filled missing 'Age' with median value {median_age}.")
    else:
        logger.warning("'Age' column not found; skipping imputation.")

    # Fill missing Embarked with mode
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'].fillna(mode_embarked, inplace=True)
        logger.info(f"Filled missing 'Embarked' with mode value '{mode_embarked}'.")
    else:
        logger.warning("'Embarked' column not found; skipping imputation.")

    # Fill missing Cabin with 'Unknown'
    if 'Cabin' in df.columns:
        df['Cabin'].fillna('Unknown', inplace=True)
        logger.info("Filled missing 'Cabin' with 'Unknown'.")
    else:
        logger.warning("'Cabin' column not found; skipping imputation.")

    # Drop Ticket column
    if 'Ticket' in df.columns:
        df.drop(columns=['Ticket'], inplace=True)
        logger.info("Dropped 'Ticket' column.")
    else:
        logger.warning("'Ticket' column not found; skipping drop.")

    # Label encode Sex and Embarked
    label_enc_cols = ['Sex', 'Embarked']
    for col in label_enc_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logger.info(f"Label encoded '{col}' column.")
        else:
            logger.warning(f"Column '{col}' not found; skipping encoding.")

    logger.info("Data cleaning completed.")
    return df

def save_data(df, output_path):
    """
    Save cleaned DataFrame to CSV.

    Parameters:
    - df: pandas DataFrame to save
    - output_path: str - Path to save the CSV file
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise

def run_pipeline(input_file, output_file):
    """
    Execute ETL pipeline: Load data, clean it, and save the result.

    Parameters:
    - input_file: str, path to input CSV
    - output_file: str, path to output CSV
    """
    logger.info("ETL pipeline started.")
    df = load_data(input_file)
    df_clean = clean_data(df)
    save_data(df_clean, output_file)
    logger.info("ETL pipeline finished successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset ETL Pipeline")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file path or URL')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file path for cleaned data')

    args = parser.parse_args()

    try:
        run_pipeline(args.input, args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
