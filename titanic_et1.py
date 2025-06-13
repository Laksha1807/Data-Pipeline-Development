import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
import sys
import os

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load the Titanic dataset from a local CSV file.
    """
    logger.info(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def clean_data(df):
    """
    Clean the Titanic dataset:
    - Fill missing 'Age' with median age
    - Fill missing 'Embarked' with mode
    - Fill missing 'Cabin' with 'Unknown'
    - Drop 'Ticket' column
    - Label encode 'Sex' and 'Embarked' columns
    """
    logger.info("Starting data cleaning...")

    # Fill missing Age
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    logger.info(f"Filled missing 'Age' with median value {median_age}.")

    # Fill missing Embarked
    if 'Embarked' in df.columns:
        mode_embarked = df['Embarked'].mode()[0]
        df['Embarked'].fillna(mode_embarked, inplace=True)
        logger.info(f"Filled missing 'Embarked' with mode '{mode_embarked}'.")

    # Fill missing Cabin
    if 'Cabin' in df.columns:
        df['Cabin'].fillna('Unknown', inplace=True)
        logger.info("Filled missing 'Cabin' with 'Unknown'.")

    # Drop Ticket
    if 'Ticket' in df.columns:
        df.drop(columns=['Ticket'], inplace=True)
        logger.info("Dropped 'Ticket' column.")

    # Label encode categorical columns
    for col in ['Sex', 'Embarked']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            logger.info(f"Label encoded '{col}' column.")

    logger.info("Data cleaning completed.")
    return df

def engineer_features(df):
    """
    Engineer new features:
    - FamilySize: SibSp + Parch + 1
    - IsAlone: 1 if FamilySize == 1 else 0
    - Title: extracted from Name and label encoded
    """
    logger.info("Starting feature engineering...")

    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    logger.info("Created 'FamilySize' feature.")

    # IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    logger.info("Created 'IsAlone' feature.")

    # Title
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 
         'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    le = LabelEncoder()
    df['Title'] = le.fit_transform(df['Title'].astype(str))
    logger.info("Extracted and encoded 'Title' feature.")

    logger.info("Feature engineering completed.")
    return df

def save_data(df, output_path):
    """
    Save the cleaned DataFrame to a CSV file.
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise

def run_pipeline(input_path, output_path='cleaned_titanic_data.csv'):
    """
    Execute the ETL pipeline: Load, clean, save.
    """
    logger.info("ETL pipeline started.")
    df = load_data(input_path)
    df_clean = clean_data(df)
    df_clean = engineer_features(df_clean)
    save_data(df_clean, output_path)
    logger.info("ETL pipeline finished successfully.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Titanic Dataset ETL Pipeline")
    parser.add_argument('--input', '-i', default='Titanic-Dataset.csv', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='cleaned_titanic_data.csv', help='Output CSV file path')

    args = parser.parse_args()

    try:
        run_pipeline(args.input, args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
