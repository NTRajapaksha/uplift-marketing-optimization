import pandas as pd
import os

# URL for Kevin Hillstrom's Email Marketing Dataset
DATA_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
OUTPUT_FOLDER = "data"
OUTPUT_FILE = "email_marketing_campaign.csv"
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILE)

def load_data():
    print("Downloading dataset...")
    
    # Create data folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    # Load and save data
    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Success! Dataset saved to: {OUTPUT_PATH}")
        print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print("\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    load_data()