import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass
from src.causal_uplift_service.logger import logging
from src.causal_uplift_service.exception import CustomException

@dataclass
class DataIngestionConfig:
    # We save the cleaned data to artifacts
    raw_data_path: str = os.path.join('artifacts', 'customer_data.csv')
    # NEW WORKING URL (Raw GitHub Content)
    source_url: str = "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    
class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Data Ingestion (Hillstrom Email Dataset)")
        try:
            # --- 1. EXTRACT (Load from URL) ---
            logging.info(f"Downloading data from {self.config.source_url}")
            df = pd.read_csv(self.config.source_url)
            logging.info(f"Download complete. Initial shape: {df.shape}")

            # --- 2. TRANSFORM (Clean & Filter) ---
            
            # A. Filter for Binary Treatment
            # The dataset has 3 groups: 'Mens E-Mail', 'Womens E-Mail', 'No E-Mail'.
            # We only want to compare 'Mens E-Mail' (Treatment) vs 'No E-Mail' (Control).
            df = df[df['segment'] != 'Womens E-Mail'].copy()
            
            # B. Create the 'treatment' column (0 or 1)
            # If segment is 'Mens E-Mail', treatment = 1. Else 0.
            df['treatment'] = df['segment'].apply(lambda x: 1 if x == 'Mens E-Mail' else 0)
            
            # C. Rename columns to match standard conventions
            # 'visit' is usually just a click, 'conversion' is a purchase. We focus on conversion.
            # We drop 'segment' since we now have 'treatment'.
            df = df.drop(columns=['segment'])

            # D. Basic Mapping (Categorical to Numerical)
            # 'urbanicity' is text (Urban/Suburban/Rural). Let's make it numeric or dummies.
            # For simplicity now, we just map them or keep as is. 
            # (We will handle detailed encoding in the Transformation step, but let's do a quick map for safety)
            df['urban'] = df['zip_code'].apply(lambda x: 1 if x == 'Urban' else 0) # Simplified proxy
            
            # --- 3. LOAD (Save to Artifacts) ---
            logging.info("Saving cleaned data to artifacts directory")
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            df.to_csv(self.config.raw_data_path, index=False)
            
            logging.info(f"Ingestion successful. Data saved at {self.config.raw_data_path}")
            return self.config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()





# python -m src.causal_uplift_service.components.data_ingestion