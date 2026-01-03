import sys
import os
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.causal_uplift_service.exception import CustomException
from src.causal_uplift_service.logger import logging
from src.causal_uplift_service.utils import save_object

@dataclass
class DataTransformationConfig:
    # We will save the preprocessor (pipeline) to this path
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates the pipeline that transforms raw data into model-ready numbers.
        """
        try:
            # 1. Define which columns are which
            numerical_columns = ['recency', 'history', 'mens', 'womens', 'newbie', 'visit']
            categorical_columns = ['zip_code', 'history_segment', 'channel']

            # 2. Pipeline for Numbers: Fill missing values -> Scale them
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # 3. Pipeline for Categories: Fill missing -> One Hot Encode -> Scale
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # 4. Combine them
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path):
        try:
            # Read the clean data
            df = pd.read_csv(train_path)
            logging.info("Read train data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Separate Features (X), Target (y), and Treatment (t)
            target_column_name = "conversion"
            treatment_column_name = "treatment"

            # X = Everything EXCEPT conversion and treatment
            input_feature_df = df.drop(columns=[target_column_name, treatment_column_name], axis=1)
            
            # y = Conversion (Did they buy?)
            target_feature_df = df[target_column_name]
            
            # t = Treatment (Did they get the email?)
            treatment_feature_df = df[treatment_column_name]

            logging.info("Applying preprocessing object on training dataframe")
            
            # Transform X (Features)
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            # Combine them into one big array [X, y, t] for the Model Trainer to use
            # We stack them horizontally using np.c_
            train_arr = np.c_[
                input_feature_arr, 
                np.array(target_feature_df), 
                np.array(treatment_feature_df)
            ]

            # Save the pipeline so we can use it later for predictions
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                self.config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Test run
    obj = DataTransformation()
    train_data_path = os.path.join('artifacts', 'customer_data.csv')
    
    # Run transformation
    train_arr, _ = obj.initiate_data_transformation(train_data_path)
    print("Transformation Complete.")
    print(f"Original Shape: {pd.read_csv(train_data_path).shape}")
    print(f"Transformed Shape: {train_arr.shape}")








# python -m src.causal_uplift_service.components.data_transformation