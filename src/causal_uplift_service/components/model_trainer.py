import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.causal_uplift_service.exception import CustomException
from src.causal_uplift_service.logger import logging
from src.causal_uplift_service.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class TLearner:
    """
    A custom wrapper for the T-Learner strategy.
    This allows us to save ONE object that contains BOTH models.
    """
    def __init__(self):
        self.model_control = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        self.model_treatment = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    
    def fit(self, X, y, t):
        # 1. Train Control Model (Treatment == 0)
        X_c = X[t == 0]
        y_c = y[t == 0]
        self.model_control.fit(X_c, y_c)
        
        # 2. Train Treatment Model (Treatment == 1)
        X_t = X[t == 1]
        y_t = y[t == 1]
        self.model_treatment.fit(X_t, y_t)
        
    def predict_uplift(self, X):
        # Predict Probabilities
        prob_control = self.model_control.predict_proba(X)[:, 1]
        prob_treatment = self.model_treatment.predict_proba(X)[:, 1]
        
        # Uplift = P(Buy|Treatment) - P(Buy|Control)
        return prob_treatment - prob_control

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array):
        try:
            logging.info("Split training and test input data")
            
            # The array structure is: [Features, Target(y), Treatment(t)]
            # We defined this in data_transformation.py
            X = train_array[:, :-2] # All columns except last two
            y = train_array[:, -2]  # Second to last column
            t = train_array[:, -1]  # Last column

            # Split data (80% Train, 20% Test)
            # Stratify by 'y' to ensure balanced conversion rates in split
            X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
                X, y, t, test_size=0.2, random_state=42, stratify=y
            )

            logging.info("Training T-Learner (Control & Treatment Models)")
            
            # Initialize our Custom Wrapper
            model = TLearner()
            model.fit(X_train, y_train, t_train)
            
            logging.info("Model training complete. Calculating metrics on Test Set...")

            # --- EVALUATION ---
            # Ideally we use Qini Score, but for simplicity in logs we check Uplift Distribution
            uplift_preds = model.predict_uplift(X_test)
            avg_uplift = np.mean(uplift_preds)
            logging.info(f"Average Predicted Uplift on Test Set: {avg_uplift:.4f}")

            # Sanity Check: The model should predict higher uplift for actual buyers in treatment
            # (We will do detailed evaluation in a Notebook later)

            logging.info(f"Saving model to {self.config.trained_model_file_path}")
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=model
            )

            return (
                self.config.trained_model_file_path,
                avg_uplift
            )

        except Exception as e:
            raise CustomException(e, sys)

# --- EXECUTION BLOCK (For Testing) ---
if __name__ == "__main__":
    # Load the transformed data we created in the previous step
    # We need to load it manually here to test this specific file
    import pickle
    
    # Simulating the pipeline flow
    # 1. Run Ingestion (Assumed done)
    # 2. Run Transformation
    from src.causal_uplift_service.components.data_transformation import DataTransformation
    trans_obj = DataTransformation()
    train_data_path = os.path.join('artifacts', 'customer_data.csv')
    train_arr, _ = trans_obj.initiate_data_transformation(train_data_path)
    
    # 3. Run Trainer
    trainer = ModelTrainer()
    model_path, uplift = trainer.initiate_model_trainer(train_arr)
    print(f"Training Complete. Model saved at: {model_path}")
    print(f"Average Uplift: {uplift:.2%}")