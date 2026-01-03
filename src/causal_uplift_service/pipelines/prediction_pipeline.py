import sys
import os
import pandas as pd
from src.causal_uplift_service.exception import CustomException
from src.causal_uplift_service.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Paths to the artifacts we created during training
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            # 1. Load Artifacts
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # 2. Transform Data (Scale/Encode)
            # Note: We do NOT fit, only transform (using rules learned during training)
            data_scaled = preprocessor.transform(features)

            # 3. Predict Uplift
            # Our custom T-Learner has a 'predict_uplift' method
            preds = model.predict_uplift(data_scaled)
            
            return preds

        except Exception as e:
            raise CustomException(e, sys)

# This class defines the exact inputs expected from the Frontend/API
class CustomData:
    def __init__(self, 
                 recency: int,
                 history: float,
                 mens: int,
                 womens: int,
                 newbie: int,
                 visit: int,
                 zip_code: str,
                 history_segment: str,
                 channel: str):
        
        self.recency = recency
        self.history = history
        self.mens = mens
        self.womens = womens
        self.newbie = newbie
        self.visit = visit
        self.zip_code = zip_code
        self.history_segment = history_segment
        self.channel = channel

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "recency": [self.recency],
                "history": [self.history],
                "mens": [self.mens],
                "womens": [self.womens],
                "newbie": [self.newbie],
                "visit": [self.visit],
                "zip_code": [self.zip_code],
                "history_segment": [self.history_segment],
                "channel": [self.channel],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)