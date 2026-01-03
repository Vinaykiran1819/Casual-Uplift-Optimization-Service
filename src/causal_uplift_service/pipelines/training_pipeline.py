import sys
from src.causal_uplift_service.components.data_ingestion import DataIngestion
from src.causal_uplift_service.components.data_transformation import DataTransformation
from src.causal_uplift_service.components.model_trainer import ModelTrainer
from src.causal_uplift_service.exception import CustomException
from src.causal_uplift_service.logger import logging

class TrainingPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            # Step 1: Ingestion (Get Data)
            logging.info("Pipeline Started: Data Ingestion")
            ingestion_obj = DataIngestion()
            data_path = ingestion_obj.initiate_data_ingestion()

            # Step 2: Transformation (Clean & Scale)
            logging.info("Pipeline Step: Data Transformation")
            transform_obj = DataTransformation()
            train_arr, _ = transform_obj.initiate_data_transformation(data_path)

            # Step 3: Training (Train T-Learner)
            logging.info("Pipeline Step: Model Training")
            trainer_obj = ModelTrainer()
            model_path, uplift_metric = trainer_obj.initiate_model_trainer(train_arr)

            print(f"Pipeline Completed Successfully.")
            print(f"Model Saved at: {model_path}")
            print(f"Test Set Average Uplift: {uplift_metric:.4%}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()


# python -m src.causal_uplift_service.pipelines.training_pipeline