import os
import sys
import numpy as np 
import pandas as pd
import dill
from src.causal_uplift_service.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object (like a model or pipeline) to a file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a pickled object (model or preprocessor) from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)