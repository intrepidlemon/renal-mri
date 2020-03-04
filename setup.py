import os

from config import config

os.makedirs(config.FEATURES_DIR, exist_ok=True)

os.makedirs(config.DATA, exist_ok=True)
os.makedirs(config.PREPROCESSED_DIR, exist_ok=True)
os.makedirs(config.TRAIN_DIR, exist_ok=True)
os.makedirs(config.TEST_DIR, exist_ok=True)
os.makedirs(config.VALIDATION_DIR, exist_ok=True)

os.makedirs(config.OUTPUT, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.STDOUT_DIR, exist_ok=True)
os.makedirs(config.STDERR_DIR, exist_ok=True)
os.makedirs(config.DATASET_RECORDS, exist_ok=True)
