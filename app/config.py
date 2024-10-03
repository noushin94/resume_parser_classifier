
import os

class Config:
    # General Configurations
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key')

    # Model Configurations
    MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCHS = 3

    # Paths
    DATA_PATH = 'data/datasets/resumes.csv'
    RESUME_DIR = 'data/resumes/'
    MODEL_SAVE_PATH = 'models/saved_model.bin'
