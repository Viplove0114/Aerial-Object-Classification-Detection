import os

# ==========================================
# Project Configuration and Constants
# ==========================================

# Base directory of the project
# This helps in constructing absolute paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to the datasets
# DATA_DIR points to the classification dataset (Bird vs Drone)
DATA_DIR = os.path.join(BASE_DIR, 'classification_dataset')
# YOLO_DATA_DIR points to the object detection dataset
YOLO_DATA_DIR = os.path.join(BASE_DIR, 'object_detection_Dataset')

# Classification Data Subdirectories
# These folders contain the images for training, validation, and testing
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Directory to save trained models and training history plots
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# Hyperparameters
# ==========================================

# Image dimensions for the classification models
# Images will be resized to 224x224 pixels
IMG_SIZE = (224, 224)

# Batch size for training
# Reduced to 32 to ensure it runs smoothly on CPU
BATCH_SIZE = 32

# Number of training epochs
# Defines how many times the model sees the entire dataset
EPOCHS = 10

# Learning rate for the optimizer (Adam)
LEARNING_RATE = 0.001

# Class labels for classification
CLASSES = ['bird', 'drone']
