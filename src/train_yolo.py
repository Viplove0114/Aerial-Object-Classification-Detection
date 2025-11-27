from ultralytics import YOLO
import os
from src.config import MODELS_DIR, YOLO_DATA_DIR

def train_yolo():
    """
    Trains YOLOv8 model for object detection.
    
    This script uses the Ultralytics YOLO library to fine-tune a pre-trained
    YOLOv8 Nano model on the custom dataset.
    
    Steps:
    1. Load a pre-trained YOLOv8n model (Nano version is fastest).
    2. Define the path to the data configuration file (data_absolute.yaml).
    3. Train the model using the specified hyperparameters.
    4. Validate the model and export it to ONNX format.
    """
    print("Starting YOLOv8 training...")
    
    # Load a model
    # 'yolov8n.pt' downloads the pre-trained weights from COCO dataset
    model = YOLO('yolov8n.pt')
    
    # Path to the data configuration file
    # This file contains paths to train/val images and class names
    data_yaml = os.path.join(YOLO_DATA_DIR, 'data_absolute.yaml')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=10,          # Number of training epochs (reduced for CPU)
        imgsz=640,          # Image size for training (standard YOLO size)
        project=MODELS_DIR, # Directory to save results
        name='yolov8_results', # Sub-directory name for this run
        exist_ok=True,      # Overwrite existing run if it exists
        device='cpu'        # Force training on CPU as requested
    )
    
    print("YOLOv8 training complete.")
    
    # Validate the model on the validation set
    metrics = model.val()
    print("Validation Metrics:", metrics)
    
    # Export the model to ONNX format for interoperability
    success = model.export(format='onnx')
    print("Model exported to ONNX:", success)

if __name__ == "__main__":
    train_yolo()
