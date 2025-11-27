import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import MODELS_DIR, CLASSES
from src.data_loader import get_data_generators

def evaluate_model(model_type='custom'):
    """
    Evaluates the specified model type on the test set.
    
    Steps:
    1. Load the test data generator.
    2. Load the trained model.
    3. Generate predictions on the test set.
    4. Print classification report (Precision, Recall, F1-score).
    5. Plot and save the confusion matrix.
    
    Args:
        model_type (str): 'custom' or 'transfer'.
    """
    print(f"Evaluating {model_type} model...")
    
    # Get test generator (we ignore train/valid generators here)
    _, _, test_gen = get_data_generators()
    
    # Construct model path
    model_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first.")
        return
        
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Generate predictions
    # We use the generator to get predictions for all test images
    predictions = model.predict(test_gen)
    
    # Convert probabilities to binary class labels (0 or 1)
    # Threshold is 0.5: < 0.5 -> Class 0 (Bird), >= 0.5 -> Class 1 (Drone)
    y_pred = (predictions > 0.5).astype(int).flatten()
    
    # Get true labels from the generator
    y_true = test_gen.classes
    
    # ==========================================
    # Metrics and Reporting
    # ==========================================
    print(f"\n--- {model_type.upper()} MODEL REPORT ---")
    # Print detailed classification metrics
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # ==========================================
    # Confusion Matrix
    # ==========================================
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'{model_type} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot
    save_path = os.path.join(MODELS_DIR, f'{model_type}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    # Evaluate both models
    evaluate_model('custom')
    evaluate_model('transfer')
