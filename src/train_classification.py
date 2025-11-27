import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.config import MODELS_DIR, EPOCHS
from src.data_loader import get_data_generators
from src.models import build_custom_cnn, build_transfer_model

def plot_history(history, model_name):
    """
    Plots training and validation accuracy/loss curves and saves them to a file.
    
    Args:
        history: Keras History object containing training metrics.
        model_name (str): Name of the model (for title and filename).
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'{model_name} - Accuracy')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'{model_name} - Loss')
    
    # Save plot
    save_path = os.path.join(MODELS_DIR, f'{model_name}_history.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def train_model(model_type='custom'):
    """
    Trains the specified model type ('custom' or 'transfer').
    
    Steps:
    1. Get data generators.
    2. Build the model architecture.
    3. Define callbacks (Checkpoint, EarlyStopping).
    4. Train the model.
    5. Plot and save training history.
    
    Args:
        model_type (str): 'custom' for CNN, 'transfer' for MobileNetV2.
    """
    print(f"\nStarting training for {model_type} model...")
    
    # Get data generators
    train_gen, valid_gen, _ = get_data_generators()
    
    # Build model
    if model_type == 'custom':
        model = build_custom_cnn()
    else:
        model = build_transfer_model()
        
    # Define path to save the best model weights
    checkpoint_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    
    # Callbacks
    callbacks = [
        # Save the model with the best validation accuracy
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        # Stop training if validation loss doesn't improve for 3 epochs
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    # Train the model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks
    )
    
    # Plot results
    plot_history(history, model_type)
    print(f"{model_type} model training complete. Saved to {checkpoint_path}")

if __name__ == "__main__":
    # Train both models sequentially
    train_model('custom')
    train_model('transfer')
