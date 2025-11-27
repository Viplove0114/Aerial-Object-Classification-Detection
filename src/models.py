import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from src.config import IMG_SIZE

def build_custom_cnn(input_shape=IMG_SIZE + (3,)):
    """
    Builds a custom Convolutional Neural Network (CNN) for binary classification.
    
    Architecture:
    - 3 Convolutional Blocks (Conv2D -> BatchNormalization -> MaxPooling2D)
    - Flatten Layer
    - Dense Layer with Dropout
    - Output Layer (Sigmoid activation)
    
    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # ==========================================
        # Block 1: Feature Extraction
        # ==========================================
        Conv2D(32, (3, 3), activation='relu', padding='same'), # 32 filters
        BatchNormalization(),                                  # Normalize activations
        MaxPooling2D((2, 2)),                                  # Downsample by 2x
        
        # ==========================================
        # Block 2: Feature Extraction
        # ==========================================
        Conv2D(64, (3, 3), activation='relu', padding='same'), # 64 filters
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ==========================================
        # Block 3: Feature Extraction
        # ==========================================
        Conv2D(128, (3, 3), activation='relu', padding='same'), # 128 filters
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # ==========================================
        # Classification Head
        # ==========================================
        Flatten(),                          # Convert 2D features to 1D vector
        Dense(128, activation='relu'),      # Fully connected layer
        Dropout(0.5),                       # Regularization to prevent overfitting
        Dense(1, activation='sigmoid')      # Output layer: 0 (Bird) to 1 (Drone)
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', # Loss function for binary classification
                  metrics=['accuracy'])       # Track accuracy during training
    return model

def build_transfer_model(input_shape=IMG_SIZE + (3,)):
    """
    Builds a Transfer Learning model using MobileNetV2.
    
    We use MobileNetV2 pre-trained on ImageNet as the base feature extractor.
    The base layers are frozen, and we add a custom classification head.
    
    Args:
        input_shape (tuple): Shape of the input image.
        
    Returns:
        model: Compiled Keras model.
    """
    # Load MobileNetV2 without the top (classification) layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze base model layers so they are not updated during training
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)     # Average pooling to reduce dimensions
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
