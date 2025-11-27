import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import TRAIN_DIR, VALID_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def get_data_generators():
    """
    Creates and returns data generators for training, validation, and testing.
    
    This function uses ImageDataGenerator to load images from directories
    and apply data augmentation to the training set.
    
    Returns:
        train_generator: Iterator for training data with augmentation.
        valid_generator: Iterator for validation data (rescaled only).
        test_generator: Iterator for test data (rescaled only).
    """
    
    # ==========================================
    # Data Augmentation for Training
    # ==========================================
    # We apply various transformations to artificially increase the size
    # of the training dataset and improve model generalization.
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalize pixel values to [0, 1]
        rotation_range=20,      # Randomly rotate images by up to 20 degrees
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2, # Randomly shift images vertically
        shear_range=0.2,        # Apply shear transformation
        zoom_range=0.2,         # Randomly zoom inside images
        horizontal_flip=True,   # Randomly flip images horizontally
        fill_mode='nearest'     # Fill newly created pixels
    )

    # ==========================================
    # Validation and Test Data Preprocessing
    # ==========================================
    # For validation and testing, we only rescale the pixel values.
    # We do NOT apply augmentation here because we want to evaluate
    # on the original images.
    valid_test_datagen = ImageDataGenerator(rescale=1./255)

    # ==========================================
    # Data Generators
    # ==========================================
    
    # Train Generator: Loads images from TRAIN_DIR
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,   # Resize images to 224x224
        batch_size=BATCH_SIZE,  # Number of images per batch
        class_mode='binary',    # Binary classification (Bird vs Drone)
        shuffle=True            # Shuffle data for training
    )

    # Validation Generator: Loads images from VALID_DIR
    valid_generator = valid_test_datagen.flow_from_directory(
        VALID_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False           # No need to shuffle for validation
    )

    # Test Generator: Loads images from TEST_DIR
    test_generator = valid_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False           # No need to shuffle for testing
    )

    return train_generator, valid_generator, test_generator
