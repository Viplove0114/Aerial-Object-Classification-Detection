import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import os
from src.config import MODELS_DIR, CLASSES, IMG_SIZE
from download_models import download_models

# ==========================================
# Streamlit Page Configuration
# ==========================================
st.set_page_config(page_title="Aerial Object Classifier", layout="wide")

# ==========================================
# Model Download (Auto-run on startup)
# ==========================================
with st.spinner("Checking and downloading models... (This may take a minute on first run)"):
    download_models()

st.title("🦅 Aerial Object Classification & Detection 🚁")
st.markdown("### Classify Birds vs Drones and Detect Objects in Aerial Imagery")

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Classification", "Object Detection"])

# ==========================================
# Helper Functions
# ==========================================

def load_classification_model(model_type):
    """
    Loads the trained Keras classification model.
    
    Args:
        model_type (str): 'custom' or 'transfer'.
        
    Returns:
        model: Loaded Keras model or None if not found.
    """
    model_path = os.path.join(MODELS_DIR, f'{model_type}_model.keras')
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        return None

def preprocess_image(image):
    """
    Preprocesses the uploaded image for the classification model.
    
    Steps:
    1. Resize to target size (224x224).
    2. Normalize pixel values to [0, 1].
    3. Add batch dimension (1, 224, 224, 3).
    
    Args:
        image (PIL.Image): Uploaded image.
        
    Returns:
        np.array: Preprocessed image batch.
    """
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ==========================================
# Classification Mode
# ==========================================
if app_mode == "Classification":
    st.header("Binary Classification: Bird vs Drone")
    
    # Model selection radio button
    model_choice = st.sidebar.radio("Select Model", ["Custom CNN", "Transfer Learning"])
    model_type = 'custom' if model_choice == "Custom CNN" else 'transfer'
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Classify"):
            with st.spinner("Loading model and predicting..."):
                # Load model
                model = load_classification_model(model_type)
                
                if model:
                    # Preprocess and predict
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)[0][0]
                    
                    # Interpret prediction
                    # Sigmoid output: < 0.5 is Class 0 (Bird), >= 0.5 is Class 1 (Drone)
                    label = "Drone" if prediction > 0.5 else "Bird"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    
                    # Display result
                    st.success(f"Prediction: **{label}**")
                    st.info(f"Confidence: {confidence:.2%}")
                else:
                    st.error(f"Model file for {model_choice} not found. Please train the model first.")

# ==========================================
# Object Detection Mode
# ==========================================
elif app_mode == "Object Detection":
    st.header("Object Detection with YOLOv8")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect Objects"):
            with st.spinner("Running YOLOv8..."):
                # Load YOLO model
                # Check for trained model, else use pretrained
                trained_yolo = os.path.join(MODELS_DIR, 'yolov8_results', 'weights', 'best.pt')
                
                if os.path.exists(trained_yolo):
                    model = YOLO(trained_yolo)
                    st.sidebar.success("Using trained YOLOv8 model.")
                else:
                    model = YOLO('yolov8n.pt')
                    st.sidebar.warning("Trained model not found. Using pretrained YOLOv8n (COCO).")
                
                # Run inference
                results = model(image)
                
                # Plot results on the image
                res_plotted = results[0].plot()
                
                # Convert BGR (OpenCV default) to RGB (Streamlit default)
                res_plotted = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Display result
                st.image(res_plotted, caption="Detection Results", use_column_width=True)
