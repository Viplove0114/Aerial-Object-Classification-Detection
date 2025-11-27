# Aerial Object Classification & Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://viplove0114-aerial-object-classification-detection-app-nkksj1.streamlit.app/)

**🚀 Live Demo:** [Click here to view the deployed app](https://viplove0114-aerial-object-classification-detection-app-nkksj1.streamlit.app/)

This project provides a deep learning solution to classify aerial images as **Bird** or **Drone** and detect them using YOLOv8.

## Project Structure

- `src/`: Contains source code for data loading, model building, training, and evaluation.
- `models/`: Stores trained models and training history plots.
- `app.py`: Streamlit application for interactive use.
- `requirements.txt`: Project dependencies.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project is configured for CPU execution. If you have a GPU, install `tensorflow-gpu` (or standard `tensorflow` with CUDA) and `torch` with CUDA support.*

2.  **Download Datasets**:
    Download the datasets from the following links and extract them into the project directory:
    - **Classification Dataset**: [Download Link](https://drive.google.com/drive/folders/1msb9EBIBE8R54Rvya7PvLykNvuSrrqDL?usp=drive_link)
    - **Object Detection Dataset**: [Download Link](https://drive.google.com/drive/folders/1DQqV5v-HXXToPDqrwRyOozxxbNprAHo_?usp=drive_link)

    Ensure the directory structure looks like this:
    - `classification_dataset/`
    - `object_detection_Dataset/`

3.  **Download Models (Automatic)**:
    When you run the app (`streamlit run app.py`), it will **automatically download** the trained models from Google Drive if they are not present locally.

## Usage

### 1. One-Click Execution (Recommended)
Run the entire pipeline (Training -> Evaluation -> Object Detection) with a single command:
```bash
python run_all.py
```

### 2. Manual Execution
If you prefer to run steps individually:

#### Train Classification Models
Train both Custom CNN and Transfer Learning (MobileNetV2) models:
```bash
python src/train_classification.py
```
Models will be saved to `models/`.

#### Evaluate Classification Models
Evaluate the trained models on the test set:
```bash
python src/evaluate_classification.py
```

#### Train YOLOv8 Object Detection
Train the YOLOv8 model:
```bash
python src/train_yolo.py
```

#### Run the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```

## Features
- **Classification**: Upload an image to classify it as a Bird or Drone.
- **Object Detection**: Detect and localize birds and drones in images.
- **Model Comparison**: Compare Custom CNN vs Transfer Learning performance.
