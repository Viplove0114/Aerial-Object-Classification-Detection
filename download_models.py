import os
import gdown
from src.config import MODELS_DIR

# ==========================================
# Google Drive File IDs
# ==========================================
# REPLACE THESE WITH YOUR ACTUAL FILE IDs FROM GOOGLE DRIVE
# To get the ID: Right-click file > Share > Copy Link
# Link format: https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing
MODEL_IDS = {
    'custom_model.keras': '1j1n592EdAbae8M2UJqmJL1G8gww73Ihx',
    'transfer_model.keras': '1JyDW0d81I-mq00cKnGkNxpq0WEtMvRV9',
    'best.pt': '1DWUbd9RPXgqmLOlNi6Mj_cmAn60qtjU8'
}

def download_models():
    """
    Downloads trained models from Google Drive if they don't exist locally.
    """
    print("Checking for model files...")
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create YOLO weights directory
    yolo_weights_dir = os.path.join(MODELS_DIR, 'yolov8_results', 'weights')
    os.makedirs(yolo_weights_dir, exist_ok=True)

    for filename, file_id in MODEL_IDS.items():
        # Determine destination path
        if filename == 'best.pt':
            output_path = os.path.join(yolo_weights_dir, filename)
        else:
            output_path = os.path.join(MODELS_DIR, filename)
            
        # Check if file exists
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, output_path, quiet=False)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please check your Google Drive File ID and permissions.")
        else:
            print(f"{filename} already exists. Skipping download.")

if __name__ == "__main__":
    download_models()
