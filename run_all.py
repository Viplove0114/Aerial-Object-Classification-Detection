import os
import subprocess
import sys

def run_module(module_name):
    """
    Executes a Python module using the current Python interpreter.
    
    Args:
        module_name (str): Name of the module to execute (e.g., 'src.train_classification').
    """
    print(f"==================================================")
    print(f"Running module {module_name}...")
    print(f"==================================================")
    
    try:
        # Run the module and wait for it to complete
        result = subprocess.run([sys.executable, '-m', module_name], check=True)
        print(f"Successfully finished {module_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {module_name}: {e}")
        sys.exit(1)

def main():
    """
    Main function to orchestrate the execution of all project scripts.
    """
    print("Starting full project execution pipeline...\n")
    
    # 1. Train Classification Models (Custom CNN & Transfer Learning)
    run_module('src.train_classification')
    
    # 2. Evaluate Classification Models
    run_module('src.evaluate_classification')
    
    # 3. Train YOLOv8 Object Detection Model
    run_module('src.train_yolo')
    
    print("==================================================")
    print("All tasks completed successfully!")
    print("You can now run the Streamlit app using: streamlit run app.py")
    print("==================================================")

if __name__ == "__main__":
    main()
