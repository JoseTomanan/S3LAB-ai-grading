"""
Test script for ImagePreprocessor: Load to bytes, brighten bytes, and save for visual check.
"""
import os
import cv2
import numpy as np

from main import ImagePreprocessor  # Assuming in main.py

if __name__ == "__main__":
    # Define the image path (update if needed)
    image_path = "dataset/2_enhanced.jpeg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Instantiate the preprocessor
    preprocessor = ImagePreprocessor()
    amount = 0.7

    # Load to bytes and brighten
    original_bytes = preprocessor.load_image(image_path)
    brightened_bytes = preprocessor.brighten(original_bytes, amount)  # amount% brighter
    
    # Save to files for visual inspection
    with open("test_original.jpg", "wb") as f:
        f.write(original_bytes)
    with open("test_brightened.jpg", "wb") as f:
        f.write(brightened_bytes)
    print("Saved: test_original.jpg and test_brightened.jpg")
    
    