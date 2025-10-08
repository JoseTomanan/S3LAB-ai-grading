"""
Test script for ImagePreprocessor: Load to bytes, adjust contrast (and brighten), save for visual check.
"""
import os
import numpy as np
import cv2

from main import ImagePreprocessor  # Assuming in main.py

if __name__ == "__main__":
    # Define the image path (update if needed)
    image_path = "dataset/2_enhanced.jpeg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Instantiate the preprocessor
    preprocessor = ImagePreprocessor()
    
    # Load to bytes
    original_bytes = preprocessor.load_image(image_path)
    
    # Apply contrast
    contrasted_bytes = preprocessor.adjust_contrast(original_bytes, alpha=2.5)
    
    # Apply brighten
    processed_bytes = preprocessor.brighten(contrasted_bytes, amount=0.1)
    
    # Save to files for visual inspection
    with open("test_original.jpg", "wb") as f:
        f.write(original_bytes)
    with open("test_contrasted.jpg", "wb") as f:
        f.write(contrasted_bytes)
    with open("test_processed.jpg", "wb") as f:  # Contrast + brighten
        f.write(processed_bytes)
    print("Saved: test_original.jpg, test_contrasted.jpg, test_processed.jpg")