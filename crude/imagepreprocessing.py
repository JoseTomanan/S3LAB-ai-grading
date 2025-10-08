"""
Test script for ImagePreprocessor: Load an image, brighten it, and visualize the difference using OpenCV.
"""
import os
import cv2

from main import ImagePreprocessor  # Assuming ImagePreprocessor is defined in main.py

if __name__ == "__main__":
    # Define the image path (update if needed)
    image_path = "dataset/2_enhanced.jpeg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Instantiate the preprocessor
    preprocessor = ImagePreprocessor()
    amount = 0.7
    # Load and process
    image_array = preprocessor.load_image(image_path)
    brightened_image = preprocessor.brighten(image_array, amount)  # amount% brighter
    
    cv2.imshow("Original", image_array)
    cv2.imshow(f"Brightened ({amount*100})", brightened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    original_mean = cv2.mean(image_array)[0] 
    brightened_mean = cv2.mean(brightened_image)[0]
    print(f"Original mean brightness (B channel): {original_mean:.3f}")
    print(f"Brightened mean brightness (B channel): {brightened_mean:.3f}")
    print(f"Increase: {((brightened_mean - original_mean) / original_mean * 100):.1f}%")