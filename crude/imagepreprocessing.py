import os
import numpy as np
import cv2

from main import ImagePreprocessor # Assuming in main.py

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
    original_image = preprocessor._decode_bytes(original_bytes)

    # Apply contrast
    contrasted_bytes = preprocessor.adjust_contrast(original_bytes, 10)

    # Apply brighten only, with no contrast
    brighten_bytes = preprocessor.brighten(original_bytes, 50)

    # Apply brighten + contrast
    processed_bytes = preprocessor.brighten(contrasted_bytes, 30)

    # Save images
    with open("test_original.jpg", "wb") as f:
        f.write(original_bytes)
    with open("test_contrasted.jpg", "wb") as f:
        f.write(contrasted_bytes)
    with open("test_brighten.jpg", "wb") as f:
        f.write(brighten_bytes)
    with open("test_processed.jpg", "wb") as f:
        f.write(processed_bytes)

    # Direct

    contrasted_image_direct = cv2.convertScaleAbs(original_image, alpha=10, beta=128*(1 - 10))
    brighten_image_direct = cv2.convertScaleAbs(original_image, alpha=1, beta=50)

    ret1, buffer1 = cv2.imencode('.jpg', contrasted_image_direct)
    ret2, buffer2 = cv2.imencode('.jpg', brighten_image_direct)

    with open("test_contrasted_direct.jpg", "wb") as f:
        f.write(buffer1.tobytes())
    with open("test_brighten_direct.jpg", "wb") as f:
        f.write(buffer2.tobytes())
    
    print("Saved: test_original.jpg, test_contrasted.jpg, test_processed.jpg", "test_brighten.jpg","test_contrasted_direct.jpg","test_brighten_direct.jpg")