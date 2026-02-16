import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR
import numpy as np

# Initialize (minimal constructor)
ocr = PaddleOCR(lang='en')

# Test with white image
image = np.ones((600, 800, 3), dtype=np.uint8) * 255

# MODERN API: Use .predict() instead of .ocr()
result = ocr.predict(
    image,
    det=True,   # Document detection only
    rec=False,  # No OCR needed for our use case
    cls=False   # No text orientation classification
)

print(f"✓ predict() works! Result type: {type(result)}")
print(f"✓ Number of detected boxes: {len(result) if result else 0}")