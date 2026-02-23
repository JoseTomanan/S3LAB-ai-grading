import cv2
import numpy as np
from pathlib import Path



class ImageModifier:
    def validate_file_extension(self, filename: str) -> bool:
        """Validate that file extension is allowed"""
        if not filename or not isinstance(filename, str):
            return False
        ext = Path(filename).suffix.lower()
        return ext in [".jpeg", ".jpg", ".png"]
    
    def brighten(self, image: np.ndarray, amount: float = 0.25) -> np.ndarray:
        """Increase image brightness using linear transform"""
        amount = max(0.0, min(1.0, float(amount)))
        beta = amount * 255
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    def adjust_contrast(self, image: np.ndarray, amount: float = 1.3) -> np.ndarray:
        """Adjust image contrast with brightness compensation"""
        amount = max(0.1, float(amount))
        beta = 128 * (1 - amount)
        return cv2.convertScaleAbs(image, alpha=amount, beta=beta)