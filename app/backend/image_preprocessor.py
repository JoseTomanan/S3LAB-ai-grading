import cv2
import numpy as np
import math
from typing import List, Tuple

class CVProcessingError(Exception):
    """Custom exception for CV processing failures"""
    pass

def mapp(h):
    """Map corner points to consistent order (TL, TR, BR, BL)"""
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]
    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]
    return hnew

def get_robust_aspect_ratio(coords):
    """Calculate aspect ratio robustly for skewed quadrilaterals"""
    pts = np.array(coords)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    
    side_a1 = pts[1] - pts[0]
    side_a2 = pts[3] - pts[2]
    side_b1 = pts[2] - pts[1]
    side_b2 = pts[0] - pts[3]
    
    w = (np.linalg.norm(side_a1) + np.linalg.norm(side_a2)) / 2
    h = (np.linalg.norm(side_b1) + np.linalg.norm(side_b2)) / 2
    return max(w, h) / min(w, h)

def is_valid_box(approx, image_area: float, min_area_ratio: float = 0.005) -> bool:
    """Filter out degenerate or too-small boxes"""
    if len(approx) != 4:
        return False
    area = cv2.contourArea(approx)
    if area < min_area_ratio * image_area:
        return False
    # Optional: check aspect ratio isn't extreme
    try:
        ratio = get_robust_aspect_ratio(approx.reshape(4, 2))
        if ratio > 10:  # e.g., very long thin shape
            return False
    except:
        return False
    return True

def find_top_boxes(image: cv2.typing.MatLike, top_k: int = 5) -> List[cv2.typing.MatLike]:
    """
    Detect up to `top_k` document-like quadrilateral regions.
    Returns list of warped (perspective-corrected) images.
    """
    orig = image.copy()
    image_h, image_w = image.shape[:2]
    image_area = image_h * image_w
    
    # Resize for performance (maintain aspect ratio)
    scale = 800 / max(image_h, image_w)
    new_w, new_h = int(image_w * scale), int(image_h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    orig_resized = resized.copy()

    # Preprocessing
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Remove horizontal lines (e.g., ruled paper)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))
    detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    gray_no_lines = cv2.subtract(gray, detected_lines)
    
    # Edge detection
    blurred = cv2.GaussianBlur(gray_no_lines, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 50)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    valid_boxes = []
    print(f"Found {len(contours)} contours. Looking for up to {top_k} valid boxes...")
    for c in contours:
        if len(valid_boxes) >= top_k:
            break
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        
        if is_valid_box(approx, image_area):
            print(f"Accepted box #{len(valid_boxes) + 1}, area: {cv2.contourArea(approx):.0f}")
            # Map corners
            approx_mapped = mapp(approx)
            
            # Compute output size
            box_ratio = math.sqrt(get_robust_aspect_ratio(approx_mapped))
            box_length = int(new_h * math.sqrt(box_ratio))
            pts_dst = np.float32([[0, 0], [box_length, 0], [box_length, new_h], [0, new_h]])
            
            # Perspective transform on ORIGINAL RESIZED image
            M = cv2.getPerspectiveTransform(approx_mapped.astype(np.float32), pts_dst)
            warped = cv2.warpPerspective(orig_resized, M, (int(new_h * box_ratio), new_h))
            valid_boxes.append(warped)
    
    if not valid_boxes:
        # Fallback: return full image as single "box"
        print("WARNING: No valid boxes found. Returning full image as fallback.")
        return [orig]
    
    return valid_boxes

class CVImagePreprocessor:
    """Production-ready image preprocessor supporting multi-box extraction."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    
    def load_image_from_bytes(self, image_bytes: bytes) -> cv2.typing.MatLike:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise CVProcessingError("Failed to decode image bytes")
        return image
    
    def encode_to_bytes(self, image: cv2.typing.MatLike) -> bytes:
        success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise CVProcessingError("Failed to encode processed image")
        return buffer.tobytes()
    
    def brighten(self, image: cv2.typing.MatLike, amount: float = 0.2) -> cv2.typing.MatLike:
        return cv2.convertScaleAbs(image, alpha=1.0, beta=amount * 255)
    
    def adjust_contrast(self, image: cv2.typing.MatLike, amount: float = 1.2) -> cv2.typing.MatLike:
        return cv2.convertScaleAbs(image, alpha=amount, beta=128 * (1 - amount))
    
    def process_assessment_image(self, image_bytes: bytes) -> List[bytes]:
        """
        Process image and return **up to 3 candidate cropped regions** as JPEG bytes.
        If no boxes found, returns [full_image_processed].
        """
        image = self.load_image_from_bytes(image_bytes)
        boxes = find_top_boxes(image, top_k=5)
        
        processed_list = []
        for box in boxes:
            brightened = self.brighten(box, amount=0.2)
            contrasted = self.adjust_contrast(brightened, amount=1.2)
            processed_bytes = self.encode_to_bytes(contrasted)
            processed_list.append(processed_bytes)
        
        return processed_list  # List of 1–3 JPEG byte blobs
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        ext = filename.lower().split('.')[-1]
        return f".{ext}" in CVImagePreprocessor.SUPPORTED_FORMATS