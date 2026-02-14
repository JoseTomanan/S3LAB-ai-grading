import cv2
import numpy as np
import math
import logging
from typing import List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CVProcessingError(Exception):
    """Custom exception for CV processing failures"""
    pass


class CVImagePreprocessor:
    """
    Image preprocessor for student assessment answer box detection.
    Implements the contract required by api.py endpoints.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    DEFAULT_OUTPUT_HEIGHT = 800  # Standard height for processed answer boxes
    
    def __init__(self):
        """Initialize preprocessor with default parameters"""
        pass
    
    # ==============================
    # Core API Methods
    # ==============================
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """
        Validate that file has supported extension.
        
        Args:
            filename: Name of uploaded file
            
        Returns:
            True if extension is supported, False otherwise
        """
        if not filename:
            return False
        ext = Path(filename).suffix.lower()
        return ext in CVImagePreprocessor.SUPPORTED_FORMATS
    
    def process_assessment_image(self, image_bytes: bytes) -> List[bytes]:
        """
        Process raw assessment image and extract answer box regions.
        
        API CONTRACT:
        - Input: raw image bytes (from await file.read())
        - Output: List[bytes] where each element is a JPEG byte array
        - Must detect answer boxes and return each as separate processed image
        - Returns up to 3 candidate regions for manual selection
        
        Args:
            image_bytes: Raw bytes of uploaded image
            
        Returns:
            List of JPEG-encoded byte arrays, one per detected answer box
            (Returns at least 1 box - full image if no boxes detected)
            
        Raises:
            CVProcessingError: If image processing fails
        """
        try:
            # Load image from bytes
            image = self._load_image_from_bytes(image_bytes)
            logger.info(f"Loaded image with shape: {image.shape}")
            
            # Detect answer box candidates
            candidate_boxes = self._detect_answer_box_candidates(image)
            logger.info(f"Detected {len(candidate_boxes)} candidate answer boxes")
            
            # Process each box (brighten, contrast, encode)
            processed_bytes_list = []
            for i, box_image in enumerate(candidate_boxes):
                try:
                    # Apply enhancements
                    enhanced = self.brighten(box_image, amount=0.2)
                    enhanced = self.adjust_contrast(enhanced, amount=1.2)
                    
                    # Encode to JPEG bytes
                    img_bytes = self._encode_to_bytes(enhanced)
                    processed_bytes_list.append(img_bytes)
                    
                    logger.debug(f"Processed box #{i}: {len(img_bytes)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to process box #{i}: {e}")
                    continue
            
            # Fallback: if no boxes detected, return full image
            if not processed_bytes_list:
                logger.warning("No valid answer boxes detected. Returning full image as fallback.")
                full_processed = self.brighten(image, amount=0.2)
                full_processed = self.adjust_contrast(full_processed, amount=1.2)
                processed_bytes_list.append(self._encode_to_bytes(full_processed))
            
            # Limit to 3 boxes max
            return processed_bytes_list[:3]
            
        except CVProcessingError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in process_assessment_image: {e}", exc_info=True)
            raise CVProcessingError(f"Image processing failed: {str(e)}")
    
    def brighten(self, image: np.ndarray, amount: float = 0.2) -> np.ndarray:
        """
        Increase image brightness.
        
        Args:
            image: Input image (BGR format)
            amount: Brightness increase factor (0.0 to 1.0)
            
        Returns:
            Brightened image
        """
        if not 0 <= amount <= 1:
            amount = max(0, min(1, amount))
        
        beta = amount * 255
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    def adjust_contrast(self, image: np.ndarray, amount: float = 1.2) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image (BGR format)
            amount: Contrast multiplier (1.0 = no change, >1.0 = increase)
            
        Returns:
            Contrast-adjusted image
        """
        if amount < 0.1:
            amount = 0.1
        
        # Calculate beta to maintain brightness balance
        beta = 128 * (1 - amount)
        return cv2.convertScaleAbs(image, alpha=amount, beta=beta)
    
    # ==============================
    # Internal Processing Methods
    # ==============================
    def load_image(self, image_path: str) -> bytes:
        """
        Load image (unencoded) and return as bytes
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert bytes to OpenCV image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            OpenCV image (BGR format)
            
        Raises:
            CVProcessingError: If decoding fails
        """
        if not image_bytes or len(image_bytes) == 0:
            raise CVProcessingError("Empty image bytes provided")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise CVProcessingError("Failed to decode image bytes - invalid format")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise CVProcessingError(f"Invalid image format: expected BGR, got shape {image.shape}")
        
        return image
    
    def _encode_to_bytes(self, image: np.ndarray) -> bytes:
        """
        Encode OpenCV image to JPEG bytes.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            JPEG-encoded bytes
            
        Raises:
            CVProcessingError: If encoding fails
        """
        if image is None or image.size == 0:
            raise CVProcessingError("Cannot encode empty image")
        
        success, buffer = cv2.imencode(
            '.jpg', 
            image, 
            [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        
        if not success:
            raise CVProcessingError("Failed to encode image to JPEG")
        
        return buffer.tobytes()
    
    def _detect_answer_box_candidates(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect potential answer box regions in the image.
        """
        orig = image.copy()
        h, w = image.shape[:2]
        image_area = h * w
        
        # Resize for performance while maintaining aspect ratio
        max_dim = 1200
        scale = min(max_dim / max(h, w), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        orig_resized = resized.copy()
        
        # Preprocessing pipeline
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Remove horizontal lines (ruled paper, table borders)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        gray_clean = cv2.subtract(gray, detected_lines)
        
        # NEW: Enhance contrast to make text/boxes more visible
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray_clean)
        
        # Edge detection with ADJUSTED parameters (more sensitive)
        blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
        edged = cv2.Canny(blurred, 20, 80)  # Lowered thresholds
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        candidate_boxes = []
        # LOWERED min_area threshold (from 1% to 0.3%)
        min_area = 0.003 * new_w * new_h
        max_boxes = 5
        
        logger.debug(f"Found {len(contours)} contours, min_area threshold: {min_area:.0f}px")
        
        for contour in contours:
            if len(candidate_boxes) >= max_boxes:
                break
            
            # Approximate contour to polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Filter: must be quadrilateral
            if len(approx) != 4:
                continue
            
            # Filter: minimum area
            area = cv2.contourArea(approx)
            if area < min_area:
                continue
            
            # Filter: reasonable aspect ratio (wider range)
            rect = cv2.minAreaRect(approx)
            box_w, box_h = rect[1]
            if box_w == 0 or box_h == 0:
                continue
            
            aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
            if aspect_ratio > 10:  # Increased from 8
                continue
            
            # Calculate confidence score
            mask = np.zeros_like(gray_clean)
            cv2.drawContours(mask, [approx], -1, 255, -1)
            edge_density = np.sum(edged[mask == 255]) / area
            
            confidence = (
                min(area / (0.2 * new_w * new_h), 1.0) * 0.4 +
                min(max(0, 3 - abs(aspect_ratio - 2)), 1.0) * 0.3 +
                min(edge_density / 100, 1.0) * 0.3
            )
            
            # LOWERED confidence threshold (from 0.3 to 0.2)
            if confidence < 0.2:
                continue
            
            logger.debug(f"Box candidate: area={area:.0f}, aspect={aspect_ratio:.2f}, "
                        f"edges={edge_density:.2f}, confidence={confidence:.3f}")
            
            # Warp perspective
            try:
                warped = self._warp_perspective_box(orig_resized, approx, new_h)
                if warped is not None and warped.size > 0:
                    candidate_boxes.append((confidence, warped))
            except Exception as e:
                logger.warning(f"Failed to warp box: {e}")
                continue
        
        # Sort by confidence
        candidate_boxes.sort(key=lambda x: x[0], reverse=True)
        result_boxes = [box for _, box in candidate_boxes]
        
        # If no boxes found, return full image as single candidate
        if not result_boxes:
            logger.info("No answer boxes detected - using full image")
            aspect = w / h
            out_w = int(self.DEFAULT_OUTPUT_HEIGHT * aspect)
            full_resized = cv2.resize(orig, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
            result_boxes.append(full_resized)
        
        return result_boxes
    
    def _warp_perspective_box(self, image: np.ndarray, pts: np.ndarray, out_height: int) -> Optional[np.ndarray]:
        """
        Apply perspective transform to extract a rectangular box.
        
        Args:
            image: Source image
            pts: 4 corner points of quadrilateral
            out_height: Desired output height
            
        Returns:
            Warped rectangular image, or None if transform fails
        """
        try:
            # Order points consistently: UL, UR, LR, LL
            ordered_pts = self._order_points(pts.reshape(4, 2))
            
            # Calculate aspect ratio
            width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
            width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
            avg_width = (width_top + width_bottom) / 2
            
            height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
            height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
            avg_height = (height_left + height_right) / 2
            
            if avg_height == 0:
                return None
            
            aspect_ratio = avg_width / avg_height
            out_width = int(out_height * aspect_ratio)
            
            # Destination points explicit array with dtype
            dst_pts = np.array([
                [0, 0],
                [out_width, 0],
                [out_width, out_height],
                [0, out_height]
            ], dtype=np.float32)

            # Perspective transform
            M = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
            warped = cv2.warpPerspective(image, M, (out_width, out_height))
            
            return warped
            
        except Exception as e:
            logger.warning(f"Perspective warp failed: {e}")
            return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points in consistent clockwise order: UL, UR, LR, LL.
        Args:
            pts: Array of 4 points (x, y)
        Returns:
            Ordered array of shape (4, 2)
        """
        # Sort by x-coordinate
        sorted_x = pts[np.argsort(pts[:, 0])]
        # Get left-most and right-most points
        left_most = sorted_x[:2]
        right_most = sorted_x[2:]
        # Sort left-most by y to get UL and LL
        left_most_sorted = left_most[np.argsort(left_most[:, 1])]
        ul, ll = left_most_sorted
        # Sort right-most by y to get UR and LR
        right_most_sorted = right_most[np.argsort(right_most[:, 1])]
        ur, lr = right_most_sorted
        
        # creates shape (4, 2) float32 array
        return np.array([ul, ur, lr, ll], dtype=np.float32)
    
    # ==============================
    # Manual Segmentation Support
    # ==============================
    
    def manual_segment_answer_box(
        self, 
        image_bytes: bytes, 
        points: dict
    ) -> bytes:
        """
        Manually segment answer box using provided corner points.
        
        Used by PATCH /api/test_instances/{test_id}/{student_no}/{item_id}
        
        Args:
            image_bytes: Raw image bytes
            points: Dictionary with ul, ur, lr, ll coordinates
            
        Returns:
            JPEG-encoded bytes of segmented and enhanced answer box
            
        Raises:
            CVProcessingError: If segmentation fails
        """
        try:
            # Validate points
            required = ["ul", "ur", "lr", "ll"]
            for corner in required:
                if corner not in points:
                    raise CVProcessingError(f"Missing corner point: {corner}")
                if not all(k in points[corner] for k in ["x", "y"]):
                    raise CVProcessingError(f"Point {corner} missing x/y coordinates")
            
            # Load image
            image = self._load_image_from_bytes(image_bytes)
            
            # Extract points
            src_pts = np.array([
                [points["ul"]["x"], points["ul"]["y"]],
                [points["ur"]["x"], points["ur"]["y"]],
                [points["lr"]["x"], points["lr"]["y"]],
                [points["ll"]["x"], points["ll"]["y"]]
            ], dtype=np.float32)
            
            # Calculate output dimensions
            def calc_aspect(pts):
                w_top = np.linalg.norm(pts[0] - pts[1])
                w_bot = np.linalg.norm(pts[3] - pts[2])
                h_left = np.linalg.norm(pts[0] - pts[3])
                h_right = np.linalg.norm(pts[1] - pts[2])
                return (w_top + w_bot) / (h_left + h_right + 1e-5)
            
            aspect_ratio = calc_aspect(src_pts)
            out_height = self.DEFAULT_OUTPUT_HEIGHT
            out_width = int(out_height * aspect_ratio)
            
            # Destination points
            dst_pts = np.array([
                [0, 0],
                [out_width, 0],
                [out_width, out_height],
                [0, out_height]
            ], dtype=np.float32)
            
            # Perspective transform
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (out_width, out_height))
            
            # Enhance
            enhanced = self.brighten(warped, amount=0.2)
            enhanced = self.adjust_contrast(enhanced, amount=1.2)
            
            # Encode
            return self._encode_to_bytes(enhanced)
            
        except CVProcessingError:
            raise
        except Exception as e:
            logger.error(f"Manual segmentation failed: {e}", exc_info=True)
            raise CVProcessingError(f"Manual segmentation failed: {str(e)}")