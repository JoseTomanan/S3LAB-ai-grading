# import cv2
# import numpy as np
# import math
# import logging
# import uuid
# from typing import List, Tuple, Optional
# from pathlib import Path

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)



# class CVProcessingError(Exception):
#     """Custom exception for CV processing failures"""
#     pass


# class CVImagePreprocessor:
#     """
#     Image preprocessor for student assessment answer box detection.
#     Implements the contract required by api.py endpoints.
#     """
    
#     SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
#     DEFAULT_OUTPUT_HEIGHT = 800  # Standard height for processed answer boxes
    
#     def __init__(self):
#         """Initialize preprocessor with default parameters"""
#         pass
    
#     # ==============================
#     # Core API Methods
#     # ==============================
    
#     @staticmethod
#     def validate_file_extension(filename: str) -> bool:
#         """
#         Validate that file has supported extension.
        
#         Args:
#             filename: Name of uploaded file
            
#         Returns:
#             True if extension is supported, False otherwise
#         """
#         if not filename:
#             return False
#         ext = Path(filename).suffix.lower()
#         return ext in CVImagePreprocessor.SUPPORTED_FORMATS
    
#     def process_assessment_image(self, image_bytes: bytes) -> List[bytes]:
#         """
#         Process raw assessment image and extract answer box regions.
        
#         API CONTRACT:
#         - Input: raw image bytes (from await file.read())
#         - Output: List[bytes] where each element is a JPEG byte array
#         - Must detect answer boxes and return each as separate processed image
#         - Returns up to 3 candidate regions for manual selection
        
#         Args:
#             image_bytes: Raw bytes of uploaded image
            
#         Returns:
#             List of JPEG-encoded byte arrays, one per detected answer box
#             (Returns at least 1 box - full image if no boxes detected)
            
#         Raises:
#             CVProcessingError: If image processing fails
#         """


#         try:
#             # Load image from bytes
#             image = self._load_image_from_bytes(image_bytes)
#             logger.info(f"Loaded image with shape: {image.shape}")
 

#             # Detect answer box candidates
#             candidate_boxes = self._detect_answer_box_candidates(image)
#             logger.info(f"Detected {len(candidate_boxes)} candidate answer boxes")
            
#             # Process each box (brighten, contrast, encode)
#             processed_bytes_list = []
#             for i, box_image in enumerate(candidate_boxes):
#                 try:
#                     # Apply enhancements
#                     enhanced = self.brighten(box_image, amount=0.2)
#                     enhanced = self.adjust_contrast(enhanced, amount=1.2)
                    
#                     # Encode to JPEG bytes
#                     img_bytes = self._encode_to_bytes(enhanced)
#                     processed_bytes_list.append(img_bytes)
                    
#                     logger.debug(f"Processed box #{i}: {len(img_bytes)} bytes")
#                 except Exception as e:
#                     logger.warning(f"Failed to process box #{i}: {e}")
#                     continue
            
#             # Fallback: if no boxes detected, return full image
#             if not processed_bytes_list:
#                 logger.warning("No valid answer boxes detected. Returning full image as fallback.")
#                 full_processed = self.brighten(image, amount=0.2)
#                 full_processed = self.adjust_contrast(full_processed, amount=1.2)
#                 processed_bytes_list.append(self._encode_to_bytes(full_processed))
            
#             # Limit to 3 boxes max
#             return processed_bytes_list[:3]
            
#         except CVProcessingError:
#             raise
#         except Exception as e:
#             logger.error(f"Unexpected error in process_assessment_image: {e}", exc_info=True)
#             raise CVProcessingError(f"Image processing failed: {str(e)}")
    
#     def brighten(self, image: np.ndarray, amount: float = 0.2) -> np.ndarray:
#         """
#         Increase image brightness.
        
#         Args:
#             image: Input image (BGR format)
#             amount: Brightness increase factor (0.0 to 1.0)
            
#         Returns:
#             Brightened image
#         """
#         if not 0 <= amount <= 1:
#             amount = max(0, min(1, amount))
        
#         beta = amount * 255
#         return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
#     def adjust_contrast(self, image: np.ndarray, amount: float = 1.2) -> np.ndarray:
#         """
#         Adjust image contrast.
        
#         Args:
#             image: Input image (BGR format)
#             amount: Contrast multiplier (1.0 = no change, >1.0 = increase)
            
#         Returns:
#             Contrast-adjusted image
#         """
#         if amount < 0.1:
#             amount = 0.1
        
#         # Calculate beta to maintain brightness balance
#         beta = 128 * (1 - amount)
#         return cv2.convertScaleAbs(image, alpha=amount, beta=beta)
    
#     # ==============================
#     # Internal Processing Methods
#     # ==============================
    
#     def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
#         """
#         Convert bytes to OpenCV image.
        
#         Args:
#             image_bytes: Raw image bytes
            
#         Returns:
#             OpenCV image (BGR format)
            
#         Raises:
#             CVProcessingError: If decoding fails
#         """
#         if not image_bytes or len(image_bytes) == 0:
#             raise CVProcessingError("Empty image bytes provided")
        
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if image is None:
#             raise CVProcessingError("Failed to decode image bytes - invalid format")
        
#         if len(image.shape) != 3 or image.shape[2] != 3:
#             raise CVProcessingError(f"Invalid image format: expected BGR, got shape {image.shape}")
        
#         return image
    
#     def _encode_to_bytes(self, image: np.ndarray) -> bytes:
#         """
#         Encode OpenCV image to JPEG bytes.
        
#         Args:
#             image: OpenCV image (BGR format)
            
#         Returns:
#             JPEG-encoded bytes
            
#         Raises:
#             CVProcessingError: If encoding fails
#         """
#         if image is None or image.size == 0:
#             raise CVProcessingError("Cannot encode empty image")
        
#         success, buffer = cv2.imencode(
#             '.jpg', 
#             image, 
#             [int(cv2.IMWRITE_JPEG_QUALITY), 95]
#         )
        
#         if not success:
#             raise CVProcessingError("Failed to encode image to JPEG")
        
#         return buffer.tobytes()
    
#     def _debug_contours(self, image: np.ndarray, contours: List[np.ndarray], 
#                     candidate_boxes: List[Tuple[float, np.ndarray, np.ndarray]], 
#                     name: str = "contours"):
#         """Visualize detected contours and candidate boxes"""
#         import uuid
#         debug_dir = Path("debug_contours")
#         debug_dir.mkdir(exist_ok=True)
        
#         debug_img = image.copy()
        
#         # Draw all contours
#         cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        
#         # Draw candidate boxes with confidence scores
#         for i, (confidence, approx, _) in enumerate(candidate_boxes):
#             # Convert from (4, 1, 2) to (4, 2)
#             if approx.ndim == 3 and approx.shape[1] == 1 and approx.shape[2] == 2:
#                 box_pts = approx.reshape(4, 2).astype(np.int32)
#             elif approx.ndim == 2 and approx.shape[1] == 2:
#                 box_pts = approx.astype(np.int32)
#             else:
#                 logger.warning(f"Unexpected approx shape: {approx.shape}")
#                 continue
                
#             cv2.polylines(debug_img, [box_pts], True, (0, 0, 255), 2)
#             cv2.putText(debug_img, f"Conf: {confidence:.2f}", 
#                     (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         cv2.imwrite(str(debug_dir / f"{name}_{uuid.uuid4().hex[:6]}.jpg"), debug_img)
#         logger.info(f"Saved contour debug visualization to {debug_dir}")

#     def _detect_answer_box_candidates(self, image: np.ndarray) -> List[np.ndarray]:
#         """
#         Detect potential answer box regions in the image.
#         """
#         orig = image.copy()
#         h, w = image.shape[:2]
#         image_area = h * w
        
#         # Resize for performance while maintaining aspect ratio
#         max_dim = 1200
#         scale = min(max_dim / max(h, w), 1.0)
#         new_w, new_h = int(w * scale), int(h * scale)
#         resized = cv2.resize(image, (new_w, new_h))
#         orig_resized = resized.copy()
        
#         # Preprocessing pipeline
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         # Remove horizontal lines (ruled paper, table borders)
#         horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
#         detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
#         gray_clean = cv2.subtract(gray, detected_lines)
        
#         # NEW: Enhance contrast to make text/boxes more visible
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         gray_enhanced = clahe.apply(gray_clean)
        
#         # Edge detection with ADJUSTED parameters (more sensitive)
#         blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
#         edged = cv2.Canny(blurred, 20, 80)  # Lowered thresholds
        
#         # Find contours
#         contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
#         candidate_boxes = []
#         # LOWERED min_area threshold (from 1% to 0.3%)
#         min_area = 0.003 * new_w * new_h
#         max_boxes = 5
        
#         logger.debug(f"Found {len(contours)} contours, min_area threshold: {min_area:.0f}px")
        
#         for contour in contours:
#             if len(candidate_boxes) >= max_boxes:
#                 break
            
#             # Approximate contour to polygon
#             perimeter = cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, 0.012 * perimeter, True)
            
#             # Filter: must be quadrilateral
#             if len(approx) != 4:
#                 continue
            
#             # Filter: minimum area
#             area = cv2.contourArea(approx)
#             if area < min_area:
#                 continue
            
#             # Filter: reasonable aspect ratio (wider range)
#             rect = cv2.minAreaRect(approx)
#             box_w, box_h = rect[1]
#             if box_w == 0 or box_h == 0:
#                 continue
            
#             aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
#             if aspect_ratio > 15:  # Increased from 10
#                 continue
            
#             # Calculate confidence score
#             mask = np.zeros_like(gray_clean)
#             cv2.drawContours(mask, [approx], -1, 255, -1)
#             edge_density = np.sum(edged[mask == 255]) / area
            
#             confidence = (
#                 min(area / (0.2 * new_w * new_h), 1.0) * 0.4 +
#                 min(max(0, 3 - abs(aspect_ratio - 2)), 1.0) * 0.3 +
#                 min(edge_density / 100, 1.0) * 0.3
#             )
            
#             # LOWERED confidence threshold (from 0.2 to 0.12)
#             if confidence < 0.12:
#                 continue
            
#             logger.debug(f"Box candidate: area={area:.0f}, aspect={aspect_ratio:.2f}, "
#                         f"edges={edge_density:.2f}, confidence={confidence:.3f}")
            
#             # Warp perspective
#             try:
#                 logger.debug(f"Contour points: len={len(contour)}, shape={contour.shape}")
#                 logger.debug(f"Approx points: len={len(approx)}, shape={approx.shape}, dtype={approx.dtype}")
#                 warped = self._warp_perspective_box(orig_resized, approx, new_h)
#                 if warped is not None and warped.size > 0:
#                     candidate_boxes.append((confidence, approx, warped))
#             except Exception as e:
#                 logger.warning(f"Failed to warp box: {e}")
#                 continue
        
#         # Sort by confidence
#         candidate_boxes.sort(key=lambda x: x[0], reverse=True)
#         self._debug_contours(orig_resized, contours, candidate_boxes)
#         result_boxes = [warped for _, _, warped in candidate_boxes]
        
#         # If no boxes found, return full image as single candidate
#         if not result_boxes:
#             logger.info("No answer boxes detected - using full image")
#             aspect = w / h
#             out_w = int(self.DEFAULT_OUTPUT_HEIGHT * aspect)
#             full_resized = cv2.resize(orig, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
#             result_boxes.append(full_resized)
        
#         return result_boxes
    
#     def _debug_warp(self, image: np.ndarray, src_pts: np.ndarray, name: str = "warp"):
#         """Visualize point ordering before/after warp for debugging"""
#         import uuid
#         debug_dir = Path("debug_warp")
#         debug_dir.mkdir(exist_ok=True)
        
#         # Draw source points with labels
#         debug_img = image.copy()
#         labels = ['UL', 'UR', 'LR', 'LL']
#         colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        
#         for i, (pt, label, color) in enumerate(zip(src_pts, labels, colors)):
#             cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 10, color, -1)
#             cv2.putText(debug_img, label, (int(pt[0])+15, int(pt[1])+15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
#         # Draw connecting lines
#         for i in range(4):
#             cv2.line(debug_img,
#                     (int(src_pts[i][0]), int(src_pts[i][1])),
#                     (int(src_pts[(i+1)%4][0]), int(src_pts[(i+1)%4][1])),
#                     (255, 255, 255), 3)
        
#         cv2.imwrite(str(debug_dir / f"{name}_{uuid.uuid4().hex[:6]}.jpg"), debug_img)
#         logger.info(f"Saved debug visualization to {debug_dir}")


#     def _warp_perspective_box(self, image: np.ndarray, pts: np.ndarray, out_height: int) -> Optional[np.ndarray]:
#         """
#         Apply perspective transform to extract a rectangular box.
#         Safely handles pts of shape (4,2), (4,1,2), or (8,) — logs and skips invalid.
#         """
        
#         try:
#             # --- SAFETY CHECK: Ensure pts is a 4-point set ---
#             if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
#                 # OpenCV approxPolyDP returns (N, 1, 2) → reshape to (N, 2)
#                 pts = pts.reshape(-1, 2)
#             elif pts.ndim == 2 and pts.shape[1] == 2:
#                 # Already (N, 2)
#                 pass
#             elif pts.ndim == 1 and pts.size == 8:
#                 # Flat array [x0,y0,x1,y1,x2,y2,x3,y3]
#                 pts = pts.reshape(4, 2)
#             else:
#                 logger.error(f"Unexpected pts shape: {pts.shape}, dtype={pts.dtype}, size={pts.size}")
#                 return None

#             if pts.shape[0] != 4:
#                 logger.warning(f"Expected 4 points, got {pts.shape[0]}. Skipping warp.")
#                 return None

#             # Now safe: pts is (4, 2)
#             ordered_pts = self._order_points(pts)  # <-- now expects (4,2)

#             # Calculate dimensions
#             width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
#             width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
#             avg_width = (width_top + width_bottom) / 2.0

#             height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
#             height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
#             avg_height = (height_left + height_right) / 2.0

#             if avg_height < 1.0:
#                 return None

#             aspect_ratio = avg_width / avg_height
#             out_width = max(1, int(out_height * aspect_ratio))

#             # Destination points — use explicit float32 array (Pylance-safe)
#             dst_pts = np.array([
#                 [0.0, 0.0],
#                 [float(out_width), 0.0],
#                 [float(out_width), float(out_height)],
#                 [0.0, float(out_height)]
#             ], dtype=np.float32)

#             src_pts = ordered_pts.astype(np.float32)

#             M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#             warped = cv2.warpPerspective(image, M, (out_width, out_height))
#             return warped

#         except Exception as e:
#             logger.exception(f"Perspective warp failed for pts shape {getattr(pts, 'shape', 'unknown')}: {e}")
#             return None
    
#     def _order_points(self, pts: np.ndarray) -> np.ndarray:
#         """
#         Order 4 points: UL, UR, LR, LL.
#         Input must be (4, 2) float32 or float64.
#         """
#         if pts.shape != (4, 2):
#             raise CVProcessingError(f"_order_points expected (4,2), got {pts.shape}")
#         if pts.dtype not in (np.float32, np.float64, np.int32, np.int64):
#             pts = pts.astype(np.float32)

#         # Sort by y, then x within top/bottom groups (robust for documents)
#         sorted_by_y = pts[np.argsort(pts[:, 1])]
#         top = sorted_by_y[:2]
#         bottom = sorted_by_y[2:]
#         ul, ur = top[np.argsort(top[:, 0])]      # left→right on top
#         ll, lr = bottom[np.argsort(bottom[:, 0])] # left→right on bottom
#         return np.array([ul, ur, lr, ll], dtype=np.float32)

#     def _is_clockwise(self, pts: np.ndarray) -> bool:
#         """Check if points are in clockwise order"""
#         # Calculate cross product of consecutive edges
#         cross_product = 0
#         for i in range(4):
#             p1 = pts[i]
#             p2 = pts[(i + 1) % 4]
#             cross_product += (p2[0] - p1[0]) * (p2[1] + p1[1])
        
#         # Positive cross product indicates clockwise order
#         return cross_product > 0
#     # ==============================
#     # Manual Segmentation Support
#     # ==============================
    
#     def manual_segment_answer_box(
#         self, 
#         image_bytes: bytes, 
#         points: dict
#     ) -> bytes:
#         """
#         Manually segment answer box using provided corner points.
        
#         Used by PATCH /api/test_instances/{test_id}/{student_no}/{item_id}
        
#         Args:
#             image_bytes: Raw image bytes
#             points: Dictionary with ul, ur, lr, ll coordinates
            
#         Returns:
#             JPEG-encoded bytes of segmented and enhanced answer box
            
#         Raises:
#             CVProcessingError: If segmentation fails
#         """
#         try:
#             # Validate points
#             required = ["ul", "ur", "lr", "ll"]
#             for corner in required:
#                 if corner not in points:
#                     raise CVProcessingError(f"Missing corner point: {corner}")
#                 if not all(k in points[corner] for k in ["x", "y"]):
#                     raise CVProcessingError(f"Point {corner} missing x/y coordinates")
            
#             # Load image
#             image = self._load_image_from_bytes(image_bytes)
            
#             # Extract points
#             src_pts = np.array([
#                 [points["ul"]["x"], points["ul"]["y"]],
#                 [points["ur"]["x"], points["ur"]["y"]],
#                 [points["lr"]["x"], points["lr"]["y"]],
#                 [points["ll"]["x"], points["ll"]["y"]]
#             ], dtype=np.float32)
            
#             # Calculate output dimensions
#             def calc_aspect(pts):
#                 w_top = np.linalg.norm(pts[0] - pts[1])
#                 w_bot = np.linalg.norm(pts[3] - pts[2])
#                 h_left = np.linalg.norm(pts[0] - pts[3])
#                 h_right = np.linalg.norm(pts[1] - pts[2])
#                 return (w_top + w_bot) / (h_left + h_right + 1e-5)
            
#             aspect_ratio = calc_aspect(src_pts)
#             out_height = self.DEFAULT_OUTPUT_HEIGHT
#             out_width = int(out_height * aspect_ratio)
            
#             # Destination points
#             dst_pts = np.array([
#                 [0.0, 0.0],
#                 [float(out_width), 0.0],
#                 [float(out_width), float(out_height)],
#                 [0.0, float(out_height)]
#             ], dtype=np.float32)
            
#             # Perspective transform
#             M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#             warped = cv2.warpPerspective(image, M, (out_width, out_height))
            
#             # Enhance
#             enhanced = self.brighten(warped, amount=0.2)
#             enhanced = self.adjust_contrast(enhanced, amount=1.2)
            
#             # Encode
#             return self._encode_to_bytes(enhanced)
            
#         except CVProcessingError:
#             raise
#         except Exception as e:
#             logger.error(f"Manual segmentation failed: {e}", exc_info=True)
#             raise CVProcessingError(f"Manual segmentation failed: {str(e)}")


# IMAGE_PREPROCESSOR_2

# import cv2
# import numpy as np
# import logging
# from typing import List, Tuple, Optional
# from pathlib import Path
# import uuid

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class CVProcessingError(Exception):
#     """Custom exception for CV processing failures"""
#     pass


# class CVImagePreprocessor:
#     """
#     Production-ready image preprocessor for student assessment answer box detection.
    
#     Key improvements from crude preprocessor:
#     - Convex hull + iterative epsilon adjustment for robust 4-point approximation
#     - Morphological closing to connect fragmented box borders
#     - Median-based skew correction (more robust than mean)
#     - Lined paper handling: preserve box borders during detection, remove lines AFTER warp
#     - Fallback to bounding rectangle when polygon approximation fails
    
#     API CONTRACT (required by api.py):
#     - Input: raw image bytes (from await file.read())
#     - Output: List[bytes] where each element is a JPEG byte array
#     - Returns up to 3 candidate regions for manual selection
#     - Returns at least 1 box (full image if no boxes detected)
#     """
    
#     SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
#     DEFAULT_OUTPUT_HEIGHT = 800  # Standard height for processed answer boxes
    
#     def __init__(
#         self,
#         min_area_ratio: float = 0.003,
#         min_confidence: float = 0.10,  # Slightly lowered for better recall
#         max_aspect_ratio: float = 20.0,  # Increased tolerance for diverse boxes
#         debug_mode: bool = False,
#         max_processing_dim: int = 1500  # Increased for better contour detection
#     ):
#         """
#         Initialize preprocessor with configurable parameters.
        
#         Args:
#             min_area_ratio: Minimum box area as fraction of image area (default: 0.3%)
#             min_confidence: Minimum confidence score to accept candidate (default: 0.10)
#             max_aspect_ratio: Maximum allowed aspect ratio (width/height or height/width)
#             debug_mode: Enable debug visualizations (writes to ./debug_* directories)
#             max_processing_dim: Maximum dimension for preprocessing resize (preserves aspect)
#         """
#         self.min_area_ratio = min_area_ratio
#         self.min_confidence = min_confidence
#         self.max_aspect_ratio = max_aspect_ratio
#         self.debug_mode = debug_mode
#         self.max_processing_dim = max_processing_dim
#         self._validate_config()
    
#     def _validate_config(self):
#         """Validate constructor parameters"""
#         if not 0.0 < self.min_area_ratio < 1.0:
#             raise ValueError("min_area_ratio must be between 0 and 1")
#         if not 0.0 < self.min_confidence < 1.0:
#             raise ValueError("min_confidence must be between 0 and 1")
#         if self.max_aspect_ratio < 1.0:
#             raise ValueError("max_aspect_ratio must be >= 1.0")
#         if self.max_processing_dim < 200:
#             raise ValueError("max_processing_dim must be >= 200")
    
#     # ==============================
#     # Core API Methods (Contract with api.py)
#     # ==============================
    
#     @staticmethod
#     def validate_file_extension(filename: str) -> bool:
#         """
#         Validate that file has supported extension.
        
#         Args:
#             filename: Name of uploaded file
            
#         Returns:
#             True if extension is supported, False otherwise
#         """
#         if not filename or not isinstance(filename, str):
#             return False
#         ext = Path(filename).suffix.lower()
#         return ext in CVImagePreprocessor.SUPPORTED_FORMATS
    
#     def process_assessment_image(self, image_bytes: bytes) -> List[bytes]:
#         """
#         Process raw assessment image and extract answer box regions.
        
#         API CONTRACT:
#         - Input: raw image bytes (from await file.read())
#         - Output: List[bytes] where each element is a JPEG byte array
#         - Must detect answer boxes and return each as separate processed image
#         - Returns up to 3 candidate regions for manual selection
        
#         Args:
#             image_bytes: Raw bytes of uploaded image
            
#         Returns:
#             List of JPEG-encoded byte arrays, one per detected answer box
#             (Returns at least 1 box - full image if no boxes detected)
            
#         Raises:
#             CVProcessingError: If image processing fails catastrophically
#         """
#         try:
#             # Load image from bytes
#             image = self._load_image_from_bytes(image_bytes)
#             logger.info(f"Loaded image with shape: {image.shape}")
            
#             # Detect answer box candidates
#             candidate_boxes = self._detect_answer_box_candidates(image)
#             logger.info(f"Detected {len(candidate_boxes)} candidate answer boxes")
            
#             # Process each box (brighten, contrast, line removal, deskew, encode)
#             processed_bytes_list = []
#             for i, box_image in enumerate(candidate_boxes):
#                 try:
#                     # Apply enhancements
#                     enhanced = self.brighten(box_image, amount=0.25)
#                     enhanced = self.adjust_contrast(enhanced, amount=1.3)
                    
#                     # CRITICAL: Remove horizontal lines AFTER warp (preserves box borders during detection)
#                     enhanced = self._remove_horizontal_lines_post_warp(enhanced)
                    
#                     # Deskew with median angle (more robust than mean)
#                     enhanced = self._deskew_image(enhanced, max_angle=15.0)
                    
#                     # Encode to JPEG bytes
#                     img_bytes = self._encode_to_bytes(enhanced)
#                     processed_bytes_list.append(img_bytes)
#                     logger.debug(f"Processed box #{i}: {len(img_bytes)} bytes")
#                 except Exception as e:
#                     logger.warning(f"Failed to process box #{i}: {e}")
#                     continue
            
#             # Fallback: if no boxes detected, return full image
#             if not processed_bytes_list:
#                 logger.warning("No valid answer boxes detected. Returning full image as fallback.")
#                 full_processed = self.brighten(image, amount=0.25)
#                 full_processed = self.adjust_contrast(full_processed, amount=1.3)
#                 full_processed = self._remove_horizontal_lines_post_warp(full_processed)
#                 full_processed = self._deskew_image(full_processed, max_angle=15.0)
#                 processed_bytes_list.append(self._encode_to_bytes(full_processed))
            
#             # Limit to 3 boxes max (sorted by confidence during detection)
#             return processed_bytes_list[:3]
            
#         except CVProcessingError:
#             raise
#         except Exception as e:
#             logger.error(f"Unexpected error in process_assessment_image: {e}", exc_info=True)
#             raise CVProcessingError(f"Image processing failed: {str(e)}")
    
#     def brighten(self, image: np.ndarray, amount: float = 0.25) -> np.ndarray:
#         """
#         Increase image brightness using linear transform.
        
#         Args:
#             image: Input image (BGR format)
#             amount: Brightness increase factor (0.0 to 1.0)
            
#         Returns:
#             Brightened image (BGR format)
#         """
#         if image is None or image.size == 0:
#             raise CVProcessingError("Cannot brighten empty image")
        
#         amount = max(0.0, min(1.0, float(amount)))
#         beta = amount * 255
#         return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
#     def adjust_contrast(self, image: np.ndarray, amount: float = 1.3) -> np.ndarray:
#         """
#         Adjust image contrast with brightness compensation.
        
#         Args:
#             image: Input image (BGR format)
#             amount: Contrast multiplier (1.0 = no change, >1.0 = increase)
            
#         Returns:
#             Contrast-adjusted image (BGR format)
#         """
#         if image is None or image.size == 0:
#             raise CVProcessingError("Cannot adjust contrast on empty image")
        
#         amount = max(0.1, float(amount))
#         beta = 128 * (1 - amount)  # Compensate brightness shift
#         return cv2.convertScaleAbs(image, alpha=amount, beta=beta)
    
#     # ==============================
#     # Internal Processing Methods
#     # ==============================
    
#     def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
#         """
#         Convert bytes to OpenCV image with validation.
        
#         Args:
#             image_bytes: Raw image bytes
            
#         Returns:
#             OpenCV image (BGR format)
            
#         Raises:
#             CVProcessingError: If decoding fails or format is invalid
#         """
#         if not image_bytes or len(image_bytes) == 0:
#             raise CVProcessingError("Empty image bytes provided")
        
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if image is None:
#             raise CVProcessingError(
#                 "Failed to decode image bytes - invalid format or corrupted data"
#             )
        
#         if len(image.shape) != 3 or image.shape[2] != 3:
#             raise CVProcessingError(
#                 f"Invalid image format: expected 3-channel BGR, got shape {image.shape}"
#             )
        
#         return image
    
#     def _encode_to_bytes(self, image: np.ndarray) -> bytes:
#         """
#         Encode OpenCV image to high-quality JPEG bytes.
        
#         Args:
#             image: OpenCV image (BGR format)
            
#         Returns:
#             JPEG-encoded bytes
            
#         Raises:
#             CVProcessingError: If encoding fails
#         """
#         if image is None or image.size == 0:
#             raise CVProcessingError("Cannot encode empty image")
        
#         success, buffer = cv2.imencode(
#             '.jpg', 
#             image, 
#             [int(cv2.IMWRITE_JPEG_QUALITY), 95]
#         )
        
#         if not success:
#             raise CVProcessingError("Failed to encode image to JPEG")
        
#         return buffer.tobytes()
    
#     def _detect_answer_box_candidates(self, image: np.ndarray) -> List[np.ndarray]:
#         """
#         Detect potential answer box regions with robust contour handling.
        
#         Key improvements from crude preprocessor:
#         1. Morphological closing to connect fragmented edges
#         2. Convex hull before polygon approximation (eliminates concavities)
#         3. Iterative epsilon adjustment to force 4-point approximation
#         4. Fallback to bounding rectangle when approximation fails
        
#         Returns:
#             List of warped box images sorted by confidence (highest first)
#         """
#         orig = image.copy()
#         h, w = image.shape[:2]
#         image_area = h * w
        
#         # Resize for performance while maintaining aspect ratio
#         scale = min(self.max_processing_dim / max(h, w), 1.0)
#         new_w, new_h = int(w * scale), int(h * scale)
#         resized = cv2.resize(image, (new_w, new_h))
#         orig_resized = resized.copy()
        
#         # Preprocessing pipeline
#         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
#         # CRITICAL CHANGE: DO NOT remove horizontal lines BEFORE detection
#         # (Preserves box borders on lined paper during contour detection)
#         # Horizontal line removal happens POST-warp instead
        
#         # Enhance contrast to make text/boxes more visible
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Increased clipLimit
#         gray_enhanced = clahe.apply(gray)
        
#         # Edge detection with tuned thresholds
#         blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
#         edged = cv2.Canny(blurred, 30, 100)  # Slightly higher thresholds for cleaner edges
        
#         # MORPHOLOGICAL IMPROVEMENT: Close gaps in edges before contour detection
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#         edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
#         edged = cv2.dilate(edged, kernel, iterations=1)  # Strengthen edges
        
#         # Find and sort contours by area (largest first)
#         contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
#         candidate_boxes = []
#         min_area = self.min_area_ratio * new_w * new_h
#         max_boxes = 5
        
#         logger.debug(
#             f"Found {len(contours)} contours; min_area threshold: {min_area:.0f}px "
#             f"(ratio: {self.min_area_ratio})"
#         )
        
#         for contour in contours:
#             if len(candidate_boxes) >= max_boxes:
#                 break
            
#             # FILTER: Minimum area (before expensive operations)
#             if cv2.contourArea(contour) < min_area:
#                 continue
            
#             # CONVEX HULL IMPROVEMENT: Eliminate concavities before approximation
#             hull = cv2.convexHull(contour)
#             hull_perimeter = cv2.arcLength(hull, True)
            
#             # ITERATIVE EPSILON ADJUSTMENT: Force 4-point approximation
#             epsilon = 0.02 * hull_perimeter
#             max_epsilon = 0.25 * hull_perimeter  # Aggressive simplification cap
#             approx = cv2.approxPolyDP(hull, epsilon, True)
#             num_sides = len(approx)
            
#             # Iteratively increase epsilon until we get ≤4 sides
#             while num_sides > 4 and epsilon < max_epsilon:
#                 epsilon *= 1.5
#                 approx = cv2.approxPolyDP(hull, epsilon, True)
#                 num_sides = len(approx)
            
#             # FINAL FALLBACK: If still not 4 points, use bounding rectangle corners
#             if num_sides != 4:
#                 # Get bounding rectangle of hull
#                 x, y, bw, bh = cv2.boundingRect(hull)
#                 approx = np.array([
#                     [[x, y]],                          # Top-left
#                     [[x + bw, y]],                     # Top-right
#                     [[x + bw, y + bh]],                # Bottom-right
#                     [[x, y + bh]]                      # Bottom-left
#                 ], dtype=np.int32)
#                 logger.debug(f"Used bounding rect fallback for contour (area={cv2.contourArea(contour):.0f})")
            
#             # Now we have 4 points - proceed with filtering
#             area = cv2.contourArea(approx)
#             if area < min_area:
#                 continue
            
#             # Filter: reasonable aspect ratio (bidirectional check)
#             rect = cv2.minAreaRect(approx)
#             box_w, box_h = rect[1]
#             if box_w < 1 or box_h < 1:
#                 continue
            
#             aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
#             if aspect_ratio > self.max_aspect_ratio:
#                 continue
            
#             # Calculate confidence score
#             mask = np.zeros_like(gray_enhanced)
#             cv2.drawContours(mask, [approx], -1, 255, -1)
#             edge_density = np.sum(edged[mask == 255]) / (area + 1e-5)
            
#             # Confidence components (less biased toward specific aspect ratios)
#             area_score = min(area / (0.3 * new_w * new_h), 1.0)  # Expect ~30% of image
#             aspect_score = 1.0 - min(abs(aspect_ratio - 2.5) / 6.0, 1.0)  # Tolerant around 2.5:1
#             edge_score = min(edge_density / 120.0, 1.0)
            
#             confidence = (
#                 area_score * 0.4 +
#                 aspect_score * 0.35 +
#                 edge_score * 0.25
#             )
            
#             if confidence < self.min_confidence:
#                 continue
            
#             logger.debug(
#                 f"Box candidate: area={area:.0f}px ({area/image_area*100:.2f}%), "
#                 f"aspect={aspect_ratio:.2f}, edges={edge_density:.2f}, "
#                 f"confidence={confidence:.3f}"
#             )
            
#             # Warp perspective with robust point ordering
#             try:
#                 warped = self._warp_perspective_box(orig_resized, approx, new_h)
#                 if warped is not None and warped.size > 0:
#                     candidate_boxes.append((confidence, approx, warped))
#             except Exception as e:
#                 logger.warning(f"Failed to warp box: {e}")
#                 continue
        
#         # Sort by confidence (highest first)
#         candidate_boxes.sort(key=lambda x: x[0], reverse=True)
        
#         # Debug visualization (only if enabled)
#         if self.debug_mode:
#             self._debug_detection(orig_resized, contours, candidate_boxes)
        
#         result_boxes = [warped for _, _, warped in candidate_boxes]
        
#         # Fallback: return full image if no boxes found
#         if not result_boxes:
#             logger.info("No answer boxes detected - using full image as fallback")
#             aspect = w / h
#             out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
#             full_resized = cv2.resize(orig, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
#             result_boxes.append(full_resized)
        
#         return result_boxes
    
#     def _order_points(self, pts: np.ndarray) -> np.ndarray:
#         """
#         Order 4 points for perspective transform at ANY rotation angle.
        
#         Uses centroid-angle sorting to handle 0°-360° rotations robustly.
#         Returns points in order: [top-left, top-right, bottom-right, bottom-left]
        
#         Args:
#             pts: Array of shape (4, 2) with corner coordinates
            
#         Returns:
#             Ordered points array shape (4, 2) float32
#         """
#         if pts.shape[0] != 4:
#             raise CVProcessingError(
#                 f"_order_points requires exactly 4 points, got {pts.shape[0]}"
#             )
        
#         # Ensure float32 for OpenCV compatibility
#         if pts.dtype not in (np.float32, np.float64):
#             pts = pts.astype(np.float32)
        
#         # Handle common OpenCV contour shapes
#         if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
#             pts = pts.reshape(4, 2)
#         elif pts.ndim == 1 and pts.size == 8:
#             pts = pts.reshape(4, 2)
        
#         if pts.shape != (4, 2):
#             raise CVProcessingError(
#                 f"Unexpected points shape {pts.shape} - expected (4, 2)"
#             )
        
#         # Compute centroid of the quadrilateral
#         centroid = np.mean(pts, axis=0)
        
#         # Sort points by angle around centroid (counter-clockwise)
#         angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
#         sorted_pts = pts[np.argsort(angles)]
        
#         # Rotate array so that top-left (min x+y) is first point
#         tl_idx = np.argmin(sorted_pts[:, 0] + sorted_pts[:, 1])
#         ordered = np.roll(sorted_pts, -tl_idx, axis=0)
        
#         return ordered.astype(np.float32)
    
#     def _warp_perspective_box(
#         self, 
#         image: np.ndarray, 
#         pts: np.ndarray, 
#         out_height: int
#     ) -> Optional[np.ndarray]:
#         """
#         Apply perspective transform with rotation-agnostic point ordering.
        
#         Handles all common OpenCV contour shapes and validates geometry.
        
#         Args:
#             image: Source image (resized version)
#             pts: Contour points from cv2.approxPolyDP
#             out_height: Target height for warped output
            
#         Returns:
#             Warped image array or None if transform fails
#         """
#         try:
#             # Normalize points to (4, 2) shape
#             if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
#                 pts = pts.reshape(-1, 2)
#             elif pts.ndim == 1 and pts.size == 8:
#                 pts = pts.reshape(4, 2)
            
#             if pts.shape[0] != 4 or pts.shape[1] != 2:
#                 logger.error(f"Invalid points shape for warp: {pts.shape}")
#                 return None
            
#             # Order points robustly for any rotation
#             ordered_pts = self._order_points(pts)
            
#             # Calculate dimensions from ordered points
#             width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
#             width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
#             avg_width = (width_top + width_bottom) / 2.0
            
#             height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
#             height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
#             avg_height = (height_left + height_right) / 2.0
            
#             if avg_height < 1.0 or avg_width < 1.0:
#                 logger.warning("Degenerate box dimensions - skipping warp")
#                 return None
            
#             # Handle portrait vs landscape orientation intelligently
#             aspect_ratio = avg_width / avg_height
#             is_portrait = aspect_ratio < 0.8  # More tolerant heuristic
            
#             if is_portrait:
#                 # Swap dimensions for portrait boxes
#                 out_width = self.DEFAULT_OUTPUT_HEIGHT
#                 out_height = max(1, int(out_width / aspect_ratio))
#                 # Rotate points 90° clockwise for correct orientation
#                 ordered_pts = np.roll(ordered_pts, 1, axis=0)
#             else:
#                 out_width = max(1, int(out_height * aspect_ratio))
            
#             # Destination points for perspective transform
#             dst_pts = np.array([
#                 [0.0, 0.0],
#                 [float(out_width), 0.0],
#                 [float(out_width), float(out_height)],
#                 [0.0, float(out_height)]
#             ], dtype=np.float32)
            
#             src_pts = ordered_pts.astype(np.float32)
            
#             # Compute and apply perspective transform
#             M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#             warped = cv2.warpPerspective(
#                 image, 
#                 M, 
#                 (out_width, out_height),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_REPLICATE
#             )
            
#             return warped
            
#         except Exception as e:
#             logger.exception(f"Perspective warp failed: {e}")
#             return None
    
#     def _remove_horizontal_lines_post_warp(self, image: np.ndarray) -> np.ndarray:
#         """
#         Remove horizontal lines AFTER perspective warp (preserves box borders during detection).
        
#         Critical improvement: Line removal happens POST-warp to avoid erasing box borders
#         during contour detection on lined paper.
        
#         Args:
#             image: Warped answer box image (BGR format)
            
#         Returns:
#             Image with horizontal lines removed/inpainted
#         """
#         if image is None or image.size == 0:
#             return image
        
#         try:
#             # Convert to grayscale for line detection
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
#             # Create horizontal structuring element (wide for ruled lines)
#             horizontal_kernel = cv2.getStructuringElement(
#                 cv2.MORPH_RECT, 
#                 (max(25, image.shape[1] // 15), 1)  # Adaptive width based on image width
#             )
            
#             # Extract horizontal lines
#             horizontal_lines = cv2.morphologyEx(
#                 gray, 
#                 cv2.MORPH_OPEN, 
#                 horizontal_kernel, 
#                 iterations=2
#             )
            
#             # Create mask and dilate to cover entire lines
#             mask = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
            
#             # Inpaint to remove lines (preserves text/ink)
#             result = cv2.inpaint(
#                 image, 
#                 mask, 
#                 inpaintRadius=3, 
#                 flags=cv2.INPAINT_TELEA
#             )
            
#             return result
            
#         except Exception as e:
#             logger.warning(f"Line removal failed (returning original): {e}")
#             return image
    
#     def _deskew_image(self, image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
#         """
#         Correct small rotational skew using MEDIAN angle (more robust than mean).
        
#         Adopted from crude preprocessor: median is less sensitive to outliers than mean.
        
#         Args:
#             image: Input image (BGR format)
#             max_angle: Maximum skew angle to correct (degrees)
            
#         Returns:
#             Deskewed image (BGR format)
#         """
#         if image is None or image.size == 0:
#             return image
        
#         try:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#             # Use adaptive thresholding for better line detection on text-heavy images
#             thresh = cv2.adaptiveThreshold(
#                 gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                 cv2.THRESH_BINARY_INV, 11, 2
#             )
            
#             # Detect lines with Hough transform
#             lines = cv2.HoughLines(
#                 thresh, 
#                 rho=1, 
#                 theta=np.pi/180, 
#                 threshold=max(50, min(150, thresh.shape[1] // 15))
#             )
            
#             if lines is None:
#                 return image
            
#             # Collect angles near horizontal (within max_angle)
#             angles = []
#             for line in lines[:75]:  # Limit to strongest lines
#                 rho, theta = line[0]
#                 # Convert theta to degrees from horizontal
#                 angle = (theta * 180 / np.pi) - 90
#                 if abs(angle) <= max_angle:
#                     angles.append(angle)
            
#             if not angles:
#                 return image
            
#             # MEDIAN IMPROVEMENT: More robust to outliers than mean
#             skew_angle = float(np.median(angles))
#             if abs(skew_angle) < 0.5:  # Negligible skew
#                 return image
            
#             # Rotate image to correct skew
#             (h, w) = image.shape[:2]
#             center = (w // 2, h // 2)
#             M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
#             # Calculate new bounding dimensions to avoid clipping
#             cos = np.abs(M[0, 0])
#             sin = np.abs(M[0, 1])
#             new_w = int((h * sin) + (w * cos))
#             new_h = int((h * cos) + (w * sin))
            
#             # Adjust rotation matrix for new center
#             M[0, 2] += (new_w / 2) - center[0]
#             M[1, 2] += (new_h / 2) - center[1]
            
#             deskewed = cv2.warpAffine(
#                 image, 
#                 M, 
#                 (new_w, new_h),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_REPLICATE
#             )
            
#             # Resize back to standard height while preserving aspect
#             aspect = new_w / new_h
#             out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
#             return cv2.resize(deskewed, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
            
#         except Exception as e:
#             logger.warning(f"Deskew failed (returning original): {e}")
#             return image
    
#     def _debug_detection(
#         self, 
#         image: np.ndarray, 
#         contours: List[np.ndarray],
#         candidate_boxes: List[Tuple[float, np.ndarray, np.ndarray]]
#     ):
#         """Generate debug visualizations for detection pipeline (conditional on debug_mode)"""
#         if not self.debug_mode:
#             return
        
#         debug_dir = Path("debug_detection")
#         debug_dir.mkdir(exist_ok=True)
#         debug_id = uuid.uuid4().hex[:6]
        
#         # Draw all contours
#         debug_img = image.copy()
#         cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        
#         # Draw candidate boxes with confidence scores
#         for i, (confidence, approx, _) in enumerate(candidate_boxes):
#             # Normalize points to (4, 2) shape
#             pts = approx.reshape(4, 2).astype(np.int32) if approx.shape[0] == 4 else approx
            
#             # Draw box polygon
#             cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
            
#             # Label with confidence
#             label_pos = (pts[0][0] + 10, pts[0][1] + 25)
#             cv2.putText(
#                 debug_img, 
#                 f"Conf: {confidence:.2f}", 
#                 label_pos,
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 0.6, 
#                 (0, 0, 255), 
#                 2
#             )
        
#         # Save visualization
#         out_path = debug_dir / f"detection_{debug_id}.jpg"
#         cv2.imwrite(str(out_path), debug_img)
#         logger.info(f"Saved detection debug visualization: {out_path}")
    
#     # ==============================
#     # Manual Segmentation Support (API endpoint handler)
#     # ==============================
    
#     def manual_segment_answer_box(
#         self, 
#         image_bytes: bytes, 
#         points: dict
#     ) -> bytes:
#         """
#         Manually segment answer box using provided corner points.
        
#         Used by PATCH /api/test_instances/{test_id}/{student_no}/{item_id}
        
#         Args:
#             image_bytes: Raw image bytes
#             points: Dictionary with ul, ur, lr, ll coordinates
            
#         Returns:
#             JPEG-encoded bytes of segmented and enhanced answer box
            
#         Raises:
#             CVProcessingError: If segmentation fails
#         """
#         try:
#             # Validate points structure
#             required_corners = ["ul", "ur", "lr", "ll"]
#             for corner in required_corners:
#                 if corner not in points:
#                     raise CVProcessingError(f"Missing required corner point: '{corner}'")
#                 if not isinstance(points[corner], dict):
#                     raise CVProcessingError(f"Corner '{corner}' must be a dictionary")
#                 if "x" not in points[corner] or "y" not in points[corner]:
#                     raise CVProcessingError(
#                         f"Corner '{corner}' missing required 'x' or 'y' coordinate"
#                     )
            
#             # Load and validate image
#             image = self._load_image_from_bytes(image_bytes)
#             h, w = image.shape[:2]
            
#             # Extract and validate source points
#             src_pts = np.array([
#                 [points["ul"]["x"], points["ul"]["y"]],
#                 [points["ur"]["x"], points["ur"]["y"]],
#                 [points["lr"]["x"], points["lr"]["y"]],
#                 [points["ll"]["x"], points["ll"]["y"]]
#             ], dtype=np.float32)
            
#             # Validate point coordinates are within image bounds
#             if not np.all((src_pts >= 0) & (src_pts[:, 0] < w) & (src_pts[:, 1] < h)):
#                 raise CVProcessingError(
#                     "One or more corner points are outside image boundaries"
#                 )
            
#             # Order points to ensure correct perspective transform
#             ordered_pts = self._order_points(src_pts)
            
#             # Calculate output dimensions with portrait/landscape detection
#             width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
#             width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
#             avg_width = (width_top + width_bottom) / 2.0
            
#             height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
#             height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
#             avg_height = (height_left + height_right) / 2.0
            
#             aspect_ratio = avg_width / avg_height
#             is_portrait = aspect_ratio < 0.8
            
#             if is_portrait:
#                 out_width = self.DEFAULT_OUTPUT_HEIGHT
#                 out_height = max(1, int(out_width / aspect_ratio))
#                 # Rotate points for correct orientation
#                 ordered_pts = np.roll(ordered_pts, 1, axis=0)
#             else:
#                 out_height = self.DEFAULT_OUTPUT_HEIGHT
#                 out_width = max(1, int(out_height * aspect_ratio))
            
#             # Destination points
#             dst_pts = np.array([
#                 [0.0, 0.0],
#                 [float(out_width), 0.0],
#                 [float(out_width), float(out_height)],
#                 [0.0, float(out_height)]
#             ], dtype=np.float32)
            
#             # Perspective transform
#             M = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
#             warped = cv2.warpPerspective(
#                 image, 
#                 M, 
#                 (out_width, out_height),
#                 flags=cv2.INTER_CUBIC,
#                 borderMode=cv2.BORDER_REPLICATE
#             )
            
#             # Enhance, remove lines, and deskew
#             enhanced = self.brighten(warped, amount=0.25)
#             enhanced = self.adjust_contrast(enhanced, amount=1.3)
#             enhanced = self._remove_horizontal_lines_post_warp(enhanced)
#             enhanced = self._deskew_image(enhanced, max_angle=15.0)
            
#             # Encode and return
#             return self._encode_to_bytes(enhanced)
            
#         except CVProcessingError:
#             raise
#         except Exception as e:
#             logger.error(f"Manual segmentation failed: {e}", exc_info=True)
#             raise CVProcessingError(f"Manual segmentation failed: {str(e)}")

# IMAGE_PREPROCESSOR_3

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import uuid
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional PaddleOCR import (fails gracefully if not installed)
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("✓ PaddleOCR available for document detection")
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("⚠ PaddleOCR not installed. Falling back to traditional CV methods.")
    logger.info("To enable PaddleOCR (CPU-only, Windows compatible):")
    logger.info("  pip install paddlepaddle paddleocr")


class CVProcessingError(Exception):
    """Custom exception for CV processing failures"""
    pass


class CVImagePreprocessor:
    """
    Production-ready image preprocessor for student assessment answer box detection.
    
    HYBRID ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │  Stage 1: Document Detection                                │
    │    • PaddleOCR (if enabled): Robust document boundary detection │
    │    • Fallback: Traditional CV contour detection              │
    ├─────────────────────────────────────────────────────────────┤
    │  Stage 2: Answer Box Extraction (YOUR UNIQUE VALUE)         │
    │    • Traditional CV on warped document                       │
    │    • Convex hull + iterative epsilon approximation           │
    │    • Morphological closing for fragmented borders            │
    │    • Bounding rect fallback for failed approximations        │
    ├─────────────────────────────────────────────────────────────┤
    │  Stage 3: Enhancement                                        │
    │    • Line removal AFTER warp (critical fix for lined paper)  │
    │    • Median-based deskew (outlier-resistant)                 │
    │    • Brightness/contrast optimization                        │
    └─────────────────────────────────────────────────────────────┘
    
    KEY ADVANTAGES:
    ✅ Works on Windows (CPU-only PaddlePaddle, no GPU required)
    ✅ ~250MB disk footprint (not 1.5GB - that's GPU version)
    ✅ Graceful fallback if PaddleOCR fails or isn't installed
    ✅ Fixes critical lined paper flaw: line removal AFTER warp
    ✅ Preserves your innovation: answer box detection within documents
    
    API CONTRACT (required by api.py):
    - Input: raw image bytes (from await file.read())
    - Output: List[bytes] where each element is a JPEG byte array
    - Returns up to 3 candidate regions for manual selection
    - Returns at least 1 box (full image if no boxes detected)
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}
    DEFAULT_OUTPUT_HEIGHT = 800  # Standard height for processed answer boxes
    
    def __init__(
        self,
        min_area_ratio: float = 0.003,
        min_confidence: float = 0.10,
        max_aspect_ratio: float = 20.0,
        debug_mode: bool = False,
        max_processing_dim: int = 1500,
        use_paddle_ocr: bool = False,
        paddle_ocr_lang: str = 'en'
    ):
        os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'  # Speed up init
        
        self.min_area_ratio = min_area_ratio
        self.min_confidence = min_confidence
        self.max_aspect_ratio = max_aspect_ratio
        self.debug_mode = debug_mode
        self.max_processing_dim = max_processing_dim
        self.use_paddle_ocr = use_paddle_ocr and PADDLEOCR_AVAILABLE
        self.paddle_ocr_lang = paddle_ocr_lang
        
        # ✅ CORRECT for 2.7+: ONLY 'lang' parameter accepted
        self.paddle_ocr = None
        if self.use_paddle_ocr:
            try:
                self.paddle_ocr = PaddleOCR(lang=self.paddle_ocr_lang)  # NO other params!
                logger.info(f"✓ PaddleOCR initialized (v{self._get_paddle_version()}) | Lang: {paddle_ocr_lang}")
            except Exception as e:
                logger.warning(f"⚠ PaddleOCR initialization failed: {e}")
                self.use_paddle_ocr = False
                self.paddle_ocr = None
        
        self._validate_config()
        status = 'ENABLED' if self.use_paddle_ocr else 'DISABLED'
        logger.info(f"Preprocessor initialized | PaddleOCR: {status}")

    def _get_paddle_version(self) -> str:
        """Get PaddleOCR version string for logging"""
        try:
            import paddleocr
            return paddleocr.__version__
        except:
            return "unknown"
    
    def _validate_config(self):
        """Validate constructor parameters"""
        if not 0.0 < self.min_area_ratio < 1.0:
            raise ValueError("min_area_ratio must be between 0 and 1")
        if not 0.0 < self.min_confidence < 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio must be >= 1.0")
        if self.max_processing_dim < 200:
            raise ValueError("max_processing_dim must be >= 200")
    
    # ==============================
    # Core API Methods (Contract with api.py)
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
        if not filename or not isinstance(filename, str):
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
            CVProcessingError: If image processing fails catastrophically
        """
        try:
            # Load image from bytes
            image = self._load_image_from_bytes(image_bytes)
            logger.info(f"✓ Loaded image: {image.shape[1]}x{image.shape[0]}px")
            
            # Detect answer box candidates (hybrid approach)
            candidate_boxes = self._detect_answer_box_candidates(image)
            logger.info(f"✓ Detected {len(candidate_boxes)} candidate answer box(es)")
            
            # Process each box (enhancements applied AFTER detection)
            processed_bytes_list = []
            for i, box_image in enumerate(candidate_boxes):
                try:
                    # Apply enhancements in optimal order
                    enhanced = self.brighten(box_image, amount=0.25)
                    enhanced = self.adjust_contrast(enhanced, amount=1.3)
                    
                    # CRITICAL FIX: Remove horizontal lines AFTER warp
                    # (Preserves box borders during detection on lined paper)
                    enhanced = self._remove_horizontal_lines_post_warp(enhanced)
                    
                    # Deskew with median angle (more robust than mean)
                    enhanced = self._deskew_image(enhanced, max_angle=15.0)
                    
                    # Encode to JPEG bytes
                    img_bytes = self._encode_to_bytes(enhanced)
                    processed_bytes_list.append(img_bytes)
                    logger.debug(f"  → Processed box #{i+1}: {len(img_bytes):,} bytes")
                except Exception as e:
                    logger.warning(f"⚠ Failed to process box #{i+1}: {e}")
                    continue
            
            # Fallback: if no boxes detected, return full image
            if not processed_bytes_list:
                logger.warning("⚠ No valid answer boxes detected. Returning full image as fallback.")
                full_processed = self.brighten(image, amount=0.25)
                full_processed = self.adjust_contrast(full_processed, amount=1.3)
                full_processed = self._remove_horizontal_lines_post_warp(full_processed)
                full_processed = self._deskew_image(full_processed, max_angle=15.0)
                processed_bytes_list.append(self._encode_to_bytes(full_processed))
            
            # Limit to 3 boxes max (sorted by confidence during detection)
            final_boxes = processed_bytes_list[:3]
            logger.info(f"✓ Returning {len(final_boxes)} box(es) for manual selection")
            return final_boxes
            
        except CVProcessingError:
            raise
        except Exception as e:
            logger.error(f"✗ Unexpected error in process_assessment_image: {e}", exc_info=True)
            raise CVProcessingError(f"Image processing failed: {str(e)}")
    
    def brighten(self, image: np.ndarray, amount: float = 0.25) -> np.ndarray:
        """Increase image brightness using linear transform."""
        if image is None or image.size == 0:
            raise CVProcessingError("Cannot brighten empty image")
        amount = max(0.0, min(1.0, float(amount)))
        beta = amount * 255
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    def adjust_contrast(self, image: np.ndarray, amount: float = 1.3) -> np.ndarray:
        """Adjust image contrast with brightness compensation."""
        if image is None or image.size == 0:
            raise CVProcessingError("Cannot adjust contrast on empty image")
        amount = max(0.1, float(amount))
        beta = 128 * (1 - amount)
        return cv2.convertScaleAbs(image, alpha=amount, beta=beta)
    
    # ==============================
    # Internal Processing Methods
    # ==============================
    
    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image with validation."""
        if not image_bytes or len(image_bytes) == 0:
            raise CVProcessingError("Empty image bytes provided")
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise CVProcessingError("Failed to decode image bytes - invalid format or corrupted data")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise CVProcessingError(f"Invalid image format: expected 3-channel BGR, got shape {image.shape}")
        return image
    
    def _encode_to_bytes(self, image: np.ndarray) -> bytes:
        """Encode OpenCV image to high-quality JPEG bytes."""
        if image is None or image.size == 0:
            raise CVProcessingError("Cannot encode empty image")
        success, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            raise CVProcessingError("Failed to encode image to JPEG")
        return buffer.tobytes()
    
    def _detect_answer_box_candidates(self, image: np.ndarray) -> List[np.ndarray]:
        """
        HYBRID DETECTION PIPELINE:
        1. Use PaddleOCR to detect and warp the entire document (if enabled)
        2. Extract answer boxes from the warped document using robust traditional CV
        3. Fall back to direct traditional CV if PaddleOCR fails/disabled
        
        This architecture leverages PaddleOCR's strength (document detection) while
        preserving your unique innovation (answer box extraction within documents).
        """
        orig = image.copy()
        h, w = image.shape[:2]
        
        # ────────────────────────────────────────────────────────
        # STAGE 1: Document Detection & Warping (PaddleOCR or Fallback)
        # ────────────────────────────────────────────────────────
        document_image = None
        
        if self.use_paddle_ocr and self.paddle_ocr is not None:
            try:
                logger.debug("→ Running PaddleOCR document detection...")
                
                # ✅ CORRECT for 2.7+: NO det/rec/cls params - they don't exist in this API
                result = self.paddle_ocr.predict(image)
                
                # Parse result format: [[ [box_points], (text, confidence) ], ...]
                # We only need the box_points (first element of each detection)
                if result and result[0]:  # result[0] = detections for first page
                    doc_boxes = []
                    for detection in result[0]:
                        # detection[0] = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        # detection[1] = (recognized_text, confidence) - we ignore this
                        box_points = detection[0]
                        doc_boxes.append(box_points)
                    
                    if doc_boxes:
                        # Take highest confidence box (first in list is usually best)
                        best_box = doc_boxes[0]
                        corners = np.array(best_box, dtype=np.float32)
                        
                        # Warp document using PaddleOCR corners
                        document_image = self._warp_from_ocr_corners(image, corners)
                        
                        if document_image is not None and document_image.size > 0:
                            logger.info("✓ PaddleOCR successfully detected and warped document")
                            return self._extract_answer_boxes_from_document(document_image)
                        else:
                            logger.warning("⚠ PaddleOCR warp failed - falling back to traditional CV")
                else:
                    logger.info("ℹ PaddleOCR found no document boundaries - falling back to traditional CV")
                    
            except Exception as e:
                logger.warning(f"⚠ PaddleOCR detection failed: {e}. Falling back to traditional CV.")
        
        # Fallback to traditional CV if PaddleOCR failed or disabled
        if document_image is None:
            logger.debug("→ Using traditional CV for direct answer box detection")
            return self._traditional_answer_box_detection(image)
        
        # ────────────────────────────────────────────────────────
        # STAGE 2: Answer Box Extraction from Warped Document
        # ────────────────────────────────────────────────────────
        logger.debug("→ Extracting answer boxes from warped document...")
        return self._extract_answer_boxes_from_document(document_image)
    
    def _warp_from_ocr_corners(self, image: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        """Warp image using corners detected by PaddleOCR."""
        try:
            if corners.shape != (4, 2):
                logger.error(f"Invalid corners shape from PaddleOCR: {corners.shape} (expected 4x2)")
                return None
            
            # Order points for perspective transform
            ordered_pts = self._order_points(corners)
            
            # Calculate output dimensions
            width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
            width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
            avg_width = (width_top + width_bottom) / 2.0
            
            height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
            height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
            avg_height = (height_left + height_right) / 2.0
            
            if avg_height < 1.0 or avg_width < 1.0:
                logger.warning("Degenerate dimensions from PaddleOCR corners")
                return None
            
            # Handle portrait vs landscape orientation
            aspect_ratio = avg_width / avg_height
            is_portrait = aspect_ratio < 0.8
            
            if is_portrait:
                out_width = self.DEFAULT_OUTPUT_HEIGHT
                out_height = max(1, int(out_width / aspect_ratio))
                ordered_pts = np.roll(ordered_pts, 1, axis=0)  # Rotate points for correct orientation
            else:
                out_height = self.DEFAULT_OUTPUT_HEIGHT
                out_width = max(1, int(out_height * aspect_ratio))
            
            # Destination points for perspective transform
            dst_pts = np.array([
                [0.0, 0.0],
                [float(out_width), 0.0],
                [float(out_width), float(out_height)],
                [0.0, float(out_height)]
            ], dtype=np.float32)
            
            src_pts = ordered_pts.astype(np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            warped = cv2.warpPerspective(
                image,
                M,
                (out_width, out_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return warped
            
        except Exception as e:
            logger.exception(f"Failed to warp from PaddleOCR corners: {e}")
            return None
    
    def _extract_answer_boxes_from_document(self, warped_doc: np.ndarray) -> List[np.ndarray]:
        """
        Extract answer boxes from a pre-warped document using robust traditional CV.
        This is YOUR unique contribution - finding answer boxes within documents.
        """
        try:
            logger.debug("  → Running traditional CV on warped document for answer box extraction")
            
            # Resize for performance while preserving aspect ratio
            h, w = warped_doc.shape[:2]
            scale = min(self.max_processing_dim / max(h, w), 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(warped_doc, (new_w, new_h))
            
            # Preprocessing pipeline (optimized from crude preprocessor)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)
            
            # Edge detection with adaptive thresholds
            blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 100)
            
            # CRITICAL IMPROVEMENT from crude preprocessor:
            # Morphological closing to connect fragmented edges BEFORE contour detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
            edged = cv2.dilate(edged, kernel, iterations=1)  # Strengthen edges
            
            # Find contours
            contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            candidate_boxes = []
            min_area = self.min_area_ratio * new_w * new_h
            max_boxes = 5
            
            for contour in contours:
                if len(candidate_boxes) >= max_boxes:
                    break
                
                # Skip small contours early (performance optimization)
                if cv2.contourArea(contour) < min_area:
                    continue
                
                # CONVEX HULL IMPROVEMENT (from crude preprocessor):
                # Eliminate concavities before approximation
                hull = cv2.convexHull(contour)
                hull_perimeter = cv2.arcLength(hull, True)
                
                # ITERATIVE EPSILON ADJUSTMENT (from crude preprocessor):
                # Force 4-point approximation even for complex contours
                epsilon = 0.02 * hull_perimeter
                max_epsilon = 0.25 * hull_perimeter
                approx = cv2.approxPolyDP(hull, epsilon, True)
                num_sides = len(approx)
                
                while num_sides > 4 and epsilon < max_epsilon:
                    epsilon *= 1.5
                    approx = cv2.approxPolyDP(hull, epsilon, True)
                    num_sides = len(approx)
                
                # FINAL FALLBACK (from crude preprocessor):
                # Use bounding rectangle if approximation fails
                if num_sides != 4:
                    x, y, bw, bh = cv2.boundingRect(hull)
                    approx = np.array([
                        [[x, y]],
                        [[x + bw, y]],
                        [[x + bw, y + bh]],
                        [[x, y + bh]]
                    ], dtype=np.int32)
                
                # Filter by aspect ratio
                rect = cv2.minAreaRect(approx)
                box_w, box_h = rect[1]
                if box_w < 1 or box_h < 1:
                    continue
                
                aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
                if aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # Calculate confidence score
                area = cv2.contourArea(approx)
                mask = np.zeros_like(gray_enhanced)
                cv2.drawContours(mask, [approx], -1, (255,), -1)
                edge_density = np.sum(edged[mask == 255]) / (area + 1e-5)
                
                area_score = min(area / (0.3 * new_w * new_h), 1.0)
                aspect_score = 1.0 - min(abs(aspect_ratio - 2.5) / 6.0, 1.0)
                edge_score = min(edge_density / 120.0, 1.0)
                
                confidence = (
                    area_score * 0.4 +
                    aspect_score * 0.35 +
                    edge_score * 0.25
                )
                
                if confidence < self.min_confidence:
                    continue
                
                # Warp the answer box
                try:
                    warped_box = self._warp_perspective_box(resized, approx, new_h)
                    if warped_box is not None and warped_box.size > 0:
                        candidate_boxes.append((confidence, approx, warped_box))
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to warp answer box: {e}")
                    continue
            
            # Sort by confidence (highest first)
            candidate_boxes.sort(key=lambda x: x[0], reverse=True)
            
            if self.debug_mode:
                self._debug_detection(resized, contours, candidate_boxes)
            
            result_boxes = [warped for _, _, warped in candidate_boxes]
            
            # Fallback: return full warped document if no boxes found
            if not result_boxes:
                logger.info("  ℹ No answer boxes detected in warped document - using full document")
                aspect = w / h
                out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
                full_resized = cv2.resize(warped_doc, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
                result_boxes.append(full_resized)
            
            return result_boxes
            
        except Exception as e:
            logger.exception(f"Failed to extract answer boxes from document: {e}")
            # Return full document as fallback
            aspect = warped_doc.shape[1] / warped_doc.shape[0]
            out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
            full_resized = cv2.resize(warped_doc, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
            return [full_resized]
    
    def _traditional_answer_box_detection(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Traditional CV approach for answer box detection (used when PaddleOCR is disabled or fails).
        Incorporates all robustness improvements from crude preprocessor.
        """
        orig = image.copy()
        h, w = image.shape[:2]
        
        # Resize for performance
        scale = min(self.max_processing_dim / max(h, w), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        orig_resized = resized.copy()
        
        # Preprocessing pipeline
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # Edge detection
        blurred = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
        edged = cv2.Canny(blurred, 30, 100)
        
        # MORPHOLOGICAL CLOSING (critical improvement from crude preprocessor)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        candidate_boxes = []
        min_area = self.min_area_ratio * new_w * new_h
        max_boxes = 5
        
        for contour in contours:
            if len(candidate_boxes) >= max_boxes:
                break
            
            if cv2.contourArea(contour) < min_area:
                continue
            
            # CONVEX HULL + ITERATIVE EPSILON (from crude preprocessor)
            hull = cv2.convexHull(contour)
            hull_perimeter = cv2.arcLength(hull, True)
            epsilon = 0.02 * hull_perimeter
            max_epsilon = 0.25 * hull_perimeter
            approx = cv2.approxPolyDP(hull, epsilon, True)
            num_sides = len(approx)
            
            while num_sides > 4 and epsilon < max_epsilon:
                epsilon *= 1.5
                approx = cv2.approxPolyDP(hull, epsilon, True)
                num_sides = len(approx)
            
            # BOUNDING RECT FALLBACK (from crude preprocessor)
            if num_sides != 4:
                x, y, bw, bh = cv2.boundingRect(hull)
                approx = np.array([
                    [[x, y]],
                    [[x + bw, y]],
                    [[x + bw, y + bh]],
                    [[x, y + bh]]
                ], dtype=np.int32)
            
            # Filter by aspect ratio
            rect = cv2.minAreaRect(approx)
            box_w, box_h = rect[1]
            if box_w < 1 or box_h < 1:
                continue
            
            aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
            if aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate confidence
            area = cv2.contourArea(approx)
            mask = np.zeros_like(gray_enhanced)
            cv2.drawContours(mask, [approx], -1, (255,), -1)
            edge_density = np.sum(edged[mask == 255]) / (area + 1e-5)
            
            area_score = min(area / (0.3 * new_w * new_h), 1.0)
            aspect_score = 1.0 - min(abs(aspect_ratio - 2.5) / 6.0, 1.0)
            edge_score = min(edge_density / 120.0, 1.0)
            
            confidence = (
                area_score * 0.4 +
                aspect_score * 0.35 +
                edge_score * 0.25
            )
            
            if confidence < self.min_confidence:
                continue
            
            # Warp perspective
            try:
                warped = self._warp_perspective_box(orig_resized, approx, new_h)
                if warped is not None and warped.size > 0:
                    candidate_boxes.append((confidence, approx, warped))
            except Exception as e:
                logger.warning(f"Failed to warp box: {e}")
                continue
        
        # Sort by confidence
        candidate_boxes.sort(key=lambda x: x[0], reverse=True)
        
        if self.debug_mode:
            self._debug_detection(orig_resized, contours, candidate_boxes)
        
        result_boxes = [warped for _, _, warped in candidate_boxes]
        
        # Fallback to full image
        if not result_boxes:
            logger.info("ℹ No answer boxes detected - using full image as fallback")
            aspect = w / h
            out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
            full_resized = cv2.resize(orig, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
            result_boxes.append(full_resized)
        
        return result_boxes
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order 4 points for perspective transform at ANY rotation angle.
        Uses centroid-angle sorting to handle 0°-360° rotations robustly.
        """
        if pts.shape[0] != 4:
            raise CVProcessingError(f"_order_points requires exactly 4 points, got {pts.shape[0]}")
        
        # Normalize shape to (4, 2)
        if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
            pts = pts.reshape(4, 2)
        elif pts.ndim == 1 and pts.size == 8:
            pts = pts.reshape(4, 2)
        
        if pts.shape != (4, 2):
            raise CVProcessingError(f"Unexpected points shape {pts.shape} - expected (4, 2)")
        
        # Ensure float32 for OpenCV compatibility
        if pts.dtype not in (np.float32, np.float64):
            pts = pts.astype(np.float32)
        
        # Centroid-angle sorting (rotation-agnostic)
        centroid = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        sorted_pts = pts[np.argsort(angles)]
        
        # Rotate so top-left (min x+y) is first
        tl_idx = np.argmin(sorted_pts[:, 0] + sorted_pts[:, 1])
        ordered = np.roll(sorted_pts, -tl_idx, axis=0)
        
        return ordered.astype(np.float32)
    
    def _warp_perspective_box(
        self, 
        image: np.ndarray, 
        pts: np.ndarray, 
        out_height: int
    ) -> Optional[np.ndarray]:
        """Apply perspective transform with rotation-agnostic point ordering."""
        try:
            # Normalize points to (4, 2) shape
            if pts.ndim == 3 and pts.shape[1] == 1 and pts.shape[2] == 2:
                pts = pts.reshape(-1, 2)
            elif pts.ndim == 1 and pts.size == 8:
                pts = pts.reshape(4, 2)
            
            if pts.shape[0] != 4 or pts.shape[1] != 2:
                logger.error(f"Invalid points shape for warp: {pts.shape}")
                return None
            
            # Order points robustly
            ordered_pts = self._order_points(pts)
            
            # Calculate dimensions
            width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
            width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
            avg_width = (width_top + width_bottom) / 2.0
            
            height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
            height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
            avg_height = (height_left + height_right) / 2.0
            
            if avg_height < 1.0 or avg_width < 1.0:
                logger.warning("Degenerate box dimensions - skipping warp")
                return None
            
            # Handle portrait orientation
            aspect_ratio = avg_width / avg_height
            is_portrait = aspect_ratio < 0.8
            
            if is_portrait:
                out_width = self.DEFAULT_OUTPUT_HEIGHT
                out_height = max(1, int(out_width / aspect_ratio))
                ordered_pts = np.roll(ordered_pts, 1, axis=0)
            else:
                out_width = max(1, int(out_height * aspect_ratio))
            
            # Destination points
            dst_pts = np.array([
                [0.0, 0.0],
                [float(out_width), 0.0],
                [float(out_width), float(out_height)],
                [0.0, float(out_height)]
            ], dtype=np.float32)
            
            src_pts = ordered_pts.astype(np.float32)
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            warped = cv2.warpPerspective(
                image,
                M,
                (out_width, out_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return warped
            
        except Exception as e:
            logger.exception(f"Perspective warp failed: {e}")
            return None
    
    def _remove_horizontal_lines_post_warp(self, image: np.ndarray) -> np.ndarray:
        """
        CRITICAL FIX: Remove horizontal lines AFTER perspective warp.
        This preserves box borders during detection on lined paper.
        """
        if image is None or image.size == 0:
            return image
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive kernel size based on image width
            kernel_width = max(25, image.shape[1] // 15)
            horizontal_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_width, 1)
            )
            
            # Extract horizontal lines
            horizontal_lines = cv2.morphologyEx(
                gray,
                cv2.MORPH_OPEN,
                horizontal_kernel,
                iterations=2
            )
            
            # Create mask and dilate to cover entire lines
            mask = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
            
            # Inpaint to remove lines (preserves text/ink)
            result = cv2.inpaint(
                image,
                mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
            
            return result
            
        except Exception as e:
            logger.warning(f"Line removal failed (returning original): {e}")
            return image
    
    def _deskew_image(self, image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
        """
        Correct small rotational skew using MEDIAN angle (more robust than mean).
        Adopted from crude preprocessor: median is less sensitive to outliers.
        """
        if image is None or image.size == 0:
            return image
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Detect lines with Hough transform
            lines = cv2.HoughLines(
                thresh,
                rho=1,
                theta=np.pi/180,
                threshold=max(50, min(150, thresh.shape[1] // 15))
            )
            
            if lines is None:
                return image
            
            # Collect angles near horizontal
            angles = []
            for line in lines[:75]:  # Limit to strongest lines
                rho, theta = line[0]
                angle = (theta * 180 / np.pi) - 90
                if abs(angle) <= max_angle:
                    angles.append(angle)
            
            if not angles:
                return image
            
            # MEDIAN IMPROVEMENT: More robust to outliers than mean
            skew_angle = float(np.median(angles))  # ← Pylance-safe conversion
            if abs(skew_angle) < 0.5:
                return image
            
            # Rotate image to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Calculate new bounding dimensions to avoid clipping
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust rotation matrix for new center
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            deskewed = cv2.warpAffine(
                image,
                M,
                (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Resize back to standard height
            aspect = new_w / new_h
            out_w = max(1, int(self.DEFAULT_OUTPUT_HEIGHT * aspect))
            return cv2.resize(deskewed, (out_w, self.DEFAULT_OUTPUT_HEIGHT))
            
        except Exception as e:
            logger.warning(f"Deskew failed (returning original): {e}")
            return image
    
    def _debug_detection(
        self, 
        image: np.ndarray, 
        contours: List[np.ndarray],
        candidate_boxes: List[Tuple[float, np.ndarray, np.ndarray]]
    ):
        """Generate debug visualizations (only if debug_mode=True)"""
        if not self.debug_mode:
            return
        
        debug_dir = Path("debug_detection")
        debug_dir.mkdir(exist_ok=True)
        debug_id = uuid.uuid4().hex[:6]
        
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        
        for i, (confidence, approx, _) in enumerate(candidate_boxes):
            pts = approx.reshape(4, 2).astype(np.int32) if approx.shape[0] == 4 else approx
            cv2.polylines(debug_img, [pts], True, (0, 0, 255), 2)
            label_pos = (pts[0][0] + 10, pts[0][1] + 25)
            cv2.putText(
                debug_img, 
                f"Conf: {confidence:.2f}", 
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
        
        out_path = debug_dir / f"detection_{debug_id}.jpg"
        cv2.imwrite(str(out_path), debug_img)
        logger.info(f"  → Saved debug visualization: {out_path}")
    
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
        """
        try:
            # Validate points structure
            required_corners = ["ul", "ur", "lr", "ll"]
            for corner in required_corners:
                if corner not in points:
                    raise CVProcessingError(f"Missing required corner point: '{corner}'")
                if not isinstance(points[corner], dict):
                    raise CVProcessingError(f"Corner '{corner}' must be a dictionary")
                if "x" not in points[corner] or "y" not in points[corner]:
                    raise CVProcessingError(
                        f"Corner '{corner}' missing required 'x' or 'y' coordinate"
                    )
            
            # Load image
            image = self._load_image_from_bytes(image_bytes)
            h, w = image.shape[:2]
            
            # Extract source points
            src_pts = np.array([
                [points["ul"]["x"], points["ul"]["y"]],
                [points["ur"]["x"], points["ur"]["y"]],
                [points["lr"]["x"], points["lr"]["y"]],
                [points["ll"]["x"], points["ll"]["y"]]
            ], dtype=np.float32)
            
            # Validate bounds
            if not np.all((src_pts >= 0) & (src_pts[:, 0] < w) & (src_pts[:, 1] < h)):
                raise CVProcessingError("One or more corner points are outside image boundaries")
            
            # Order points
            ordered_pts = self._order_points(src_pts)
            
            # Calculate output dimensions
            width_top = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
            width_bottom = np.linalg.norm(ordered_pts[3] - ordered_pts[2])
            avg_width = (width_top + width_bottom) / 2.0
            
            height_left = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
            height_right = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
            avg_height = (height_left + height_right) / 2.0
            
            aspect_ratio = avg_width / avg_height
            is_portrait = aspect_ratio < 0.8
            
            if is_portrait:
                out_width = self.DEFAULT_OUTPUT_HEIGHT
                out_height = max(1, int(out_width / aspect_ratio))
                ordered_pts = np.roll(ordered_pts, 1, axis=0)
            else:
                out_height = self.DEFAULT_OUTPUT_HEIGHT
                out_width = max(1, int(out_height * aspect_ratio))
            
            # Perspective transform
            dst_pts = np.array([
                [0.0, 0.0],
                [float(out_width), 0.0],
                [float(out_width), float(out_height)],
                [0.0, float(out_height)]
            ], dtype=np.float32)
            
            M = cv2.getPerspectiveTransform(ordered_pts.astype(np.float32), dst_pts)
            warped = cv2.warpPerspective(
                image,
                M,
                (out_width, out_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Enhance
            enhanced = self.brighten(warped, amount=0.25)
            enhanced = self.adjust_contrast(enhanced, amount=1.3)
            enhanced = self._remove_horizontal_lines_post_warp(enhanced)
            enhanced = self._deskew_image(enhanced, max_angle=15.0)
            
            return self._encode_to_bytes(enhanced)
            
        except CVProcessingError:
            raise
        except Exception as e:
            logger.error(f"Manual segmentation failed: {e}", exc_info=True)
            raise CVProcessingError(f"Manual segmentation failed: {str(e)}")