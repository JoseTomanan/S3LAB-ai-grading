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
    
    def process_assessment_image(self, image_bytes: bytes, num_boxes: Optional[int] = None) -> List[bytes]:
            """
            Process raw assessment image and extract answer box regions.
            API CONTRACT:
            - Input: raw image bytes (from await file.read())
            - Output: List[bytes] where each element is a JPEG byte array
            - Must detect answer boxes and return each as separate processed image
            - Returns up to 'num_boxes' candidate regions (default 3)
            
            Args:
                image_bytes: Raw bytes of uploaded image
                num_boxes: Optional maximum number of boxes to return (default: 3)
                
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
                        enhanced = self._remove_horizontal_lines_post_warp(enhanced)
                        # Deskew with median angle (more robust than mean)
                        enhanced = self._deskew_image(enhanced, max_angle=15.0)
                        # Encode to JPEG bytes
                        img_bytes = self._encode_to_bytes(enhanced)
                        processed_bytes_list.append(img_bytes)
                        logger.debug(f"  Processed box #{i+1}: {len(img_bytes):,} bytes")
                    except Exception as e:
                        logger.warning(f"Failed to process box #{i+1}: {e}")
                        continue
                
                # Fallback: if no boxes detected, return full image
                if not processed_bytes_list:
                    logger.warning("No valid answer boxes detected. Returning full image as fallback.")
                    full_processed = self.brighten(image, amount=0.25)
                    full_processed = self.adjust_contrast(full_processed, amount=1.3)
                    full_processed = self._remove_horizontal_lines_post_warp(full_processed)
                    full_processed = self._deskew_image(full_processed, max_angle=15.0)
                    processed_bytes_list.append(self._encode_to_bytes(full_processed))
                
                # FIX: Use num_boxes parameter to limit results (default to 3 if None)
                limit = num_boxes if num_boxes is not None else 3
                final_boxes = processed_bytes_list[:limit]
                
                logger.info(f"Returning {len(final_boxes)} box(es) for manual selection")
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
        
    def crop_top_left_corner(self, image: np.ndarray, width_ratio: float = 0.30, height_ratio: float = 0.25) -> np.ndarray:
        """
        Crop top-left corner where encircled item numbers typically appear.
        
        Args:
            image: Input OpenCV image
            width_ratio: Fraction of width to crop (default: 30%)
            height_ratio: Fraction of height to crop (default: 25%)
        
        Returns:
            Cropped image region
        """
        h, w = image.shape[:2]
        crop_h = int(h * height_ratio)
        crop_w = int(w * width_ratio)
        return image[0:crop_h, 0:crop_w]