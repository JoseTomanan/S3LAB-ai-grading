import numpy as np
import cv2
from cv2.typing import MatLike
from logic.document_scanner import DocumentScanner, NORMAL_SIZE, _mapp, _get_robust_aspect_ratio
from logic.ai_interface import AIAnswerEvaluator
from logic.blob_detector import BlobDetector



AREA = NORMAL_SIZE ** 2
MIN_AREA = AREA * 0.01
MAX_AREA = AREA * 0.90
MAX_SKEW_DEG = 15.0
MAX_ASPECT_RATIO = 9.0



#region Class
class BoxSegmenter(DocumentScanner):
    def get_boxes(self, image_bytes: bytes, num_boxes: int, debug: bool = False,) -> list[bytes]:
        """Get best boxes (non-overlapping) from a scanned image. Currently tuned for white paper only."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_forgivingly(image)
        # image_cannied = self._filter_only_handdrawn_lines(image_cannied)    # FIXME: experimental extra step
        image_dilated = self._dilate_edges(image_cannied)

        self.save_image(
                    self._encode_to_bytes(image_dilated),
                    "./TEMP/output/DEBUG_canny_regularize_dilate.jpg"
                    )

        contours, _ = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        images_good_contours = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                perimeter = cv2.arcLength(c, True)
                approximate = cv2.approxPolyDP(c, 0.06*perimeter, True)
                
                if debug:
                    debug_img = self._highlight_contours(image_cannied, approximate, c)
                    self.save_image(
                                self._encode_to_bytes(debug_img),
                                f"./TEMP/output/DEBUG_contour_box{i}.jpg"
                                )
                if 4 <= len(approximate) <= 10:
                    hull = cv2.convexHull(c)
                    rect = cv2.minAreaRect(hull)
                    approximate = cv2.boxPoints(rect).astype(int)  # always exactly 4 pts
                    (_, _, w, h) = cv2.boundingRect(approximate)
                    aspect_ratio = w / float(h)
                    if 1/MAX_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                        print(f"INFO:\tAccepted and stored contour {i}")
                        images_good_contours.append(approximate)
                    else:
                        print(f"INFO:\tBad ratio, AR={aspect_ratio}.")
                else:
                    print(f"INFO:\tFound non-box at approxPolyDP of contour {i}")
            else:
                print(f"INFO:\tDid not pass for area={area}")

        if images_good_contours == []:
            raise ValueError("Could not find any boxes.")
        
        print(f"INFO:\tResult # of boxes: {len(images_good_contours)} (take top {num_boxes})")
        images_good_contours = sorted(images_good_contours, key=lambda b : cv2.boundingRect(b)[1])
        images_warped = [self._warp_from_original(c, image_original) for c in images_good_contours]

        return [self._encode_to_bytes(i) for i in images_warped]

    def beautify_scan(self, image_bytes: bytes) -> bytes:
        array = self._load_array(image_bytes)
        img = self._adjust_contrast(
                            self._brighten(array, amount=0.25),
                            amount=1.3
                            )
        return self._unload_array(img)

    #region Auxiliary functions
    def _regularize_forgivingly(self, image_mat: MatLike) -> list[MatLike]:
        def _pre_canny(i: MatLike) -> MatLike:
            ruled_line_mask = cv2.inRange(i, 20, 80)
            h_kernel_length = int(NORMAL_SIZE*0.30)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
            ruled_line_mask = cv2.morphologyEx(ruled_line_mask, cv2.MORPH_CLOSE, horizontal_kernel)
            self.save_image(
                    self._encode_to_bytes(ruled_line_mask),
                    "./TEMP/output/DEBUG_horizontal_candidates.jpg"
                    )
            return cv2.subtract(i, ruled_line_mask)

        # return self._regularize_image(image_mat, (30, 150), _pre_canny)

        return self._regularize_image(image_mat, canny_thresholds=(30, 150))
    
    def _dilate_edges(self, image: MatLike, dilate_size: int = 3) -> MatLike:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        image_dilated = cv2.dilate(image, kernel, iterations=2)
        image_closed = cv2.morphologyEx(image_dilated, cv2.MORPH_CLOSE, kernel)
        return image_closed
   
    def _load_array(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to OpenCV image with validation"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    def _unload_array(self, image: np.ndarray) -> bytes:
        """Encode OpenCV image to high-quality JPEG bytes."""
        _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return buffer.tobytes()

    def _brighten(self, image: np.ndarray, amount: float = 0.25) -> np.ndarray:
        """Increase image brightness using linear transform"""
        amount = max(0.0, min(1.0, float(amount)))
        beta = amount * 255
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    def _adjust_contrast(self, image: np.ndarray, amount: float = 1.3) -> np.ndarray:
        """Adjust image contrast with brightness compensation"""
        amount = max(0.1, float(amount))
        beta = 128 * (1 - amount)
        return cv2.convertScaleAbs(image, alpha=amount, beta=beta)
    #endregion

    #region Archived unstable functions
    def _get_boxes_via_dots(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """
        Same as previous function, but using solid fill blobs instead of handdrawn boxes.
        # UNTESTED
        """
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_forgivingly(image)
        image_dilated = self._dilate_edges(image_cannied)
        if debug:
            self.save_image(
                    self._encode_to_bytes(image_dilated),
                    "./TEMP/output/DEBUG_canny_regularize_dilate.jpg"
                    )

        BLOB_DETECTOR = BlobDetector()
        scale = (image_original.shape[1] / image_dilated.shape[1],
                    image_original.shape[0] / image_dilated.shape[0])
        image_sections = BLOB_DETECTOR.detect_sections_via_anchors(image_dilated, scale)

        images_warped = [self._warp_from_original(i, image_original) for i in image_sections[:num_boxes]]
        return [self._encode_to_bytes(i) for i in images_warped]

    def _filter_only_handdrawn_lines(self, image: MatLike, length_percent: int = 0.60) -> MatLike:
        """
        Remove ruled pad-paper lines from a binary/edge image leaving only hand-drawn content.
        ## UNSTABLE
        ### Handdrawn lines on white paper get filtered out. FIXME: make fail-safe
        """
        h_kernel_length = int(NORMAL_SIZE * length_percent)

        #### Detect horizontal lines: open with a wide horizontal kernel.
        #### Only structures wider than h_kernel_length survive — i.e. ruled lines.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
        horizontal_candidates = cv2.morphologyEx(image, cv2.MORPH_OPEN, h_kernel)

        self.save_image(
                    self._encode_to_bytes(horizontal_candidates),
                    "./TEMP/output/DEBUG_horizontal_candidates.jpg"
                    )
        # ruled_mask = self._get_ruled_mask(horizontal_candidates, image)
        return cv2.subtract(image, horizontal_candidates)

    def _get_ruled_mask(self, horizontal_candidates: MatLike, image: MatLike):
        """## UNSTABLE
        ### Experimental (removed) substep to filter handdrawn lines."""
        ### Use HoughLinesP to validate: only accept near-perfect horizontal lines
        lines = cv2.HoughLinesP(horizontal_candidates, rho=1, theta=np.pi / 180, threshold=80,
                                minLineLength=int(NORMAL_SIZE * 0.30),
                                maxLineGap=10)

        #### Filter to only near-perfect horizontal lines (angle < 1 degree)
        strict_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 1.0:
                strict_lines.append((x1, y1, x2, y2))

        #### Validate even spacing: ruled lines should be evenly spaced
        # y_positions = sorted(set(y1 for x1, y1, x2, y2 in strict_lines))
        # gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        # median_gap = np.median(gaps)

        ruled_mask = np.zeros_like(image)
        for x1, y1, x2, y2 in strict_lines:
            cv2.line(ruled_mask, (x1, y1), (x2, y2), 255, 3)

        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        ruled_mask = cv2.dilate(ruled_mask, cleanup_kernel, iterations=1)

        return ruled_mask
    #endregion
#endregion



if __name__ == "__main__":
    # ================ DEFINITIONS ================
    FILENAME = "testWhiteC.jpeg"
    GET_INPUT = lambda x : f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
    

    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]
    
    BOX_SEGMENTER = BoxSegmenter()
    AI_EVALUATOR = AIAnswerEvaluator()
    
    image_before_before = BOX_SEGMENTER.load_image(GET_INPUT(FILENAME))
    image_before = BOX_SEGMENTER.scan_page(image_before_before, debug=False)
    images_after_box = BOX_SEGMENTER.get_boxes(image_before, num_boxes=3, debug=True)
    # images_after_box = BOX_SEGMENTER.get_boxes_via_dots(image_before, num_boxes=3, debug=True)

    for i in range(len(images_after_box)):
        label="X"
        # label = AI_EVALUATOR.get_nearest_item_number(images_after_box[i], ["1", "2", "3"])
        BOX_SEGMENTER.save_image(
                    images_after_box[i],
                    GET_OUTPUT(f"{_onlyfilename}/box{i}_item{label}.jpg")
                    )