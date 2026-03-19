import numpy as np
import cv2
from itertools import combinations
from cv2.typing import MatLike

from core.constants import *
from logic.document_scanner import DocumentScanner
from logic.ai_interface import AIAnswerEvaluator
from logic.blob_detector import BlobDetector, DOT_DEDUP_DIST, OversimplifiedBlobDetector

from utils import is_valid_quad, mapp



#region Class
class BoxSegmenter(DocumentScanner):
    debug_dir = "./TEMP/output/DEBUG"

    def get_answer_sections(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """Use dots to find section corners, then lines to verify rectangle that serves as section."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_image(image, canny_thresholds=(30,150),
                                                                gaussian_blur_kernel_size=None)
        # image_dilated = self._dilate_edges(image_cannied)

        if debug:
            self.save_image(self._encode_to_bytes(image_cannied), f"{self.debug_dir}/_01_canny.jpg")

        images_answers = self._detect_dotted_boxes(image_original, image_cannied, debug=debug)
        if images_answers == []:
            raise ValueError("Could not find any dotted boxes.")

        images_warped = [self._warp_from_original(i, image_original) for i in images_answers[:num_boxes]]
        return [self._encode_to_bytes(i) for i in images_warped]

    def _detect_dotted_boxes(self, image_original: MatLike, image_cannied: MatLike, debug: bool = False):
        """From canny and original image, use dots to find section corners, then lines to verify rectangle that serves as section."""
        BLOB_DETECTOR = BlobDetector(image_cannied)

        # Pass 1: detect circular contours directly from Canny (no fill/line-removal needed)
        pts_pass1_contour = BLOB_DETECTOR.detect_circular_contours_from_canny(image_cannied)

        ## Pass 2: lenient SimpleBlobDetector on filled+eroded image
        # image_holes_filled = BLOB_DETECTOR.fill_blobs(image_cannied)
        # image_eroded = BLOB_DETECTOR.erode_connections(image_holes_filled)
        # pts_pass2_blob = BLOB_DETECTOR.detect_simple_blobs(image_eroded)
        pts_pass2_blob = OversimplifiedBlobDetector.detect(image_cannied)

        # Consensus: keep only dots both methods agree on
        pts_consensus = BlobDetector.intersect_points(pts_pass1_contour, pts_pass2_blob)
        print(f"INFO:\tConsensus dots: {len(pts_consensus)} (contour={len(pts_pass1_contour)}, blob={len(pts_pass2_blob)})")

        if debug:
            debug_sets = [
                (pts_pass1_contour, (0, 255, 0), "_02A_dots_pass1contour.jpg"),
                (pts_pass2_blob, (0, 255, 255), "_02B_dots_pass2blob.jpg"),
                (pts_consensus, (0, 0, 255), "_03_dots_consensus.jpg"),
                ]

            for pts, color, filename in debug_sets:
                debug_img = image_cannied.copy()
                for p in pts:
                    debug_img = self._highlight_dot(debug_img, p, color)
                self.save_image(self._encode_to_bytes(debug_img),
                                    f"{self.debug_dir}/{filename}")

        if len(pts_consensus) < 4:
            print(f"INFO:\tOnly {len(pts_consensus)} blobs detected")
            return []

        pts = self._filter_out_dup_pts(pts_consensus)

        if debug:
            debug_img = image_cannied.copy()
            for p in pts:
                debug_img = self._highlight_dot(debug_img, p)
            self.save_image(self._encode_to_bytes(debug_img),
                                f"{self.debug_dir}/_04_dots_deduped.jpg" )

        quads = self._group_dots_into_quads(pts)
        print(f"INFO:\tObtained total of {len(quads)} quads")

        image_good_sections = []
        for i, q in enumerate(quads):
            contour = q.reshape((-1, 1, 2)).astype(np.int32)
            
            area = cv2.contourArea(contour)
            if MIN_AREA <= area <= MAX_AREA:
                perimeter = cv2.arcLength(contour, True)
                approximate = cv2.approxPolyDP(contour, 0.06*perimeter, True)
                
                if len(approximate) == 4:
                    if debug:
                        debug_img = self._highlight_contours(image_cannied, approximate, contour)
                        self.save_image(self._encode_to_bytes(debug_img),
                                            f"{self.debug_dir}/_05_sections/box{i}.jpg" )
                    approximate = approximate.reshape(4, 2)
                    (_, _, w, h) = cv2.boundingRect(approximate)
                    aspect_ratio = w / float(h)
                    if 1/MAX_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                        print(f"INFO:\tAccepted and stored dot-quad {i}")
                        image_good_sections.append(approximate)
                    else:
                        print(f"INFO:\tBad ratio, AR={aspect_ratio}.")
                else:
                    print(f"INFO:\tFound non-box at approxPolyDP of dot-quad {i}")
            else:
                # print(f"INFO:\tDid not pass for area={area}")
                continue

        return image_good_sections

    #region Secondary functions
    def beautify_scan(self, image_bytes: bytes) -> bytes:
        """Enhance scan by adjusting contrast and brightening. Sana hindi mo taken for granted yung pinagdaanan ko para sayo"""
        array = self._decode_bytes(image_bytes)
        img = self._adjust_contrast(
                            self._brighten(array, amount=0.175),
                            amount=1.3
                            )
        return self._encode_to_bytes(img)

    def get_boxes(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """[DEPRECATED] Get best boxes (non-overlapping) from a scanned image. Currently tuned for white paper only."""
        return BoxSegmenterOldFunctions().get_boxes(image_bytes, num_boxes, debug)

    def get_boxes_via_dots(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """[DEPRECATED] [CURRENTLY UNUSED] Same as `get_boxes` function, but uses solid fill blobs as section indicator (instead of handdrawn boxes)."""
        return BoxSegmenterOldFunctions().get_boxes_via_dots(image_bytes, num_boxes, debug)
    #endregion

    #region Auxiliary functions: Answer section detection
    def _filter_out_dup_pts(self, pts: np.ndarray) -> np.ndarray:
        """Filter out duplicate points within DOT_DEDUP_DIST of each other."""
        filtered_pts = []
        for p in pts:
            is_similar = False
            for fp in filtered_pts:
                if np.linalg.norm(np.array(p) - np.array(fp)) < DOT_DEDUP_DIST:
                    is_similar = True
                    break
            if not is_similar:
                filtered_pts.append(p)
        return filtered_pts

    def _group_dots_into_quads(self, pts: np.ndarray) -> bool:
        """Return only point-sets that could plausibly be rectangle corners."""
        # quad_combinations = [pts[list(c)] for c in combinations(range(len(pts)), 4)]
        quad_combinations = [np.array([pts[i] for i in c]) for c in combinations(range(len(pts)), 4)]

        quads = []
        for q in quad_combinations:
            q_ordered = mapp(q.flatten())
            if is_valid_quad(q_ordered):
                quads.append(q_ordered)

        return quads
    #endregion

    #region Auxiliary functions: Image preloading, postloading
    def _highlight_dot(self, image: MatLike, coordinate: tuple[int, int], color: tuple[int,int,int] = (0,255,0)) -> MatLike:
        """FOR DEBUGGING; Highlight a single dot on the given image by drawing a green filled circle."""
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        x, y = coordinate
        cv2.circle(debug_img, (int(x), int(y)), radius=10, color=color, thickness=-1)
        return debug_img

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

    #region [UNUSED] Auxiliary functions
    def _regularize_forgivingly(self, image_mat: MatLike) -> list[MatLike]:
        """[UNUSED]"""
        def _pre_canny(i: MatLike) -> MatLike:
            ruled_line_mask = cv2.inRange(i, 20, 80)
            h_kernel_length = int(NORMAL_SIZE*0.30)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
            ruled_line_mask = cv2.morphologyEx(ruled_line_mask, cv2.MORPH_CLOSE, horizontal_kernel)
            self.save_image(
                    self._encode_to_bytes(ruled_line_mask),
                    f"{self.debug_dir}/horizontal_candidates.jpg"
                    )
            return cv2.subtract(i, ruled_line_mask)
        # return self._regularize_image(image_mat, (30, 150), _pre_canny)
        return self._regularize_image(image_mat, canny_thresholds=(30, 150))
    
    def _dilate_edges(self, image: MatLike, dilate_size: int = 3) -> MatLike:
        """[UNUSED]"""
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        image_dilated = cv2.dilate(image, kernel, iterations=2)
        image_closed = cv2.morphologyEx(image_dilated, cv2.MORPH_CLOSE, kernel)
        return image_closed
    #endregion
#endregion


#region Archive of old box segmenter functions
class BoxSegmenterOldFunctions(BoxSegmenter):
    """Container/archive for deprecated functions."""
    def get_boxes(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """Get best boxes (non-overlapping) from a scanned image. Currently tuned for white paper only."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_forgivingly(image)
        image_dilated = self._dilate_edges(image_cannied)

        if debug:
            self.save_image(
                        self._encode_to_bytes(image_dilated),
                        f"{self.debug_dir}/canny_regularize_dilate.jpg"
                        )

        images_good_contours = self._detect_contours(image_dilated, image_cannied, debug=debug)
        if images_good_contours == []:
            raise ValueError("Could not find any boxes.")
        
        print(f"INFO:\tResult # of boxes: {len(images_good_contours)} (take top {num_boxes})")
        images_good_contours = sorted(images_good_contours, key=lambda b : cv2.boundingRect(b)[1])
        images_warped = [self._warp_from_original(c, image_original) for c in images_good_contours]

        return [self._encode_to_bytes(i) for i in images_warped]

    def _detect_contours(self, image_dilated: MatLike, image_cannied: MatLike, debug: bool = False) -> list[MatLike]:
        """From `image_dilated` and `image_cannied`, get contours, then return only the contours that look the most like an answer box."""
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
                                f"{self.debug_dir}/contours/box{i}.jpg"
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

        return images_good_contours

    def get_boxes_via_dots(self, image_bytes: bytes, num_boxes: int, debug: bool = False) -> list[bytes]:
        """Same as `get_boxes` function, but uses solid fill blobs as section indicator (instead of handdrawn boxes)."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_forgivingly(image)
        image_dilated = self._dilate_edges(image_cannied)
        
        if debug:
            self.save_image(
                    self._encode_to_bytes(image_dilated),
                    f"{self.debug_dir}/canny_regularize_dilate.jpg"
                    )

        images_good_sections = self._detect_boxes_via_dots(image_original, debug=debug)
        if images_good_sections == []:
            raise ValueError("Could not find any boxes.")
        images_warped = [self._warp_from_original(i, image_original) for i in images_good_sections[:num_boxes]]

        return [self._encode_to_bytes(i) for i in images_warped]

    def _detect_boxes_via_dots(self, image_mat: MatLike, debug: bool = False) -> list[MatLike]:
        """Detect rectangular sections in image using dark solid blobs (corners)"""
        BLOB_DETECTOR = BlobDetector()

        keypoints = BLOB_DETECTOR.detect(image_mat)
        if len(keypoints) < 4:
            print(f"INFO:\tOnly {len(keypoints)} blobs detected")
            return []

        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        if debug:
            debug_img = image_mat.copy()
            for p in pts:
                debug_img = self._highlight_dot(debug_img, p)
            self.save_image(self._encode_to_bytes(debug_img), f"{self.debug_dir}/canny_marked_dots.jpg")

        if len(pts) == 4:
            quad_candidates = [pts]
        else:
            quad_candidates = [pts[list(c)] for c in combinations(range(len(pts)), 4)]

        image_good_sections = []
        for i, q in enumerate(quad_candidates):
            if not is_valid_quad(q):
                continue

            ordered = np.array(mapp(q.flatten()), dtype=np.float32).reshape(4, 2)
            contour = np.array(ordered, dtype=np.int32).reshape((-1, 1, 2))
            perimeter = cv2.arcLength(contour, True)
            approximate = cv2.approxPolyDP(contour, 0.06 * perimeter, True)

            if len(approximate) == 4:
                if debug:
                    debug_img = self._highlight_contours(cv2.cvtColor(image_mat, cv2.COLOR_BGR2GRAY), approximate, contour)
                    self.save_image(
                                self._encode_to_bytes(debug_img),
                                f"{self.debug_dir}/blobs/box{i}.jpg"
                                )
                approximate = approximate.reshape(4, 2)
                (_, _, w, h) = cv2.boundingRect(approximate)
                aspect_ratio = w / float(h)
                if 1 / MAX_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                    print(f"INFO:\tAccepted and stored dot-quad {i}")
                    image_good_sections.append(approximate)
                else:
                    print(f"INFO:\tBad ratio, AR={aspect_ratio}.")
            else:
                print(f"INFO:\tFound non-box at approxPolyDP of dot-quad {i}")

        return image_good_sections
#endregion


# MARK: Main
if __name__ == "__main__":
    # ================ DEFINITIONS ================
    FILENAME = "testRuledDottedD.jpeg"
    GET_INPUT = lambda x : f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
    

    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]
    
    BOX_SEGMENTER = BoxSegmenter()
    BOX_SEGMENTER.debug_dir = f"./TEMP/output/{_onlyfilename}"
    AI_EVALUATOR = AIAnswerEvaluator()
    
    image_before_before = BOX_SEGMENTER.load_image(GET_INPUT(FILENAME))
    image_before = BOX_SEGMENTER.scan_page(image_before_before, debug=True)
    images_after_box = BOX_SEGMENTER.get_answer_sections(image_before, num_boxes=3, debug=True)

    for i, b in enumerate(images_after_box):
        BOX_SEGMENTER.save_image(b, GET_OUTPUT(f"{_onlyfilename}/section{i}.jpg"))