import sys
import numpy as np
import cv2
from itertools import combinations
from cv2.typing import MatLike

from core.constants import *
from logic.document_scanner import DocumentScanner
from logic.ai_interface import AIAnswerEvaluator
from logic.blob_detector import BlobDetector, DOT_DEDUP_DIST, OversimplifiedBlobDetector
from logic.image_modifier import ImageModifier
from utils import is_valid_quad, mapp, get_robust_aspect_ratio



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
            self.save_image(self._encode_to_bytes(image_cannied), f"{self.debug_dir}/_01A_canny.jpg")

        marker_dots = self._detect_dots(image_original, image_cannied, debug=debug)
        images_answers = self._segment_dots_into_boxes(image_original, marker_dots, debug=debug)

        if images_answers == []:
            raise ValueError("Could not find any dotted boxes.")

        images_warped = [self._warp_from_original(i, image_original) for i in images_answers[:num_boxes]]
        return [self._encode_to_bytes(i) for i in images_warped]

    def _detect_dots(self, image_original: MatLike, image_cannied: MatLike, debug: bool = False) -> list[list[float]]:
        """From image, use marker dots to find section corners."""
        BLOB_DETECTOR = BlobDetector(image_cannied)

        image_binarized = ImageModifier().pseudocanny(image_original)
        if debug:
            self.save_image(self._encode_to_bytes(image_binarized), f"{self.debug_dir}/_01B_binarized.jpg")

        image_eroded = BLOB_DETECTOR.erode_connections(image_binarized)

        ## Pass 1: circular contours
        image_dilated = BLOB_DETECTOR.dilate_dots(image_eroded)
        pts_pass1_contour = BLOB_DETECTOR.detect_dot_contours(image_dilated)

        ## Pass 2: blobs
        pts_pass2_blob = OversimplifiedBlobDetector.detect_white(image_eroded)

        if debug:
            # self.save_image(self._encode_to_bytes(image_lines_removed), f"{self.debug_dir}/_02A_linesRemoved.jpg")
            # for key, img in images_vertical_kernel.items():
            #     self.save_image(self._encode_to_bytes(img), f"{self.debug_dir}/_02A_{key}.jpg")
            self.save_image(self._encode_to_bytes(image_eroded), f"{self.debug_dir}/_02A_eroded.jpg")
            self.save_image(self._encode_to_bytes(image_dilated), f"{self.debug_dir}/_02B_dilated.jpg")

        ## Consensus: keep only dots both methods agree on
        pts_consensus = BlobDetector.intersect_points(pts_pass1_contour, pts_pass2_blob)
        print(f"INFO:\tConsensus dots: {len(pts_consensus)} (contour={len(pts_pass1_contour)})")

        if debug:
            debug_sets = [
                (pts_pass1_contour, (0, 0, 255), "_03A_dots_pass1contour.jpg"),
                (pts_pass2_blob, (0, 0, 255), "_03B_dots_pass2blob.jpg"),
                (pts_consensus, (0, 0, 255), "_04_dots_consensus.jpg"),
                ]

            for pts, color, filename in debug_sets:
                debug_img = image_binarized.copy()
                for p in pts:
                    debug_img = self._highlight_dot(debug_img, (int(p[0]), int(p[1])), color)
                self.save_image(self._encode_to_bytes(debug_img),
                                    f"{self.debug_dir}/{filename}")

        if len(pts_consensus) < 4:
            print(f"INFO:\tOnly {len(pts_consensus)} blobs detected")
            return []

        pts = self._filter_out_dup_pts(pts_consensus)

        if debug:
            debug_img = image_binarized.copy()
            for p in pts:
                debug_img = self._highlight_dot(debug_img, (int(p[0]), int(p[1])))
            self.save_image(self._encode_to_bytes(debug_img),
                                f"{self.debug_dir}/_05_dots_deduped.jpg" )

        return pts

    def _segment_dots_into_boxes(self, image_original: MatLike, pts: list[list[float]], debug: bool = False) -> list[MatLike]:
        """From list of points, crop what seems the most like the dotted boxes (answer sections), and return this."""
        quads = self._group_dots_into_quads(np.array(pts))
        print(f"INFO:\tObtained total of {len(quads)} quads")

        ## Step 1: Collect all valid quads (no approxPolyDP — input is already 4 points)
        valid_quads = []
        for i, q in enumerate(quads):
            contour = q.reshape((-1, 1, 2)).astype(np.int32)

            area = cv2.contourArea(contour)
            if MIN_AREA <= area <= MAX_AREA:
                ## Use get_robust_aspect_ratio instead of axis-aligned boundingRect
                aspect_ratio = get_robust_aspect_ratio(q)
                if 1/MAX_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
                    print(f"INFO:\tAccepted dot-quad {i} (area={area:.0f}, AR={aspect_ratio:.2f})")
                    valid_quads.append(q)
                else:
                    print(f"INFO:\tBad ratio, AR={aspect_ratio:.2f}.")
            else:
                continue

        ## Step 2: Deduplicate overlapping quads (keep the one with larger area)
        deduped_quads = self._deduplicate_quads(valid_quads)
        print(f"INFO:\tAfter dedup: {len(deduped_quads)} quads (was {len(valid_quads)})")

        ## Step 3: Sort by vertical position (top of page first)
        deduped_quads.sort(key=lambda q: np.min(q[:, 1]))

        if debug:
            for i, q in enumerate(deduped_quads):
                contour = q.reshape((-1, 1, 2)).astype(np.int32)
                debug_img = self._highlight_contours(image_original, contour, contour)
                self.save_image(self._encode_to_bytes(debug_img),
                                    f"{self.debug_dir}/_06_sections/box{i}.jpg" )

        return deduped_quads

    #region Secondary functions
    def beautify_scan(self, image_bytes: bytes) -> bytes:
        """Enhance scan by adjusting contrast and brightening. Sana hindi mo taken for granted yung pinagdaanan ko para sayo"""
        array = self._decode_bytes(image_bytes)
        img = self._adjust_contrast(
                            self._brighten(array, amount=-0.05),
                            amount=2.0
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
    def _filter_out_dup_pts(self, pts: list[list[float]]) -> list[list[float]]:
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

    def _group_dots_into_quads(self, pts: np.ndarray) -> list[np.ndarray]:
        """Return only point-sets that could plausibly be rectangle corners."""
        quad_combinations = [np.array([pts[i] for i in c]) for c in combinations(range(len(pts)), 4)]

        quads = []
        for q in quad_combinations:
            ## Spatial pre-filter: skip if bounding box area is too small or too large
            xs = q[:, 0] if q.ndim == 2 else [p[0] for p in q]
            ys = q[:, 1] if q.ndim == 2 else [p[1] for p in q]
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            bbox_area = bbox_w * bbox_h
            if bbox_area < MIN_AREA or bbox_area > MAX_AREA:
                continue

            q_ordered = mapp(q.flatten())
            if is_valid_quad(q_ordered):
                quads.append(q_ordered)

        return quads

    @staticmethod
    def _deduplicate_quads(quads: list[np.ndarray]) -> list[np.ndarray]:
        """Remove overlapping quads. Prefer smaller quads (individual sections) over
        larger cross-section combinations that span multiple physical boxes."""
        if not quads:
            return []

        ## Compute centers
        centers = [np.mean(q, axis=0) for q in quads]
        areas = [cv2.contourArea(q.reshape((-1, 1, 2)).astype(np.int32)) for q in quads]

        ## Two quads overlap if one's center falls inside the other
        ## Greedily pick smallest-area quads first (individual sections are smaller
        ## than cross-box combinations formed from dots of different physical boxes)
        used = [False] * len(quads)
        result = []
        order = sorted(range(len(quads)), key=lambda i: areas[i])  ## smallest first
        for i in order:
            if used[i]:
                continue
            result.append(quads[i])
            used[i] = True
            contour_i = quads[i].reshape((-1, 1, 2)).astype(np.int32)
            ## Mark quads whose center falls inside this quad as used
            for j in range(len(quads)):
                if not used[j]:
                    dist = cv2.pointPolygonTest(contour_i, tuple(centers[j].astype(float)), False)  # type: ignore[call-overload]
                    if dist >= 0:  ## center of j is inside quad i
                        used[j] = True
            ## Also mark quads that contain this quad's center
            for j in range(len(quads)):
                if not used[j]:
                    contour_j = quads[j].reshape((-1, 1, 2)).astype(np.int32)
                    dist = cv2.pointPolygonTest(contour_j, tuple(centers[i].astype(float)), False)  # type: ignore[call-overload]
                    if dist >= 0:  ## center of i is inside quad j
                        used[j] = True

        return result
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
            ruled_line_mask = cv2.inRange(i, 20, 80)  # type: ignore[call-overload]
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

    #region Auxiliary functions
    @staticmethod
    def _build_ruled_mask(candidates: MatLike, shape: tuple,
                          min_span: float = 0.50, max_angle_deg: float = 3.0) -> MatLike:
        """Return a binary mask of near-horizontal, full-width ruled lines."""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        lines = cv2.HoughLinesP(candidates, rho=1, theta=np.pi/180,
                                threshold=50,
                                minLineLength=int(shape[1] * min_span),
                                maxLineGap=20)
        if lines is None:
            return mask
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle <= max_angle_deg:
                cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)  # type: ignore[call-overload]
        return mask

    def _filter_only_handdrawn_lines(self, image: MatLike, length_percent: float = 0.60) -> MatLike:
        """Remove ruled pad-paper lines from a binary/edge image leaving only hand-drawn content."""
        h_kernel_length = int(image.shape[1] * length_percent)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
        candidates = cv2.morphologyEx(image, cv2.MORPH_OPEN, h_kernel)
        ruled_mask = BoxSegmenterOldFunctions._build_ruled_mask(candidates, image.shape)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        ruled_mask = cv2.dilate(ruled_mask, v_kernel, iterations=1)
        return cv2.subtract(image, ruled_mask)
    
    def _regularize_forgivingly(self, image_mat: MatLike) -> list[MatLike]:
        def _pre_canny(gray: MatLike) -> MatLike:
            binary = cv2.adaptiveThreshold(gray, 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 5)
            h_kernel_length = int(gray.shape[1] * 0.55)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_length, 1))
            candidates = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
            ruled_mask = BoxSegmenterOldFunctions._build_ruled_mask(candidates, gray.shape, min_span=0.50, max_angle_deg=3.0)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
            ruled_mask = cv2.dilate(ruled_mask, v_kernel, iterations=1)
            output = gray.copy()
            output[ruled_mask > 0] = 255
            return output

        returnable = self._regularize_image(image_mat,
                                canny_thresholds=(30, 150),
                                additional_pre_canny_step=_pre_canny)
        return returnable
    
    def _dilate_edges(self, image: MatLike, dilate_size: int = 3) -> MatLike:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        image_dilated = cv2.dilate(image, kernel, iterations=2)
        image_closed = cv2.morphologyEx(image_dilated, cv2.MORPH_CLOSE, kernel)
        return image_closed
    #endregion



if __name__ == "__main__":
    # ================ DEFINITIONS ================
    FILENAME = sys.argv[1] if len(sys.argv) > 1 else "testRuledDottedA.jpeg"
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