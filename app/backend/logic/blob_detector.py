from itertools import combinations
from cv2.typing import MatLike
import numpy as np
import cv2
from core.constants import *
from logic.document_scanner import DocumentScanner
from logic.utility import _mapp, _get_robust_aspect_ratio, _is_valid_quad


DOCUMENT_SCANNER = DocumentScanner()


class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0  # dark blobs
        params.filterByArea = True
        params.minArea = NORMAL_SIZE*0.05
        params.maxArea = NORMAL_SIZE*10.0
        params.filterByCircularity = True
        params.minCircularity = 0.55
        params.filterByConvexity = True
        params.minConvexity = 0.80
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        self._detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        return self._detector.detect(image_mat)

    def detect_sections_via_anchors(self, image_mat: MatLike, debug: bool = False) -> list[MatLike]:
        """
        Detect rectangular sections in a preprocessed image using detected dark solid blobs (corners/anchors).

        This function finds keypoints (anchor points) in the input image using a blob detector. It then considers 
        all possible combinations of four keypoints, checks if they can form a valid quadrilateral based on area, 
        aspect ratio, and skew angle, and collects those that pass these checks as valid sections. The returned
        result is an ndarray where each element is a 4x2 array of corner points.

        Args:
            image_mat (MatLike): Input image for blob detection.

        Returns:
            list[MatLike]: An array of detected section corner arrays of shape (num_sections, 4, 2).
        """
        keypoints = self.detect(image_mat)
        if len(keypoints) < 4:
            print(f"INFO:\tOnly {len(keypoints)} blobs detected")
            return []

        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        if debug:
            debug_img = image_mat.copy()
            for p in pts:
                debug_img = self._highlight_dot(debug_img, p)
            DOCUMENT_SCANNER.save_image(DOCUMENT_SCANNER._encode_to_bytes(debug_img), "./TEMP/output/DEBUG_mark_dots.jpg")

        if len(pts) == 4:
            quad_candidates = [pts]
        else:
            quad_candidates = [pts[list(c)] for c in combinations(range(len(pts)), 4)]

        image_good_sections = []
        for i, c in enumerate(quad_candidates):
            c
            
            if _is_valid_quad(c):
                ordered = np.array(_mapp(c.flatten()), dtype=np.float32).reshape(4, 2)
                image_good_sections.append(ordered)
            else:
                continue

        if image_good_sections == []:
            raise ValueError("Could not find any boxes.")
        
        return image_good_sections

    def _highlight_dot(self, image: MatLike, coordinate: tuple[int, int]) -> MatLike:
        """FOR DEBUGGING; Highlight a single dot on the given image by drawing a green filled circle."""
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        x, y = coordinate
        cv2.circle(debug_img, (int(x), int(y)), radius=10, color=(0, 255, 0), thickness=-1)
        return debug_img
