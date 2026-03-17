from itertools import combinations
import numpy as np
import cv2
from core.constants import *
from logic.document_scanner import _mapp, _get_robust_aspect_ratio



def _is_valid_quad(pts: np.ndarray) -> bool:
    """Get points, return whether or not valid section (through area, aspect ratio, skew angle)."""
    ordered = _mapp(pts.flatten())
    tl, tr, br, bl = ordered
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    if w * h < MIN_AREA:
        return False

    aspect = _get_robust_aspect_ratio(pts)
    if aspect > MAX_ASPECT_RATIO or aspect < 1/MAX_ASPECT_RATIO:
        return False

    angle_top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    angle_bot = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    if abs(angle_top - angle_bot) > MAX_SKEW_DEG:
        return False
    return True



class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 0  # dark blobs
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        params.filterByCircularity = True
        params.minCircularity = 0.55
        params.filterByConvexity = True
        params.minConvexity = 0.80
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        self._detector = cv2.SimpleBlobDetector_create(params)

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
        keypoints = self._detector.detect(image_preprocessed)
        if len(keypoints) < 4:
            return []

        pts = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        scale_x, scale_y = scale

        quad_candidates = (
                    [list(range(4))] if len(pts) == 4
                    else [list(c) for c in combinations(range(len(pts)), 4)]
                    )

        sections = []
        for indices in quad_candidates:
            quad_pts = pts[indices]
            if not _is_valid_quad(quad_pts):
                continue

            ordered = _mapp(quad_pts.flatten())
            ordered_orig = ordered * np.array([scale_x, scale_y])
            sections.append({
                        'corners': ordered_orig,
                        'keypoints': [keypoints[i] for i in indices],
                        })

        sections.sort(key=lambda s: s['corners'][0][1])
        return sections
