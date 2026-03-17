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
        params.minArea = MIN_AREA
        params.maxArea = MAX_AREA
        params.filterByCircularity = True
        params.minCircularity = 0.55
        params.filterByConvexity = True
        params.minConvexity = 0.80
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        self._detector = cv2.SimpleBlobDetector_create(params)

    def detect_sections_via_anchors(self, image_preprocessed: np.ndarray, scale: tuple[float, float]) -> list[dict]:
        """
        Detect rectangular sections in a preprocessed image using detected dark solid blobs (corners/anchors).

        This function finds keypoints (anchor points) in the input image using a blob detector. It then considers 
        all possible combinations of four keypoints, checks if they can form a valid quadrilateral based on area, 
        aspect ratio, and skew angle, and collects those that pass these checks as valid sections. The corner 
        coordinates are scaled back to the original image size. Each section returned includes its ordered corners 
        and the corresponding keypoints.

        Args:
            image_preprocessed (np.ndarray): The input image that has already been preprocessed for blob detection.
            scale (tuple[float, float]): Scaling factors (x, y) to map detected keypoint positions back to the original image scale.

        Returns:
            list[dict]: A list of detected sections, where each section is a dict containing 'corners' (the 4 points of the box)
                        and 'keypoints' (the keypoint objects for these corners).
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
