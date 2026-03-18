import numpy as np
from core.constants import *



def mapp(h):
    """
    Reorders and reorganizes a set of 4 points (corners) based on their spatial properties.
    
    Takes a flattened array of 4 points (8 coordinates) and reshapes it to (4, 2),
    then reorders them by:
    - Index 0: Point with minimum sum of coordinates (top-left area)
    - Index 1: Point with minimum difference between x and y (leftmost)
    - Index 2: Point with maximum sum of coordinates (bottom-right area)
    - Index 3: Point with maximum difference between x and y (rightmost)
    
    Args:
        h: Array-like of shape (8,) containing 4 points as flattened coordinates
        
    Returns:
        np.ndarray: Reordered points of shape (4, 2) with dtype float32
    """
    h = h.reshape((4, 2))
    hnew = np.zeros(
                (4, 2),
                dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def get_robust_aspect_ratio(coords):
    """
    Calculate the aspect ratio of a quadrilateral defined by its corner coordinates.
    This function takes a set of coordinates representing the corners of a quadrilateral,
    sorts them by angle relative to their center point, and computes the aspect ratio
    (width to height) by averaging the lengths of opposite sides.
    Args:
        coords: Array-like of shape (4, 2) containing the (x, y) coordinates of the
                quadrilateral's four corners.
    Returns:
        float: The aspect ratio (width / height) of the quadrilateral.
    """
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

    return w / h


def is_valid_quad(pts: np.ndarray) -> bool:
    """
    Check if a set of four points forms a valid section (quadrilateral)
    based on area, aspect ratio, and skew angle.

    This function ALLOWS non-perfect quadrilaterals. It considers a quadrilateral
    valid if it approximately meets the size, shape, and skew constraints—i.e.,
    the four points do not need to form a mathematically perfect rectangle or square,
    but must "closely" resemble one within tolerance defined by constants.

    Args:
        pts (np.ndarray): 4x2 array of corner points.

    Returns:
        bool: True if the points approximate a valid quadrilateral section, False otherwise.
    """
    ordered = mapp(pts.flatten())
    tl, tr, br, bl = ordered
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2

    if w * h < MIN_AREA:
        return False

    aspect = get_robust_aspect_ratio(pts)
    if aspect > MAX_ASPECT_RATIO or aspect < 1/MAX_ASPECT_RATIO:
        return False

    angle_top = np.degrees(np.arctan2(tr[1] - tl[1], tr[0] - tl[0]))
    angle_bot = np.degrees(np.arctan2(br[1] - bl[1], br[0] - bl[0]))
    if abs(angle_top - angle_bot) > MAX_SKEW_DEG:
        return False
    
    # if not _corners_are_right_angles(np.array(ordered)):
    #     return False
    
    return True


def _corners_are_right_angles(ordered: np.ndarray, tol_deg: float=10.0) -> bool:
    """Return whether or not corners are approximately 90 degrees"""
    tl, tr, br, bl = ordered
    corners = [
        (bl, tl, tr),   # angle at TL
        (tl, tr, br),   # angle at TR
        (tr, br, bl),   # angle at BR
        (br, bl, tl),   # angle at BL
    ]
    for a, vertex, b in corners:
        v1 = a - vertex
        v2 = b - vertex
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if abs(angle - 90) > tol_deg:
            return False
    return True 
