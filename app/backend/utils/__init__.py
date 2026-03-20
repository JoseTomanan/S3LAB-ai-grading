import numpy as np
from core.constants import MAX_SKEW_DEG, MAX_TILT_DEG, SECTION_CORNER_ANGLE_TOL



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


def is_valid_quad(pts: np.ndarray, max_skew_deg: float = MAX_SKEW_DEG, max_tilt_deg: float = MAX_TILT_DEG, tol_deg: float = SECTION_CORNER_ANGLE_TOL) -> bool:
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
    _tl, _tr, _br, _bl = ordered

    angle_top = np.degrees(np.arctan2(_tr[1] - _tl[1], _tr[0] - _tl[0]))
    angle_bot = np.degrees(np.arctan2(_br[1] - _bl[1], _br[0] - _bl[0]))
    if abs(angle_top - angle_bot) > max_skew_deg:
        return False

    ## Check absolute tilt: top/bottom edges should be near-horizontal
    avg_tilt = (abs(angle_top) + abs(angle_bot)) / 2
    if avg_tilt > max_tilt_deg:
        return False

    #### Check if corner angle is approximately 90 degrees for all
    corners = [
            (_bl, _tl, _tr),   # angle at TL
            (_br, _bl, _tl),   # angle at BL
            (_tr, _br, _bl),   # angle at BR
            (_tl, _tr, _br),   # angle at TR
            ]
    for a, vertex, b in corners:
        v1 = a - vertex
        v2 = b - vertex
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
        if abs(angle - 90) > tol_deg:
            return False
    
    return True
