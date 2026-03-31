import math

from cv2.typing import MatLike
import cv2
import numpy as np
from core.constants import *



#region Auxiliary functions
def measure_line_thickness(target_image_canny: MatLike, min_line_length_ratio: float = 0.05) -> float:
    """Measure median ruled-line thickness in a binary image.
    Returns 0 if no ruled lines are detected."""
    gray = cv2.cvtColor(target_image_canny, cv2.COLOR_BGR2GRAY) if len(target_image_canny.shape) == 3 \
                    else target_image_canny

    kernel_len = int(NORMAL_SIZE * min_line_length_ratio)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)

    thicknesses = []
    for col in range(0, horizontal.shape[1], 50):
        column = horizontal[:, col]
        in_run = False
        run_len = 0
        for px in column:
            if px > 0:
                in_run = True
                run_len += 1
            else:
                if in_run and run_len > 0:
                    thicknesses.append(run_len)
                in_run = False
                run_len = 0
    return float(np.median(thicknesses)) if thicknesses else 0.0

def _deduplicate_dots(dots: list[tuple], min_dist: float = 10.0) -> list[tuple]:
    """Remove near-duplicate dots, keeping the one with largest area.
    dots: list of (cx, cy, area) tuples."""
    deduped = []
    for d in dots:
        merged = False
        for i, existing in enumerate(deduped):
            if math.hypot(d[0] - existing[0], d[1] - existing[1]) < min_dist:
                # Keep the detection with larger area (more likely the full dot)
                if d[2] > existing[2]:
                    deduped[i] = d
                merged = True
                break
        if not merged:
            deduped.append(d)
    return deduped
#endregion

#region [UNUSED] Auxiliary functions
def _deduplicate_points(pts: list[list[float]], min_dist: float = 10.0) -> list[list[float]]:
    """Remove near-duplicate points, keeping the first occurrence."""
    deduped = []
    for p in pts:
        is_dup = False
        for d in deduped:
            if math.hypot(p[0] - d[0], p[1] - d[1]) < min_dist:
                is_dup = True
                break
        if not is_dup:
            deduped.append(p)
    return deduped

def _is_fillable_contour(contour) -> bool:
    """[UNUSED] Check if a contour is small and compact enough to be a dot."""
    area = cv2.contourArea(contour)
    if area > MAX_DOT_AREA:
        return False
    _, _, w, h = cv2.boundingRect(contour)
    if max(w, h) == 0:
        return False
    if min(w, h) / max(w, h) < 0.05:
        return False
    return True
#endregion


#region Class
class BlobDetector:
    """Note that name is a misnomer; FIXME: Rename to CircleBlobDetector"""
    def __init__(self, image_canny: MatLike):
        line_thickness = measure_line_thickness(image_canny)
        print(f"BACKEND:\tMeasured ruled line thickness: {line_thickness:.1f}px")

        if line_thickness > 0:
            min_dot_diameter = line_thickness * 3
            max_dot_diameter = line_thickness * 15
            dyn_min_area = math.pi * (min_dot_diameter / 2) ** 2
            dyn_max_area = math.pi * (max_dot_diameter / 2) ** 2
        else:
            dyn_min_area = AREA_FACTOR * 62.5
            dyn_max_area = AREA_FACTOR * 6250

        self.dyn_min_area = dyn_min_area
        self.dyn_max_area = dyn_max_area

        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = dyn_min_area
        params.maxArea = dyn_max_area
        params.filterByCircularity = True
        params.minCircularity = 0.10
        params.filterByConvexity = True
        params.minConvexity = 0.40
        params.filterByInertia = True
        params.minInertiaRatio = 0.15
        self._blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect_circular_contours_from_canny(self, img: MatLike) -> list[list[float]]:
        """[UNUSED] Detect dots via inner contours of Canny rings.
        In Canny output, dots appear as hollow rings. The inner contour (hole)
        is isolated from ruled lines and other edge noise, making it a robust
        circular indicator. Uses RETR_CCOMP to find child contours whose
        geometry passes dot filters."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return []

        centroids = []
        for i, contour in enumerate(contours):
            ## Only consider inner contours (those with a parent)
            if hierarchy[0][i][3] == -1:
                continue

            area = cv2.contourArea(contour)
            if not (self.dyn_min_area <= area <= self.dyn_max_area):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < MIN_SOLIDITY:
                continue

            _, _, w, h = cv2.boundingRect(contour)
            bbox_aspect = min(w, h) / max(w, h)
            if bbox_aspect < MIN_BBOX_ASPECT:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy, area))

        deduped = _deduplicate_dots(centroids, min_dist=DOT_DEDUP_DIST)
        filtered = self._filter_by_ring_density(gray, deduped)
        return [[cx, cy] for cx, cy, _ in filtered]

    def _filter_by_ring_density(self, img: MatLike, dots: list[tuple],
                                inner_mult: int = 2, outer_mult: int = 5) -> list[tuple]:
        """Remove dots surrounded by too much white (handwriting clutter).
        dots: list of (cx, cy, area). Returns filtered list of same format.
        Measures white-pixel density in a ring from inner_mult*r to outer_mult*r."""
        h_img, w_img = img.shape[:2]
        result = []
        for cx, cy, area in dots:
            r = max(5, int(math.sqrt(area / math.pi)))
            ix, iy = int(cx), int(cy)

            inner_r = r * inner_mult
            outer_r = r * outer_mult
            y1, y2 = max(0, iy - outer_r), min(h_img, iy + outer_r)
            x1, x2 = max(0, ix - outer_r), min(w_img, ix + outer_r)
            roi = img[y1:y2, x1:x2]
            center = (ix - x1, iy - y1)

            mask_outer = np.zeros_like(roi)
            mask_inner = np.zeros_like(roi)
            cv2.circle(mask_outer, center, outer_r, 255, -1)  # type: ignore[call-overload]
            cv2.circle(mask_inner, center, inner_r, 255, -1)  # type: ignore[call-overload]
            ring_mask = cv2.subtract(mask_outer, mask_inner)

            ring_pixels = np.sum(ring_mask > 0)
            if ring_pixels == 0:
                result.append((cx, cy, area))
                continue
            ring_white = np.sum((roi > 0) & (ring_mask > 0))
            density = ring_white / ring_pixels

            if density <= MAX_RING_DENSITY:
                result.append((cx, cy, area))
        return result

    #region [UNUSED] Auxiliary/secondary functions
    def remove_horizontal_lines(self, img: MatLike, min_line_length_ratio: float = 0.05) -> MatLike:
        """[UNUSED] Remove long horizontal structures (ruled lines) from a binary image, while keeping dots and short strokes intact."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        kernel_len = int(NORMAL_SIZE * min_line_length_ratio)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
        return cv2.subtract(gray, horizontal)

    def detect_dot_contours(self, img: MatLike, debug_images: dict | None = None) -> list[list[float]]:
        """Detect round dot contours from a binary image via geometric filtering.
        First, vertically reconnects dot halves split by horizontal line removal (dual-pass close).
        Then, filters by ring density to reject dots surrounded by handwriting clutter.
        Returns centroid points as list of [x, y].
        If debug_images dict is provided, populates it with intermediate closed images."""
        # FIXME: Move vertical reconnecting to a separate function
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Dual-pass: a small close preserves delicate dots, a tall close
        # reconnects dot halves split by line removal. Union of both
        # catches dots that either kernel alone would miss.
        all_dots = []  # list of (cx, cy, area)
        for close_height in [5, 11]:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, close_height))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
            all_dots.extend(self._find_dot_centroids(closed, min_area=self.dyn_min_area, max_area=self.dyn_max_area))
            if debug_images is not None:
                debug_images[f"vclose_h{close_height}"] = closed

        deduped = _deduplicate_dots(all_dots, min_dist=DOT_DEDUP_DIST)
        filtered = self._filter_by_ring_density(gray, deduped)
        return [[cx, cy] for cx, cy, _ in filtered]

    def _find_dot_centroids(self, img: MatLike, min_area=MIN_DOT_AREA, max_area=MAX_DOT_AREA) -> list[tuple]:
        """Find centroids of contours that pass dot geometry filters.
        Returns list of (cx, cy, area) tuples."""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (min_area <= area <= max_area):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area == 0:
                continue
            solidity = area / hull_area
            if solidity < MIN_SOLIDITY:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_aspect = min(w, h) / max(w, h)
            if bbox_aspect < MIN_BBOX_ASPECT:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy, area))

        return centroids

    @staticmethod
    def intersect_points(pts_a, pts_b, max_dist=DOT_DEDUP_DIST):
        """Keep only points from pts_a that have a nearby match in pts_b.
        Result is deduplicated at max_dist."""
        result = []
        for a in pts_a:
            for b in pts_b:
                if math.hypot(a[0]-b[0], a[1]-b[1]) < max_dist:
                    result.append(a)
                    break
        return _deduplicate_points(result, min_dist=max_dist)

    def fill_blobs(self, img: MatLike) -> MatLike:
        """[UNUSED] Fills small, compact hollow contours (e.g. Canny circles) into solid shapes.
        Skips large enclosed regions and elongated contours (ruled lines)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        filled = np.zeros_like(gray)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:  # has parent → skip (inner contour)
                    continue
                area = cv2.contourArea(contour)
                if area > MAX_DOT_AREA:
                    ## Large contour rejected, but check its children (e.g. corner dots inside a box border)
                    child = hierarchy[0][i][2]  # first child index
                    while child != -1:
                        if _is_fillable_contour(contours[child]):
                            cv2.drawContours(filled, contours, child, 255, thickness=cv2.FILLED)  # type: ignore[call-overload]
                        child = hierarchy[0][child][0]  # next sibling
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if max(w, h) == 0:
                    continue
                if min(w, h) / max(w, h) < 0.05:
                    continue
                cv2.drawContours(filled, contours, i, 255, thickness=cv2.FILLED)  # type: ignore[call-overload]

        return filled

    def erode_connections(self, img: MatLike, kernel_size: int = 3, iterations: int = 4) -> MatLike:
        """Erodes thin line/stroke connections between blobs."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(gray, kernel, iterations=iterations)

    def dilate_dots(self, img: MatLike, kernel_size: int = 3, iterations: int = 4) -> MatLike:
        """Dilate to restore dot mass after erosion (erode + dilate = morphological open)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.dilate(gray, kernel, iterations=iterations)
    #endregion
#endregion


class OversimplifiedBlobDetector:
    """Oversimplified blob detector using OpenCV's SimpleBlobDetector. Used as Pass 2 in the dual-pass dot detection pipeline."""
    @staticmethod
    def _base_detect(image_canny: MatLike, is_dark: bool) -> list[list[float]]:
        params = cv2.SimpleBlobDetector_Params()
        
        params.filterByColor = True
        params.blobColor = 0 if is_dark else 255
        params.filterByArea = True

        line_thickness = measure_line_thickness(image_canny)
        print(f"BACKEND:\tMeasured ruled line thickness: {line_thickness:.1f}px")

        if line_thickness > 0:
            min_dot_diameter = line_thickness * 2
            max_dot_diameter = line_thickness * 15
            params.minArea = math.pi * (min_dot_diameter / 2) ** 2
            params.maxArea  = math.pi * (max_dot_diameter / 2) ** 2
        else:
            print("BACKEND:\tpre naman")
            params.minArea = AREA_FACTOR * 30
            params.maxArea  = AREA_FACTOR * 6250
        
        params.filterByCircularity = True
        params.minCircularity = 0.15
        params.filterByConvexity = True
        params.minConvexity = 0.40
        params.filterByInertia = True
        params.minInertiaRatio = 0.15

        _blob_detector = cv2.SimpleBlobDetector_create(params)
        _keypoints =  _blob_detector.detect(image_canny)
        return [[kp.pt[0], kp.pt[1]] for kp in _keypoints]

    @staticmethod
    def detect(image_canny: MatLike) -> list[list[float]]:
        """[UNUSED]"""
        return OversimplifiedBlobDetector._base_detect(image_canny, is_dark=True)

    @staticmethod
    def detect_white(image_canny: MatLike) -> list[list[float]]:
        return OversimplifiedBlobDetector._base_detect(image_canny, is_dark=False)