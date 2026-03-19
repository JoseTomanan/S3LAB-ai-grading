import math

from cv2.typing import MatLike
import cv2
import numpy as np
from core.constants import NORMAL_SIZE



AREA_FACTOR = (NORMAL_SIZE/1000)**2

# Contour-based dot detection thresholds
MIN_DOT_AREA = AREA_FACTOR * 50
MAX_DOT_AREA = AREA_FACTOR * 8000
MIN_CIRCULARITY = 0.40
MIN_SOLIDITY = 0.65
MIN_BBOX_ASPECT = 0.35
DOT_DEDUP_DIST = math.sqrt(MAX_DOT_AREA) / 2.5  # ~73px at NORMAL_SIZE=2048
MAX_RING_DENSITY = 0.12  # max white-pixel density in ring around dot (filters handwriting clutter)


class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = AREA_FACTOR*62.5
        params.maxArea = AREA_FACTOR*6250
        params.filterByCircularity = True
        params.minCircularity = 0.15
        params.filterByConvexity = True
        params.minConvexity = 0.55
        params.filterByInertia = True
        params.minInertiaRatio = 0.25

        self._detector = cv2.SimpleBlobDetector_create(params)

        lenient_params = cv2.SimpleBlobDetector_Params()
        lenient_params.filterByColor = True
        lenient_params.blobColor = 255
        lenient_params.filterByArea = True
        lenient_params.minArea = AREA_FACTOR*62.5
        lenient_params.maxArea = AREA_FACTOR*6250
        lenient_params.filterByCircularity = True
        lenient_params.minCircularity = 0.10
        lenient_params.filterByConvexity = True
        lenient_params.minConvexity = 0.40
        lenient_params.filterByInertia = True
        lenient_params.minInertiaRatio = 0.15

        self._lenient_detector = cv2.SimpleBlobDetector_create(lenient_params)

    def remove_horizontal_lines(self, img: MatLike, min_line_length_ratio: float = 0.05) -> MatLike:
        """Remove long horizontal structures (ruled lines) from a binary image.
        Keeps dots and short strokes intact."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        kernel_len = int(NORMAL_SIZE * min_line_length_ratio)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, h_kernel)
        return cv2.subtract(gray, horizontal)

    @staticmethod
    def measure_line_thickness(img: MatLike, min_line_length_ratio: float = 0.05) -> float:
        """Measure median ruled-line thickness in a binary image.
        Returns 0 if no ruled lines are detected."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
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

    def detect_dot_contours(self, img: MatLike, debug_images: dict | None = None, line_thickness: float = 0) -> list[list[float]]:
        """Detect round dot contours from a binary image via geometric filtering.
        Runs a dual-pass vertical close (small then tall) to recover dots that
        were split by horizontal line removal, then filters by ring density
        to reject dots surrounded by handwriting clutter.
        Returns centroid points as list of [x, y].
        If debug_images dict is provided, populates it with intermediate closed images."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Dual-pass: a small close preserves delicate dots, a tall close
        # reconnects dot halves split by line removal. Union of both
        # catches dots that either kernel alone would miss.
        all_dots = []  # list of (cx, cy, area)
        for close_height in [5, 11]:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, close_height))
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, close_kernel)
            all_dots.extend(self._find_dot_centroids(closed))
            if debug_images is not None:
                debug_images[f"vclose_h{close_height}"] = closed

        deduped = self._deduplicate_dots(all_dots, min_dist=DOT_DEDUP_DIST)
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
            cv2.circle(mask_outer, center, outer_r, 255, -1)
            cv2.circle(mask_inner, center, inner_r, 255, -1)
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

    def _find_dot_centroids(self, img: MatLike) -> list[tuple]:
        """Find centroids of contours that pass dot geometry filters.
        Returns list of (cx, cy, area) tuples."""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (MIN_DOT_AREA <= area <= MAX_DOT_AREA):
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

    @staticmethod
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

    def detect_lenient(self, image_mat: MatLike) -> list[list[float]]:
        """Detect white blobs with lenient params. Returns centroids as list of [x, y]."""
        keypoints = self._lenient_detector.detect(image_mat)
        return [[kp.pt[0], kp.pt[1]] for kp in keypoints]

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
        return BlobDetector._deduplicate_points(result, min_dist=max_dist)

    def detect(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        """[DEPRECATED] Detect white blobs from cannied image."""
        return self._detector.detect(image_mat)

    def fill_blobs(self, img: MatLike) -> MatLike:
        """Fills hollow contours (e.g. Canny circles) into solid shapes.
        Note: Still used by detect_dot_contours pipeline."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        contours, hierarchy = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        filled = np.zeros_like(gray)
        if hierarchy is not None:
            for i, contour in enumerate(contours):
                # RETR_CCOMP gives 2-level hierarchy: outer=0, hole=1
                # Only draw top-level contours (no parent), filled solid
                if hierarchy[0][i][3] == -1:  # no parent → outer contour
                    cv2.drawContours(filled, contours, i, 255, thickness=cv2.FILLED)

        return filled

    def erode_connections(self, img: MatLike, kernel_size: int = 3, iterations: int = 4) -> MatLike:
        """Erodes thin line/stroke connections between blobs."""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.erode(gray, kernel, iterations=iterations)
