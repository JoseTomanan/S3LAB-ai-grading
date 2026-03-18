from cv2.typing import MatLike
import cv2
import numpy as np
from core.constants import NORMAL_SIZE



AREA_FACTOR = (NORMAL_SIZE/1000)**2


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

        dark_params = cv2.SimpleBlobDetector_Params()
        dark_params.filterByColor = True
        dark_params.blobColor = 0
        dark_params.minThreshold = 0
        dark_params.maxThreshold = 100
        dark_params.filterByArea = True
        dark_params.minArea = AREA_FACTOR*62.5
        dark_params.maxArea = AREA_FACTOR*6250
        dark_params.filterByCircularity = True
        dark_params.minCircularity = 0.15
        dark_params.filterByConvexity = True
        dark_params.minConvexity = 0.55
        dark_params.filterByInertia = True
        dark_params.minInertiaRatio = 0.25

        self._dark_detector = cv2.SimpleBlobDetector_create(dark_params)

    def detect(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        """Detect white blobs from cannied image."""
        return self._detector.detect(image_mat)

    def detect_dark_blob(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        """Detect small dark filled dots on a light background (e.g. original grayscale scan)."""
        return self._dark_detector.detect(
                                cv2.cvtColor(image_mat, cv2.COLOR_BGR2GRAY)
                                if len(image_mat.shape) == 3 else image_mat )

    def fill_blobs(self, img: MatLike) -> MatLike:
        """Fills hollow contours (e.g. Canny circles) into solid shapes."""
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
