from cv2.typing import MatLike
import cv2
from core.constants import *
from logic.document_scanner import DocumentScanner


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

