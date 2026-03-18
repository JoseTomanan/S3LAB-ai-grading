from cv2.typing import MatLike
import cv2
from core.constants import NORMAL_SIZE



AREA_FACTOR = (NORMAL_SIZE/1000)**2

class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        
        params.filterByColor = True
        params.blobColor = 0  # dark blobs
        params.minThreshold = 10
        params.maxThreshold = 180
        params.thresholdStep = 5
        
        params.filterByArea = True
        params.minArea = AREA_FACTOR*62.5
        params.maxArea = AREA_FACTOR*6250
        
        params.filterByCircularity = True
        params.minCircularity = 0.20
        
        params.filterByConvexity = True
        params.minConvexity = 0.70
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.25
        
        self._detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        return self._detector.detect(image_mat)

