from cv2.typing import MatLike
import cv2
from core.constants import NORMAL_SIZE


class BlobDetector:
    def __init__(self):
        params = cv2.SimpleBlobDetector_Params()
        
        params.filterByColor = True
        params.blobColor = 0  # dark blobs
        params.minThreshold = 10
        params.maxThreshold = 180
        params.thresholdStep = 5
        
        areaFactor = (NORMAL_SIZE/1000)**2
        params.filterByArea = True
        params.minArea = areaFactor * 50
        params.maxArea = areaFactor * 6250
        
        params.filterByCircularity = True
        params.minCircularity = 0.30
        
        params.filterByConvexity = True
        params.minConvexity = 0.70
        
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        
        self._detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, image_mat: MatLike) -> list[cv2.KeyPoint]:
        return self._detector.detect(image_mat)

