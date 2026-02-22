import numpy as np
import cv2

from functionality.DocumentScanner import DocumentScanner, NORMAL_SIZE


AREA = NORMAL_SIZE ** 2
MIN_AREA = AREA * 0.05
MAX_AREA = AREA * 0.75


# ================================
#region Class
# ================================
class BoxSegmenter(DocumentScanner):
    def get_boxes(self, 
                    image_bytes: bytes,
                    num_boxes: int
                    ) -> list[bytes]:
        """Get best boxes (non-overlapping) from the image given. Note that image is expected to have been scanned already."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_forgivingly(image)

        contours, _ = cv2.findContours(image_cannied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        images_good_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                perimeter = cv2.arcLength(c, True)
                approximate = cv2.approxPolyDP(c, 0.15*perimeter, True)
                if len(approximate == 4):
                    (_, _, w, h) = cv2.boundingRect(approximate)
                    aspect_ratio = w / float(h)
                    if 0.4 <= aspect_ratio <= 2.5:
                        images_good_contours.append(approximate)

        if images_good_contours == []:
            raise ValueError("INFO:\tCould not find any boxes.")
        
        images_good_contours = sorted(images_good_contours, key=lambda b : cv2.boundingRect(b)[1])
        
        images_warped = []
        for image in images_good_contours:
            images_warped.append(self._warp_from_original(image, image_original))

        return [self._encode_to_bytes(image) for image in images_warped]


    #region Auxiliary functions
    def _regularize_forgivingly(self, image_mat: cv2.typing.MatLike) -> list[cv2.typing.MatLike]:
        return self._regularize_image(image_mat, canny_thresholds=(30, 150))
    #endregion
#endregion


if __name__ == "__main__":
    ...