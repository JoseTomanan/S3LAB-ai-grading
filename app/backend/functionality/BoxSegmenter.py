import numpy as np
import cv2
from functionality.DocumentScanner import DocumentScanner, NORMAL_SIZE
from functionality.ai_interface import AIAnswerEvaluator


AREA = NORMAL_SIZE ** 2
MIN_AREA = AREA * 0.01
MAX_AREA = AREA * 0.90



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
        image_dilated = self._dilate_edges(image_cannied)

        contours, _ = cv2.findContours(image_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        images_good_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_AREA < area < MAX_AREA:
                perimeter = cv2.arcLength(c, True)
                approximate = cv2.approxPolyDP(c, 0.06*perimeter, True)
                if len(approximate == 4):
                    approximate = approximate.reshape(4,2)
                    (_, _, w, h) = cv2.boundingRect(approximate)
                    aspect_ratio = w / float(h)
                    if 0.25 <= aspect_ratio <= 4:
                        images_good_contours.append(approximate)
                    else:
                        print(f"INFO:\tBad ratio, AR={aspect_ratio}.")
                else:
                    print(f"INFO:\tFound not box at approximate={approximate}.")
            else:
                print(f"INFO:\tDid not pass for area={area}")

        if images_good_contours == []:
            raise ValueError("Could not find any boxes.")
        
        print(f"INFO:\tResult # of boxes: {len(images_good_contours)}")
        
        images_good_contours = sorted(images_good_contours, key=lambda b : cv2.boundingRect(b)[1])
        
        images_warped = []
        for image in images_good_contours:
            images_warped.append(self._warp_from_original(image, image_original))

        return [self._encode_to_bytes(image) for image in images_warped]


    #region Auxiliary functions
    def _regularize_forgivingly(self, image_mat: cv2.typing.MatLike) -> list[cv2.typing.MatLike]:
        return self._regularize_image(image_mat, canny_thresholds=(30, 150))
    
    def _dilate_edges(self, image: cv2.typing.MatLike, dilate_size: int = 3) -> cv2.typing.MatLike:
        kernel = np.ones((dilate_size, dilate_size), np.uint8)
        image_dilated = cv2.dilate(image, kernel, iterations=2)
        image_closed = cv2.morphologyEx(image_dilated, cv2.MORPH_CLOSE, kernel)
        return image_closed
    #endregion
#endregion


if __name__ == "__main__":
    BOX_SEGMENTER = BoxSegmenter()
    AI_EVALUATOR = AIAnswerEvaluator()
    
    image_before_before = BOX_SEGMENTER.load_image("./TEMP/input/testA.jpeg")
    image_before = BOX_SEGMENTER.scan_page(image_before_before)
    # image_before = BOX_SEGMENTER.load_image("./TEMP/input/alreadyscannedA.jpg")
    images_after_box = BOX_SEGMENTER.get_boxes(image_before, num_boxes=2)

    for i in range(len(images_after_box)):
        label = AI_EVALUATOR.get_item_number(images_after_box[i])
        BOX_SEGMENTER.save_image(images_after_box[i], f"./TEMP/output/+test{i}_itemlabel_{label}.jpg")