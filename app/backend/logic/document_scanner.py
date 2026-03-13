import numpy as np
import cv2
from cv2.typing import MatLike

NORMAL_SIZE = 2048


def _mapp(h):
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

def _get_robust_aspect_ratio(coords):
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



# ================================
#region Class
class DocumentScanner:
    def scan_page(self,
                    image_bytes: bytes,
                    debug: bool = False,
                    ) -> bytes:
        """Take unscanned image, return scanned image."""
        image = self._decode_bytes(image_bytes)
        image_original, image_cannied = self._regularize_image(image)
        
        if debug:
            self.save_image(
                        self._encode_to_bytes(image_cannied),
                        "./TEMP/output/DEBUG_CANNY.jpg"
                        )

        contours, _ = cv2.findContours(image_cannied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_good_contour = None
        for i, c in enumerate(contours):
            perimeter = cv2.arcLength(c, True)
            approximate = cv2.approxPolyDP(c, 0.04 * perimeter, closed=True)
            
            if debug:
                debug_img = self._highlight_contours(image_cannied, approximate, c)
                self.save_image(
                            self._encode_to_bytes(debug_img),
                            f"./TEMP/output/DEBUG_CONTOUR_{i}.jpg"
                            )
            if len(approximate) == 4:
                image_good_contour = approximate.reshape(4,2)
                break

        if image_good_contour is None:
            raise ValueError("INFO:\tCould not find document outline.")

        image_warped = self._warp_from_original(image_good_contour, image_original)

        return self._encode_to_bytes(image_warped)


    #region Image modification functions
    def load_image(self, image_path: str) -> bytes:
        """Load image path and return as bytes."""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()


    def brighten(self, image_bytes: bytes, amount: float) -> bytes:
        """Scale pixel values with (1 + amount). Amount > 0 increases brightness; < 0 decreases it."""
        image = self._decode_bytes(image_bytes)
        brightened = cv2.convertScaleAbs(image, alpha=1, beta=amount)
        return self._encode_to_bytes(brightened)
    

    def adjust_contrast(self, image_bytes: bytes, amount: float) -> bytes:
        """Increase/decrease contrast by given alpha"""
        image = self._decode_bytes(image_bytes)
        contrasted = cv2.convertScaleAbs(image, alpha=amount, beta=128*(1 - amount))
        return self._encode_to_bytes(contrasted)
    

    def save_image(self, image_bytes: bytes, save_path: str) -> None:
        """Save image to specified path."""
        with open(save_path, "wb") as f:
            ret = f.write(image_bytes)
        if not ret:
            raise ValueError(f"Failed to save image to {save_path}")
        print(f"Image saved to {save_path}")
    #endregion


    #region Private functions
    def _decode_bytes(self, image_bytes: bytes) -> MatLike:
        """Decode bytes into BGR uint8 array"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image bytes")
        return image
    
    def _encode_to_bytes(self, image_matrix: MatLike) -> bytes:
        """Encode BGR uint8 array back to JPEG bytes"""
        ret, buffer = cv2.imencode('.jpg', image_matrix)
        if not ret:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()
    
    def _regularize_image(self,
                          image_mat: MatLike,
                          canny_thresholds: tuple[int,int] = (75, 200),
                          ) -> list[MatLike]:
        """Step before contour ranking. Resize, greyscale, blur, then canny to reduce noises in image."""
        h, w, _ = image_mat.shape
        ratio = w/h

        original_img = cv2.resize(image_mat,
                                    (int(NORMAL_SIZE*ratio), NORMAL_SIZE)
                                    )
        iterated_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        iterated_img = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8)) \
                            .apply(iterated_img)    # CLAHE
        iterated_img = cv2.GaussianBlur(iterated_img, (5,5), 0)
        iterated_img = cv2.Canny(iterated_img, *canny_thresholds)
        
        #### This block finds external contours in the canny-processed image, filters out small contours
        #### (by arc length), draws the remaining contours onto a mask, and then applies a morphological 
        #### closing operation to fill small holes/gaps in the mask. The result is a cleaner binary image 
        #### emphasizing large prominent edges and shapes, which is useful for robust contour detection later.
        contours_raw, _ = cv2.findContours(iterated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(iterated_img)
        for c in contours_raw:
            if cv2.arcLength(c, False) > 100:
                cv2.drawContours(mask, [c], -1, 255, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        iterated_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return [original_img, iterated_img]
    
    def _warp_from_original(self, screen_contour: MatLike, original: MatLike) -> MatLike:
        """Take given screen contour from original image."""
        approximation = _mapp(screen_contour)
        box_ratio = _get_robust_aspect_ratio(approximation)
        box_height = int(NORMAL_SIZE * box_ratio)

        points = np.float32(np.array([
                        [0, 0],
                        [box_height, 0],
                        [box_height, NORMAL_SIZE],
                        [0, NORMAL_SIZE]
                    ]))

        image_transformed = cv2.getPerspectiveTransform(approximation, points)  # pyright: ignore
        image_warped = cv2.warpPerspective(
                                original,
                                image_transformed,
                                dsize=(box_height, NORMAL_SIZE)
                                )

        return image_warped

    def _highlight_contours(self, image_cannied: MatLike, approxPolyDpResult: MatLike, contour: MatLike) -> MatLike:
        """FOR DEBUGGING; Highlight contours and add detected # of verts."""
        debug_img = cv2.cvtColor(image_cannied, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_img, [approxPolyDpResult], -1, (0, 255, 0), 3)
        cv2.putText(debug_img,
                    f"verts={len(approxPolyDpResult)} area={cv2.contourArea(contour):.0f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
        return debug_img

    #endregion

#endregion
# ================================


if __name__ == "__main__":  
    # ================ DEFINITIONS ================
    FILENAME = "testRuledB.jpg"
    GET_INPUT = lambda x : f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
    

    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]
    
    DOCUMENT_SCANNER = DocumentScanner()
    
    image_before = DOCUMENT_SCANNER.load_image(GET_INPUT(FILENAME))
    image_after = DOCUMENT_SCANNER.scan_page(image_before, debug=True)

    DOCUMENT_SCANNER.save_image(
                    image_after,
                    GET_OUTPUT(f"{_onlyfilename}_scan.jpg")
                    )