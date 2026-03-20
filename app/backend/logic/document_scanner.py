import pathlib
from typing import Callable
import numpy as np
import cv2
from cv2.typing import MatLike
from core.constants import NORMAL_SIZE, MIN_PAGE_AREA, MAX_AREA, MAX_ASPECT_RATIO
from utils import mapp, get_robust_aspect_ratio, is_valid_quad



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
                        "./TEMP/output/DEBUG/CANNY.jpg"
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
                            f"./TEMP/output/DEBUG/CONTOUR_{i}.jpg"
                            )
            if len(approximate) == 4:
                candidate = approximate.reshape(4, 2)
                if self._is_good_page_contour(candidate):
                    image_good_contour = candidate
                    break

        if image_good_contour is None:
            image_good_contour = self._fallback_otsu_detection(image_original)

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
        path = pathlib.Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            ret = f.write(image_bytes)
        if not ret:
            raise ValueError(f"Failed to save image to {save_path}")
        print(f"INFO:\tImage saved --> {save_path}")
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
                          gaussian_blur_kernel_size: tuple[int,int] | None = (5,5),
                          additional_pre_canny_step: Callable[MatLike, MatLike] = None
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

        if gaussian_blur_kernel_size is not None:
            iterated_img = cv2.GaussianBlur(iterated_img, gaussian_blur_kernel_size, 0)
        
        if additional_pre_canny_step is not None:
            iterated_img = additional_pre_canny_step(iterated_img)
        
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
        approximation = mapp(screen_contour)
        box_ratio = get_robust_aspect_ratio(approximation)
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

    def _is_good_page_contour(self, approximate: MatLike) -> bool:
        """Validate that a 4-point contour is a plausible page outline."""
        contour = approximate.reshape((-1, 1, 2)).astype(np.int32)
        area = cv2.contourArea(contour)
        if not (MIN_PAGE_AREA <= area <= MAX_AREA):
            return False

        quad = approximate.reshape(4, 2)
        ratio = get_robust_aspect_ratio(quad)
        if ratio > MAX_ASPECT_RATIO or ratio < 1.0 / MAX_ASPECT_RATIO:
            return False

        if not is_valid_quad(quad, max_tilt_deg=180.0):
            return False

        return True

    def _fallback_otsu_detection(self, image_original: MatLike):
        """Fallback: use Otsu threshold to find white paper on dark background."""
        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximate = cv2.approxPolyDP(c, 0.04 * perimeter, closed=True)
            if len(approximate) == 4:
                candidate = approximate.reshape(4, 2)
                if self._is_good_page_contour(candidate):
                    return candidate

        return None

    def _highlight_contours(self, image_mat: MatLike, approxPolyDpResult: MatLike, contour: MatLike) -> MatLike:
        """FOR DEBUGGING; Highlight contours and add detected # of verts."""
        debug_img = image_mat.copy()
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
    FILENAME = "testRuledDottedF.jpeg"
    GET_INPUT = lambda x : f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
    

    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]
    
    DOCUMENT_SCANNER = DocumentScanner()
    
    image_before = DOCUMENT_SCANNER.load_image(GET_INPUT(FILENAME))
    image_after = DOCUMENT_SCANNER.scan_page(image_before, debug=True)

    DOCUMENT_SCANNER.save_image(
                    image_after,
                    GET_OUTPUT(f"{_onlyfilename}/scan.jpg")
                    )