import pathlib
from typing import Callable
import numpy as np
import cv2
from cv2.typing import MatLike
from core.constants import NORMAL_SIZE, MIN_PAGE_AREA, MAX_PAGE_AREA, MAX_ASPECT_RATIO, BORDER_MARGIN_RATIO
from utils import mapp, get_robust_aspect_ratio, is_valid_quad



# ================================
#region Class
class DocumentScanner:
    debug_dir = "./TEMP/output/DEBUG"

    def scan_page(self,
                    image_bytes: bytes,
                    debug: bool = False,
                    ) -> bytes:
        """
        [WRAPPER] Take unscanned image, return scanned image.
        """
        image = self._decode_bytes(image_bytes)
        image_processed = self.scan_page_mat(image, debug)
        return self._encode_to_bytes(image_processed)

    def scan_page_mat(self, image: MatLike, debug: bool = False) -> MatLike:
        """Take unscanned image (MatLike), return scanned image (MatLike). No encode/decode overhead."""
        image_original, image_cannied = self._regularize_image(image)

        if debug:
            self.save_image(
                        self._encode_to_bytes(image_cannied),
                        f"{self.debug_dir}/CANNY.jpg"
                        )

        contours, _ = cv2.findContours(image_cannied, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        candidates = []
        for i, c in enumerate(contours):
            perimeter = cv2.arcLength(c, True)
            approximate = cv2.approxPolyDP(c, 0.04 * perimeter, closed=True)

            if debug:
                debug_img = self._highlight_contours(image_cannied, approximate, c)
                self.save_image(
                            self._encode_to_bytes(debug_img),
                            f"{self.debug_dir}/CONTOUR_{i}.jpg"
                            )
            if len(approximate) == 4:
                candidate = approximate.reshape(4, 2)
                original_area = cv2.contourArea(c)
                if self._is_good_page_contour(candidate, original_area):
                    is_border = self._is_border_contour(candidate, image_cannied.shape)
                    candidates.append((candidate, is_border, c))

        image_good_contour = self._pick_best_contour(candidates)

        if image_good_contour is not None and debug:
            selected_quad, _, selected_c = next(
                (item for item in candidates if np.array_equal(item[0], image_good_contour)), (image_good_contour, None, None)
            )
            contour_img = self._highlight_contours(image_original, image_good_contour.reshape((-1, 1, 2)).astype(np.int32), selected_c if selected_c is not None else image_good_contour.reshape((-1, 1, 2)).astype(np.int32))
            self.save_image(
                        self._encode_to_bytes(contour_img),
                        f"{self.debug_dir}/_00_SCANNEDCONTOUR.jpg"
                        )

        if image_good_contour is None:
            image_good_contour = self._fallback_otsu_detection(image_original)

        if image_good_contour is None:
            image_good_contour = self._fallback_low_contrast_detection(image_original)

        if image_good_contour is None:
            raise ValueError("Could not find document outline.")

        return self._warp_from_original(image_good_contour, image_original)


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
        print(f"BACKEND:\tImage saved --> {save_path}")
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
                          additional_pre_canny_step: Callable[[MatLike], MatLike] | None = None,
                          clahe_clip_limit: float = 0.5
                          ) -> list[MatLike]:
        """Step before contour ranking. Resize, greyscale, blur, then canny to reduce noises in image."""
        h, w, _ = image_mat.shape
        ratio = w/h

        original_img = cv2.resize(image_mat,
                                    (int(NORMAL_SIZE*ratio), NORMAL_SIZE)
                                    )
        iterated_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        iterated_img = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8,8)) \
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
                cv2.drawContours(mask, [c], -1, 255, 2)  # type: ignore[call-overload]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        iterated_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return [original_img, iterated_img]
    
    def _warp_from_original(self, screen_contour: MatLike, original: MatLike) -> MatLike:
        """Take given screen contour from original image."""
        approximation = mapp(screen_contour)
        box_ratio = get_robust_aspect_ratio(approximation)
        box_height = int(NORMAL_SIZE * box_ratio)

        points = np.array([
                        [0, 0],
                        [box_height, 0],
                        [box_height, NORMAL_SIZE],
                        [0, NORMAL_SIZE]
                    ], dtype=np.float32)

        image_transformed = cv2.getPerspectiveTransform(approximation, points)  # pyright: ignore
        image_warped = cv2.warpPerspective(
                                original,
                                image_transformed,
                                dsize=(box_height, NORMAL_SIZE)
                                )

        return image_warped

    def _is_good_page_contour(self, approximate: MatLike, original_contour_area: float | None = None) -> bool:
        """Validate that a 4-point contour is a plausible page outline."""
        contour = approximate.reshape((-1, 1, 2)).astype(np.int32)
        area = cv2.contourArea(contour)
        if not (MIN_PAGE_AREA <= area <= MAX_PAGE_AREA):
            return False
        ## Also reject if the original (un-approximated) contour is too large;
        ## rounded shapes like binder edges can approximate to a smaller quad that passes the area check.
        if original_contour_area is not None and original_contour_area > MAX_PAGE_AREA:
            return False

        quad = approximate.reshape(4, 2)
        ratio = get_robust_aspect_ratio(quad)
        if ratio > MAX_ASPECT_RATIO or ratio < 1.0 / MAX_ASPECT_RATIO:
            return False

        if not is_valid_quad(quad, max_tilt_deg=180.0):
            return False

        return True

    def _is_border_contour(self, quad: MatLike, image_shape: tuple) -> bool:
        """Return True if 3 or more corners of quad are near the image border.
        Contours hugging the border are likely the scene boundary (e.g. binder edge), not the paper."""
        h, w = image_shape[:2]
        margin = BORDER_MARGIN_RATIO * max(h, w)
        is_near_border_count = sum(
            1 for (x, y) in quad
            if x < margin or x > w - margin or y < margin or y > h - margin
        )
        return is_near_border_count >= 3

    def _pick_best_contour(self, candidates: list) -> MatLike | None:
        """From a list of (quad, is_border, original_contour) tuples, return the best quad.
        Prefers non-border contours. Among ties, picks the first (largest by area, since
        candidates are already in descending area order)."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        non_border = [quad for quad, is_border, _ in candidates if not is_border]
        if non_border:
            return non_border[0]

        ## All candidates are border contours — fall back to the largest (first)
        return candidates[0][0]

    def _fallback_otsu_detection(self, image_original: MatLike):
        """Fallback: use Otsu threshold to find white paper on dark background."""
        gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        candidates = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximate = cv2.approxPolyDP(c, 0.04 * perimeter, closed=True)
            if len(approximate) == 4:
                candidate = approximate.reshape(4, 2)
                original_area = cv2.contourArea(c)
                if self._is_good_page_contour(candidate, original_area):
                    is_border = self._is_border_contour(candidate, binary.shape)
                    candidates.append((candidate, is_border, c))

        return self._pick_best_contour(candidates)

    def _fallback_low_contrast_detection(self, image_original: MatLike):
        """Fallback for low-contrast scenes (e.g. cream paper on gray desk).
        Uses bilateral filter (edge-preserving) + boosted CLAHE + lower Canny thresholds."""
        _, enhanced_edges = self._regularize_image(
            image_original,
            canny_thresholds=(25, 75),
            gaussian_blur_kernel_size=None,
            additional_pre_canny_step=lambda img: cv2.bilateralFilter(img, 9, 75, 75),
            clahe_clip_limit=2.0
        )

        contours, _ = cv2.findContours(enhanced_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        candidates = []
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approximate = cv2.approxPolyDP(c, 0.04 * perimeter, closed=True)
            if len(approximate) == 4:
                candidate = approximate.reshape(4, 2)
                original_area = cv2.contourArea(c)
                if self._is_good_page_contour(candidate, original_area):
                    is_border = self._is_border_contour(candidate, enhanced_edges.shape)
                    candidates.append((candidate, is_border, c))

        return self._pick_best_contour(candidates)

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
    import sys
    FILENAME = sys.argv[1] if len(sys.argv) > 1 else "properTest1.jpeg"
    GET_INPUT = lambda x : f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x : f"./TEMP/output/{x}"
    

    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]
    
    DOCUMENT_SCANNER = DocumentScanner()
    
    image_before = DOCUMENT_SCANNER.load_image(GET_INPUT(FILENAME))
    image_after = DOCUMENT_SCANNER.scan_page(image_before, debug=True)

    DOCUMENT_SCANNER.save_image(
                    image_after,
                    GET_OUTPUT(f"{_onlyfilename}/_00_SCANNED.jpg")
                    )