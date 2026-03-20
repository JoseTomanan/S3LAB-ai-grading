import sys
import cv2
from cv2.typing import MatLike
import numpy as np
from pathlib import Path

from logic.document_scanner import DocumentScanner



class ImageModifier:
    debug_dir = "./TEMP/output/DEBUG"

    def validate_file_extension(self, filename: str) -> bool:
        """Validate that file extension is allowed"""
        if not filename or not isinstance(filename, str):
            return False
        ext = Path(filename).suffix.lower()
        return ext in [".jpeg", ".jpg", ".png"]
    
    def brighten(self, image: np.ndarray, amount: float = 0.25) -> np.ndarray:
        """Increase image brightness using linear transform"""
        amount = max(0.0, min(1.0, float(amount)))
        beta = amount * 255
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)
    
    def adjust_contrast(self, image: np.ndarray, amount: float = 1.3) -> np.ndarray:
        """Adjust image contrast with brightness compensation"""
        amount = max(0.1, float(amount))
        beta = 128 * (1 - amount)
        return cv2.convertScaleAbs(image, alpha=amount, beta=beta)

    def pseudocanny(self, image_mat: MatLike, block_size: int = 161, C: int = 13, gaussian_blur_kernel_size: tuple[int,int] | None = (5, 5)) -> MatLike:
        """Convert image to binary (white features on black background) using adaptive thresholding.
        Unlike Canny edge detection, this preserves solid fills — a filled dot stays
        a solid white blob instead of becoming a hollow ring."""
        ## Grayscale
        img_iterated = cv2.cvtColor(image_mat, cv2.COLOR_BGR2GRAY) if len(image_mat.shape) == 3 else image_mat

        ## Gaussian blur (optional)
        if gaussian_blur_kernel_size is not None:
            img_iterated = cv2.GaussianBlur(img_iterated, gaussian_blur_kernel_size, 0)

        ## Adaptive threshold (inverted: white features on black background)
        img_iterated = cv2.adaptiveThreshold(
            img_iterated, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, C
            )

        return img_iterated



if __name__ == "__main__":
    # ================ DEFINITIONS ================
    FILENAME = sys.argv[1] if len(sys.argv) > 1 else "testRuledDottedA.jpeg"
    GET_INPUT = lambda x: f"./TEMP/input/{x}"
    GET_OUTPUT = lambda x: f"./TEMP/output/{x}"


    # ================ ACTUAL TEST ================
    _onlyfilename = FILENAME.split(".")[0]

    DOCUMENT_SCANNER = DocumentScanner()
    IMAGE_MODIFIER = ImageModifier()
    IMAGE_MODIFIER.debug_dir = f"./TEMP/output/{_onlyfilename}"

    asMat = lambda x : DOCUMENT_SCANNER._decode_bytes(x)
    asBytes = lambda x : DOCUMENT_SCANNER._encode_to_bytes(x)

    image_before_before = DOCUMENT_SCANNER.load_image(GET_INPUT(FILENAME))
    image_before = DOCUMENT_SCANNER.scan_page(image_before_before)
    image_after_mat = IMAGE_MODIFIER.pseudocanny(asMat(image_before))

    DOCUMENT_SCANNER.save_image(
                        asBytes(image_after_mat),
                        GET_OUTPUT(f"{_onlyfilename}/_01B_binarized.jpg") )
