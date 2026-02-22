import numpy as np
import cv2

from .DocumentScanner import DocumentScanner



# ================================
#region Class
# ================================
class BoxSegmenter:
    def __init__(self):
        self._document_scanner = DocumentScanner()


    def get_boxes(self, 
                    image_bytes: bytes,
                    num_boxes: int
                    ) -> list[bytes]:
        """Get best boxes (non-overlapping) from the image given. Note that image is expected to have been scanned already."""
        ...


    #region Imported functions
    def load_image(self, image_path: str) -> bytes:
        return self._document_scanner.load_image(image_path)
    
    def save_image(self, image_bytes: bytes, save_path: str) -> None:
        self._document_scanner.save_image(image_bytes, save_path)

    def _decode_bytes(self, image_bytes: bytes) -> cv2.typing.MatLike:
        return self._document_scanner._decode_bytes(image_bytes)

    def _encode_to_bytes(self, image_matrix: cv2.typing.MatLike) -> bytes:
        return self._document_scanner._encode_to_bytes(image_matrix)

    def _warp_from_original(self, screen_contour: cv2.typing.MatLike, original: cv2.typing.MatLike) -> cv2.typing.MatLike:
        return self._document_scanner._warp_from_original(screen_contour, original)
    #endregion
#endregion


if __name__ == "__main__":
    ...