"""
This belonged to DocumentScanner.py, but moved out because of its experimetal and unworking nature. Can be brought back in desperate times.
"""

#     def scan_page_using_llm(self, image_bytes: bytes) -> bytes:
#         """Scan image by sourcing corner coordinates from LLM."""
#         image = self._decode_bytes(image_bytes)
#         image_original, _ = self._regularize_image(image)

#         response = AIAnswerEvaluator().find_four_points(
#                     self._encode_to_bytes(image_original)
#                     )
#         assert response is not None
#         llm_coordinates = np.array(response, dtype=np.float32).reshape((4, 1, 2)).astype(np.int32)
        
#         image_warped = self._warp_from_original(llm_coordinates, image_original)
#         return self._encode_to_bytes(image_warped)