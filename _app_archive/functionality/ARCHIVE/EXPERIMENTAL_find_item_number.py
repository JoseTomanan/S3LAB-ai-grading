"""
This belonged to ai_interface.py previously, under AIAnswerEvaluator; I moved it out because it's mostly CV stuff and does not belong there.
"""

    # # OCR Item Number Label - Hamdi
    # # TODO: segregate; this isn't very SOC of us
    # def get_item_number_ocr(self, image_bytes: bytes) -> str:
    #     """
    #     Extract item number using traditional OpenCV.
    #     Skips PaddleOCR to avoid oneDNN compatibility issues on Windows.
    #     """
    #     try:
    #         # Load image
    #         nparr = np.frombuffer(image_bytes, np.uint8)
    #         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
    #         if image is None:
    #             logger.warning("❌ Failed to decode image for item number detection")
    #             return "UNKNOWN"
            
    #         h, w = image.shape[:2]
    #         logger.info(f"✓ Image loaded: {w}x{h}px")
            
    #         # Crop top-left corner (where encircled number typically is)
    #         # ADJUSTED: Larger crop area to ensure number is captured
    #         crop_h = int(h * 0.40)  # Was 0.30
    #         crop_w = int(w * 0.40)  # Was 0.35
    #         cropped = image[0:crop_h, 0:crop_w]
            
    #         # DEBUG: Save cropped region for inspection
    #         cv2.imwrite("debug_crop.jpg", cropped)
    #         logger.info(f"✓ Saved debug_crop.jpg (crop: {crop_w}x{crop_h})")
            
    #         # Convert to grayscale
    #         gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            
    #         # Apply Gaussian blur
    #         blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
    #         # Threshold using Otsu's method (automatic threshold)
    #         _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
    #         # DEBUG: Save thresholded image
    #         cv2.imwrite("debug_thresh.jpg", thresh)
            
    #         # Morphological operations to connect broken strokes
    #         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #         dilated = cv2.dilate(thresh, kernel, iterations=2)
    #         eroded = cv2.erode(dilated, kernel, iterations=1)
            
    #         # DEBUG: Save morphological result
    #         cv2.imwrite("debug_morph.jpg", eroded)
            
    #         # Find contours
    #         contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         logger.info(f"✓ Found {len(contours)} total contours")
            
    #         # Filter contours - MORE PERMISSIVE THRESHOLDS
    #         candidates = []
    #         for i, contour in enumerate(contours):
    #             area = cv2.contourArea(contour)
    #             x, y, cw, ch = cv2.boundingRect(contour)
                
    #             # Log ALL contours for debugging
    #             if area > 50:  # Log anything visible
    #                 aspect = cw / ch if ch > 0 else 0
    #                 logger.debug(f"  Contour #{i}: area={area:.0f}, aspect={aspect:.2f}, bbox=({x},{y},{cw},{ch})")
                
    #             # ⚠️ MORE PERMISSIVE FILTERS (adjust based on your images)
    #             if 100 < area < 60000:  # Was 300 < area < 40000
    #                 aspect_ratio = cw / ch if ch > 0 else 0
    #                 if 0.2 < aspect_ratio < 2.0:  # Was 0.3 < aspect < 1.5
    #                     candidates.append((x, y, cw, ch, contour, area))
            
    #         logger.info(f"✓ Found {len(candidates)} candidate digit contours (passed filters)")
            
    #         # DEBUG: Save all contours visualization
    #         if not candidates:
    #             debug_img = cropped.copy()
    #             cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
    #             cv2.imwrite("debug_all_contours.jpg", debug_img)
    #             logger.warning("⚠ No candidate contours passed filters - see debug_all_contours.jpg")
            
    #         if not candidates:
    #             return "UNKNOWN"
            
    #         # Sort by area (largest first - likely the encircled number)
    #         candidates.sort(key=lambda k: k[5], reverse=True)
    #         x, y, cw, ch, contour, area = candidates[0]
            
    #         logger.info(f"✓ Top candidate: area={area:.0f}, bbox=({x},{y},{cw},{ch})")
            
    #         # Extract ROI
    #         digit_roi = gray[y:y+ch, x:x+cw]
    #         cv2.imwrite("debug_digit_roi.jpg", digit_roi)
            
    #         # Count holes (for classification)
    #         roi_thresh = thresh[y:y+ch, x:x+cw]
    #         contour_hierarchy = cv2.findContours(
    #             roi_thresh,
    #             cv2.RETR_TREE,
    #             cv2.CHAIN_APPROX_SIMPLE
    #         )
            
    #         hole_count = 0
    #         if len(contour_hierarchy) > 1 and contour_hierarchy[1] is not None:
    #             hierarchy = contour_hierarchy[1][0]
    #             for i, h in enumerate(hierarchy):
    #                 if h[3] != -1:
    #                     hole_count += 1
            
    #         logger.info(f"✓ Hole count: {hole_count}, Aspect ratio: {cw/ch:.2f}")
            
    #         # Classification
    #         aspect = cw / ch if ch > 0 else 0
            
    #         if aspect < 0.4:
    #             result = "1"
    #         elif hole_count >= 2:
    #             result = "8"
    #         elif hole_count == 1:
    #             result = "0" if aspect > 0.9 else "6"
    #         else:
    #             result = "2"  # Default fallback
            
    #         logger.info(f"✓ Classified as: '{result}'")
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"✗ CV extraction failed: {e}", exc_info=True)
    #         return "UNKNOWN"