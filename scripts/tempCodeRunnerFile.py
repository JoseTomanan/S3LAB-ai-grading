import cv2
import numpy as np
import os


def flatten_document(image_path, save_debug=False):
    step_counter = 1  # Global step counter for this function
    # Get the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the debug output folder relative to this script
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "output"))
    # Ensure the directory exists
    if save_debug and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Load and preprocess
    print(f"Step {step_counter}: Loading original image from {image_path}")
    image = cv2.imread(image_path)
    orig = image.copy()
    step_counter += 1
    print(f"Step {step_counter}: Converting to grayscale")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if save_debug:
        gray_path = os.path.join(output_dir, f"{step_counter}_gray.jpg")
        cv2.imwrite(gray_path, gray)
        print(f"Intermediate output produced by Step {step_counter}: {gray_path}")
    step_counter += 1
    print(f"Step {step_counter}: Applying Gaussian blur")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    if save_debug:
        blurred_path = os.path.join(output_dir, f"{step_counter}_blurred.jpg")
        cv2.imwrite(blurred_path, blurred)
        print(f"Intermediate output produced by Step {step_counter}: {blurred_path}")
    step_counter += 1
    print(f"Step {step_counter}: Detecting edges with Canny")
    edges = cv2.Canny(blurred, 50, 150)
    if save_debug:
        edges_path = os.path.join(output_dir, f"{step_counter}_canny.jpg")
        cv2.imwrite(edges_path, edges)
        print(f"Intermediate output produced by Step {step_counter}: {edges_path}")
    step_counter += 1
    print(f"Step {step_counter}: Dilating edges")
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    if save_debug:
        dilated_path = os.path.join(output_dir, f"{step_counter}_dilated.jpg")
        cv2.imwrite(dilated_path, edges)
        print(f"Intermediate output produced by Step {step_counter}: {dilated_path}")
    step_counter += 1
    print(f"Step {step_counter}: Applying morphological close operation")
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if save_debug:
        closed_path = os.path.join(output_dir, f"{step_counter}_closed.jpg")
        cv2.imwrite(closed_path, edges)
        print(f"Intermediate output produced by Step {step_counter}: {closed_path}")
    step_counter += 1
    print(f"Step {step_counter}: Finding contours")
    contours, i = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    step_counter += 1
    print(f"Step {step_counter}: Sorting and filtering contours by area (>1000)")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000] # Filter contour to ignore random small shapes
    if save_debug:
        all_contours_img = image.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
        contours_path = os.path.join(output_dir, f"{step_counter}_all_contours.jpg")
        cv2.imwrite(contours_path, all_contours_img)
        print(f"Intermediate output produced by Step {step_counter}: {contours_path}")
   
    if save_debug:
        for i, contour in enumerate(contours[:5]):
            debug_img = image.copy()
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            contour_debug_path = os.path.join(output_dir, f"{step_counter}_contour_{i+1}.jpg")
            cv2.imwrite(contour_debug_path, debug_img)
            print(f"Intermediate output produced by Step {step_counter} (sub): {contour_debug_path}")
    step_counter += 1
    print(f"Step {step_counter}: Identifying document contour (4-sided polygon)")
    doc_contour = None
    selected_raw = None  # Track raw for debug
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)  # Tuned epsilon for crisper approx
        if len(approx) == 4:
            doc_contour = approx
            selected_raw = contour  # Capture raw here
            break
    if doc_contour is None:
        raise Exception("Could not find a 4-cornered contour (document boundary).")
    if save_debug:
        contour_img = image.copy()
        cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 2)
        doc_contour_path = os.path.join(output_dir, f"{step_counter}_doc_contour.jpg")
        cv2.imwrite(doc_contour_path, contour_img)
        print(f"Intermediate output produced by Step {step_counter}: {doc_contour_path}")
        
        # Fixed debug overlay: Raw vs. approx
        if selected_raw is not None:
            debug_img = image.copy()
            cv2.drawContours(debug_img, [selected_raw], -1, (0, 255, 0), 2)  # Raw in green
            cv2.drawContours(debug_img, [doc_contour], -1, (255, 0, 0), 3)  # Approx in blue, thicker
            raw_vs_approx_path = os.path.join(output_dir, f"{step_counter}_raw_vs_approx.jpg")
            cv2.imwrite(raw_vs_approx_path, debug_img)
            print(f"Debug overlay produced by Step {step_counter}: {raw_vs_approx_path}")
    step_counter += 1
    print(f"Step {step_counter}: Ordering corner points and computing destination rectangle")
    # Order and warp
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    if save_debug:
        # Save ordered points visualization (simple text overlay or just log)
        ordered_path = os.path.join(output_dir, f"{step_counter}_ordered_pts.jpg")
        cv2.imwrite(ordered_path, image)  # Placeholder; could draw pts if needed
        print(f"Intermediate output produced by Step {step_counter}: {ordered_path}")
    step_counter += 1
    print(f"Step {step_counter}: Computing perspective transformation and warping")
    # Transformation matrix + warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    if save_debug:
        flattened_path = os.path.join(output_dir, f"{step_counter}_flattened.jpg")
        cv2.imwrite(flattened_path, warped)
        print(f"Intermediate output produced by Step {step_counter}: {flattened_path}")
    print(f"Flatten_document completed after {step_counter} steps.")
    return warped


class CVImagePreprocessor:
    def __init__(self):
        pass

    def load_image(self, image_path: str) -> bytes:
        """
        Load image (unencoded) and return as bytes
        """
        print(f"Step (Load): Loading image from {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
       
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            raise ValueError("Failed to encode image")
        print("Step (Load) completed: Image loaded and encoded to bytes.")
        return buffer.tobytes()

    def detect_paper_type(self, image_bytes: bytes, line_threshold: int = 5, min_line_length: int = 100) -> str:
        """
        Detect if image is lined (Group L) or blank (Group B) paper.
        - line_threshold: Min number of horizontal lines to classify as lined.
        - Returns: 'lined' or 'blank'.
        """
        print("Step (Detect): Starting paper type detection")
        image = self._decode_bytes(image_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("Step (Detect sub1): Edge detection with Canny")
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using probabilistic Hough Transform
        print("Step (Detect sub2): HoughLinesP for line detection")
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            print(f"Step (Detect sub3): Counting horizontal lines ({len(lines)} total lines detected)")
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if mostly horizontal (small angle)
                if abs(y2 - y1) < 10:  # Vertical tolerance for "horizontal"
                    horizontal_lines += 1
        
        paper_type = 'lined' if horizontal_lines >= line_threshold else 'blank'
        print(f"Step (Detect) completed: Detected {horizontal_lines} horizontal lines → Paper type: {paper_type}")
        return paper_type

    def remove_horizontal_lines(self, image_bytes: bytes, kernel_size: int = 3) -> bytes:
        """
        Remove horizontal lines from lined paper using morphology.
        - kernel_size: Width of horizontal structuring element.
        """
        print("Step (Remove Lines): Starting horizontal line removal")
        image = self._decode_bytes(image_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Horizontal kernel for line detection
        print("Step (Remove Lines sub1): Creating horizontal kernel and extracting lines")
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size * 20, 1))  # Wide for horizontals
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)  # Extract lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size * 20))  # For any vertical noise
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine and mask
        print("Step (Remove Lines sub2): Combining line masks and dilating")
        lines_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
        lines_mask = cv2.dilate(lines_mask, horizontal_kernel, iterations=1)  # Thicken mask
        
        # Inpaint: Replace lines with surrounding pixels (seamless clone for better results)
        print("Step (Remove Lines sub3): Applying inpaint")
        mask_inv = cv2.bitwise_not(lines_mask)
        result = cv2.inpaint(image, lines_mask, 1, cv2.INPAINT_TELEA)  # Telea algorithm for smooth fill
        
        print("Step (Remove Lines) completed: Lines removed.")
        return self._encode_to_bytes(result)

    def deskew_for_blank(self, image_bytes: bytes, angle_threshold: float = 10) -> bytes:
        """
        Deskew (straighten) for blank paper to handle uneven layouts.
        """
        print("Step (Deskew): Starting deskew for blank paper")
        image = self._decode_bytes(image_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection and Hough for skew angle
        print("Step (Deskew sub1): Edge detection and Hough lines")
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if 0 < abs(angle) < angle_threshold:  # Small angles only
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:  # Worth correcting
                    print(f"Step (Deskew sub2): Applying rotation by median angle {median_angle:.2f}°")
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    print("Step (Deskew) completed: Deskew applied.")
                    return self._encode_to_bytes(deskewed)
                else:
                    print("Step (Deskew sub2): Skew angle too small; no correction needed.")
            else:
                print("Step (Deskew sub2): No suitable angles found; no correction needed.")
        else:
            print("Step (Deskew sub2): No lines detected; no correction needed.")
        
        print("Step (Deskew) completed: No changes applied.")
        return image_bytes  # No skew detected

    def _decode_bytes(self, image_bytes: bytes) -> cv2.typing.MatLike:
        """
        Decode bytes into BGR uint8 array
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes")
       
        return image
   
    def _encode_to_bytes(self, image_matrix: cv2.typing.MatLike) -> bytes:
        """
        Encode BGR uint8 array back to JPEG bytes
        """
        ret, buffer = cv2.imencode('.jpg', image_matrix)
        if not ret:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    def save_image(self, image_bytes: bytes, save_path: str) -> None:
        """
        Save the processed image (in JPEG format) to the specified path.
        """
        with open(save_path, "wb") as f:
            ret = f.write(image_bytes)
        if not ret:
            raise ValueError(f"Failed to save image to {save_path}")
        print(f"Final output saved: {save_path}")


# Test the integrated functionalities
if __name__ == "__main__":
    global_step = 0  # Reset for main block
    # Get the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the correct path to dataset
    image_path = os.path.join(script_dir, "..", "dataset", "contour_9.jpg")  # Updated to match your run
    # Normalize to absolute path
    image_path = os.path.normpath(image_path)
    
    global_step += 1
    print(f"Main Step {global_step}: Flattening document")
    flattened_doc = flatten_document(image_path, save_debug=True)
    
    # Now apply paper type detection and conditional preprocessing
    global_step += 1
    print(f"Main Step {global_step}: Initializing preprocessor and loading flattened image")
    preprocessor = CVImagePreprocessor()
    flattened_path = os.path.join(script_dir, "..", "output", "10_flattened.jpg")  # Matches last step in flatten
    if os.path.exists(flattened_path):
        flattened_bytes = preprocessor.load_image(flattened_path)
    else:
        print("Warning: Flattened path not found; skipping paper type steps.")
        exit(1)
    
    global_step += 1
    print(f"Main Step {global_step}: Detecting paper type")
    paper_type = preprocessor.detect_paper_type(flattened_bytes)
    if paper_type == 'lined':
        global_step += 1
        print(f"Main Step {global_step}: Applying line removal")
        processed_bytes = preprocessor.remove_horizontal_lines(flattened_bytes)
        save_path = os.path.join(script_dir, "..", "output", f"{global_step}_lined_removed.jpg")
    else:
        global_step += 1
        print(f"Main Step {global_step}: Applying deskew")
        processed_bytes = preprocessor.deskew_for_blank(flattened_bytes)
        save_path = os.path.join(script_dir, "..", "output", f"{global_step}_blank_deskewed.jpg")
    
    # Save the processed image for inspection
    preprocessor.save_image(processed_bytes, save_path)
    
    global_step += 1
    print(f"Main Step {global_step}: Displaying final processed image")
    # Display the final processed image
    processed_image = preprocessor._decode_bytes(processed_bytes)
    cv2.imshow("Processed Document (Paper Type Handling)", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Main execution completed after {global_step} steps.")