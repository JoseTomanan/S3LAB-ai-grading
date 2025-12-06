import cv2
import numpy as np
import os

def flatten_document(image_path, save_debug=False):
    # Step 1 – Load and Preprocess Image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "output"))
    if save_debug and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
   
    # Step 2 – Apply Gaussian Blur
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3 – Detect Edges with Canny
    edges = cv2.Canny(blurred, 50, 150)
    if save_debug:
        # Step 4 – Save Grayscale Image
        cv2.imwrite(os.path.join(output_dir, "4_gray.jpg"), gray)
        # Step 5 – Save Canny Edges
        cv2.imwrite(os.path.join(output_dir, "5_canny.jpg"), edges)
   
    # Step 6 – Dilate Edges
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 7 – Apply Morphological Close
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Step 8 – Find Contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000] # Filter contour to ignore random small shapes
    if save_debug:
        # Step 9 – Save All Contours
        all_contours_img = image.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "9_all_contours.jpg"), all_contours_img)
   
    if save_debug:
        # Step 10 – Save Top 5 Individual Contours
        for i, contour in enumerate(contours[:5]):
            debug_img = image.copy()
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"10_individual_contour_{i+1}.jpg"), debug_img)
   
    # Step 11 – Detect Document Contour (4 Corners)
    doc_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) == 4:
            doc_contour = approx
            break
    if doc_contour is None:
        raise Exception("Could not find a 4-cornered contour (document boundary).")
    if save_debug:
        # Step 12 – Save Detected Document Contour
        contour_img = image.copy()
        cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "12_doc_contour.jpg"), contour_img)
   
    # Step 13 – Order Corner Points
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    
    # Step 14 – Calculate Output Dimensions
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
    
    # Step 15 – Compute Perspective Transformation
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # Step 16 – Warp Perspective
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    if save_debug:
        # Step 17 – Save Flattened Document
        cv2.imwrite(os.path.join(output_dir, "17_flattened.jpg"), warped)
    print("Document flattening completed.")
    return warped

class CVImagePreprocessor:
    def __init__(self):
        pass

    def load_image(self, image_path: str) -> bytes:
        """
        Load image (unencoded) and return as bytes
        """
        # Step 1 – Load Image into Bytes
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
     
        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            raise ValueError("Failed to encode image")
        return buffer.tobytes()

    def detect_paper_type(self, image_bytes: bytes, line_threshold: int = 5, min_line_length: int = 100) -> str:
        """
        Detect if image is lined (Group L) or blank (Group B) paper.
        - line_threshold: Min number of horizontal lines to classify as lined.
        - Returns: 'lined' or 'blank'.
        """
        # Step 2 – Decode Image for Paper Type Detection
        image = self._decode_bytes(image_bytes)
        
        # Step 3 – Convert to Grayscale for Detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 4 – Edge Detection for Lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
      
        # Step 5 – Detect Lines with Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=min_line_length, maxLineGap=10)
      
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if mostly horizontal (small angle)
                if abs(y2 - y1) < 10: # Vertical tolerance for "horizontal"
                    horizontal_lines += 1
      
        paper_type = 'lined' if horizontal_lines >= line_threshold else 'blank'
        print(f"Detected {horizontal_lines} horizontal lines; Paper type: {paper_type}")
        return paper_type

    def remove_horizontal_lines(self, image_bytes: bytes, kernel_size: int = 3) -> bytes:
        """
        Remove horizontal lines from lined paper using morphology.
        - kernel_size: Width of horizontal structuring element.
        """
        # Step 6 – Decode Image for Line Removal
        image = self._decode_bytes(image_bytes)
        
        # Step 7 – Convert to Grayscale for Line Extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
        # Step 8 – Create Horizontal Structuring Element
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size * 20, 1)) # Wide for horizontals
        
        # Step 9 – Extract Horizontal Lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel) # Extract lines
        
        # Step 10 – Create Vertical Structuring Element
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size * 20)) # For any vertical noise
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
      
        # Step 11 – Combine Line Masks
        lines_mask = cv2.addWeighted(horizontal_lines, 1, vertical_lines, 1, 0)
        
        # Step 12 – Dilate Line Mask
        lines_mask = cv2.dilate(lines_mask, horizontal_kernel, iterations=1) # Thicken mask
      
        # Step 13 – Inpaint to Remove Lines
        result = cv2.inpaint(image, lines_mask, 1, cv2.INPAINT_TELEA) # Telea algorithm for smooth fill
      
        # Step 14 – Encode Processed Image
        return self._encode_to_bytes(result)

    def deskew_for_blank(self, image_bytes: bytes, angle_threshold: float = 10) -> bytes:
        """
        Deskew (straighten) for blank paper to handle uneven layouts.
        """
        # Step 15 – Decode Image for Deskewing
        image = self._decode_bytes(image_bytes)
        
        # Step 16 – Convert to Grayscale for Skew Detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
        # Step 17 – Edge Detection for Skew Lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Step 18 – Detect Lines for Skew Angle
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
      
        if lines is not None:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if 0 < abs(angle) < angle_threshold: # Small angles only
                    angles.append(angle)
          
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5: # Worth correcting
                    print(f"Step: Detected skew angle {median_angle:.2f} degrees, correcting")
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    # Step 19 – Encode Deskewed Image
                    return self._encode_to_bytes(deskewed)
      
        # Step 20 – No Significant Skew Detected, Return Original
        return image_bytes # No skew detected

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
        # Step 21 – Save Processed Image
        with open(save_path, "wb") as f:
            ret = f.write(image_bytes)
        if not ret:
            raise ValueError(f"Failed to save image to {save_path}")
        print(f"Image saved to {save_path}")

# Test the integrated functionalities
if __name__ == "__main__":
    # Step 22 – Set Up Paths for Testing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.normpath(os.path.join(script_dir, "..", "dataset", "contour_11.jpg"))
  
    print("Starting document flattening process...")
    # Step 23 – Flatten Document
    flattened_doc = flatten_document(image_path, save_debug=True)
  
    print("Starting paper type detection and preprocessing...")
    # Step 24 – Initialize Preprocessor and Load Flattened Image
    preprocessor = CVImagePreprocessor()
    flattened_path = os.path.join(script_dir, "..", "output", "17_flattened.jpg") # Adjusted to match save name
    if os.path.exists(flattened_path):
        flattened_bytes = preprocessor.load_image(flattened_path)
    else:
        flattened_bytes = preprocessor.load_image(image_path) # Fallback
  
    # Step 25 – Detect Paper Type
    paper_type = preprocessor.detect_paper_type(flattened_bytes)
    
    # Step 26 – Apply Conditional Preprocessing and Save
    if paper_type == 'lined':
        processed_bytes = preprocessor.remove_horizontal_lines(flattened_bytes)
        print("Applied line removal for lined paper.")
        save_path = os.path.join(script_dir, "..", "output", "processed_lined_removed.jpg")
    else:
        processed_bytes = preprocessor.deskew_for_blank(flattened_bytes)
        print("Applied deskew for blank paper.")
        save_path = os.path.join(script_dir, "..", "output", "processed_blank_deskewed.jpg")
  
    preprocessor.save_image(processed_bytes, save_path)
  
    # Step 27 – Display Final Processed Image
    processed_image = preprocessor._decode_bytes(processed_bytes)
    cv2.imshow("Processed Document (Paper Type Handling)", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Processing completed.")