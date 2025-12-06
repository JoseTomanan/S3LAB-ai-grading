import cv2
import numpy as np
import os

def flatten_document(image_path, save_debug=False, top_n=5):
    """
    Flatten document using top N largest contours.
    - top_n: Number of top contours to process (default=3 for efficiency).
    Returns: Dict of {index: warped_image} for each processed contour.
    """
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
    
    # Step 8 – Find Contours (Full Hierarchy + Top-Level Filter)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Filter to top-level: Contours with no parent (hierarchy[0, i, 3] == -1)
    top_level_indices = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]  # h[3] = parent
    contours = [contours[i] for i in top_level_indices]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if save_debug:
        # Step 9 – Save All Contours
        all_contours_img = image.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "9_all_contours.jpg"), all_contours_img)
    
    if save_debug:
        # Step 10 – Save Top 10 Individual Contours (Extended to top_n if >10)
        top_contours = contours[:min(10, len(contours))]
        for i, contour in enumerate(top_contours):
            debug_img = image.copy()
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"10_individual_contour_{i+1}.jpg"), debug_img)
    
    # Step 11 – Force Top N Contours to Convex 4-Point Approximations
    if not contours:
        raise Exception("No contours available for document boundary detection.")

    top_contours = contours[:top_n]
    warped_results = {}  # {index: warped_image}

    for idx, largest_contour in enumerate(top_contours):
        raw_area = cv2.contourArea(largest_contour)
        print(f"Processing contour {idx} (file 10_individual_contour_{idx+1}.jpg): Raw area {raw_area:.0f} square px")

        # Substep 11a: Compute Convex Hull (Eliminates Concavities)
        hull = cv2.convexHull(largest_contour)
        hull_perimeter = cv2.arcLength(hull, True)
        print(f"  Convex hull perimeter: {hull_perimeter:.0f} px")

        # Substep 11b: Approximate Hull to Polygon (Tune Epsilon for ~4 Sides)
        epsilon = 0.03 * hull_perimeter  # Starting point
        approx = cv2.approxPolyDP(hull, epsilon, True)
        num_sides = len(approx)

        # Iteratively Increase Epsilon Until ≤4 Sides (Aggressive Simplification)
        while num_sides > 4 and epsilon < 0.2 * hull_perimeter:  # Cap at 20% to avoid over-flattening
            epsilon *= 1.5  # Progressive increase (e.g., 0.03 → 0.045 → 0.0675 → ...)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            num_sides = len(approx)

        # Substep 11c: Clamp to Exactly 4 if Needed (<4 is Rare, but Handle)
        if num_sides > 4:
            print(f"  Warning: Could not simplify to ≤4 sides (got {num_sides}); using as-is for transform.")
            # Optional: Force extrema (uncomment if you want aggressive clamping)
            # pts = hull.reshape(-1, 2)
            # approx = np.array([...])  # As before
        elif num_sides < 4:
            print(f"  Warning: Under-simplified to {num_sides} sides; clamping to 4 extrema.")
            pts = hull.reshape(-1, 2)
            approx = np.array([
                [pts[:, 0].min(), pts[:, 1].min()],  # BL
                [pts[:, 0].max(), pts[:, 1].min()],  # BR
                [pts[:, 0].max(), pts[:, 1].max()],  # TR
                [pts[:, 0].min(), pts[:, 1].max()]   # TL
            ], dtype="float32")
            num_sides = 4

        # Final Assignment
        doc_contour = approx
        print(f"  Final approx: {num_sides} convex sides, final epsilon={epsilon:.2f}")

        # NEW: Validate Non-Degenerate Contour (Fixes Draw Error)
        unique_pts = np.unique(doc_contour, axis=0)  # Remove duplicates
        approx_area = cv2.contourArea(doc_contour)
        if len(unique_pts) < 3 or approx_area < 100:  # <3 pts or tiny area = degenerate
            print(f"  Skipping contour {idx}: Degenerate approx (area {approx_area:.0f}, unique pts {len(unique_pts)})")
            continue

        # Quick Coverage Check (Now on Validated Area)
        image_area = image.shape[0] * image.shape[1]
        if approx_area < 0.1 * image_area:
            print(f"  Skipping contour {idx}: Forced approx too small ({approx_area:.0f} vs {image_area:.0f} image area); check preprocessing.")
            continue

        # NEW: Normalize Shape for Drawing (Fixes Assertion Error)
        if doc_contour.ndim == 2:  # (4,2) from clamp → Reshape to (4,1,2)
            doc_contour_draw = doc_contour.reshape(4, 1, 2).astype(np.int32)
        else:  # Already (4,1,2) from approxPolyDP
            doc_contour_draw = doc_contour.astype(np.int32)  # Ensure int32
        # Re-check area post-reshape (catches float→int rounding degenerates)
        approx_area_draw = cv2.contourArea(doc_contour_draw)
        if approx_area_draw < 100:
            print(f"  Skipping contour {idx}: Degenerate after reshape (area {approx_area_draw:.0f})")
            continue

        if save_debug:
            # Enhanced Debug: Draw Raw, Hull, and Approx for this contour
            debug_img = image.copy()
            cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 3)  # Red: Raw
            cv2.drawContours(debug_img, [hull], -1, (0, 255, 255), 2)           # Yellow: Hull
            cv2.drawContours(debug_img, [doc_contour_draw], -1, (0, 0, 255), 4)  # Blue: Final approx (Safe now)
            cv2.imwrite(os.path.join(output_dir, f"11_forced_approx_{idx+1}.jpg"), debug_img)

        # Step 13 – Order Corner Points (Use original doc_contour (4,2) float32)
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
        warped_results[idx] = warped  # Store by 0-based index

        if save_debug:
            # Step 17 – Save Flattened Document for this contour
            cv2.imwrite(os.path.join(output_dir, f"17_flattened_{idx+1}.jpg"), warped)

    if not warped_results:
        raise Exception("No valid contours produced warped images.")

    print(f"Document flattening completed for top {len(warped_results)} contours.")
    return warped_results  # Dict {0: warped0, 1: warped1, ...}





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
    image_path = os.path.normpath(os.path.join(script_dir, "..", "dataset", "contour_14.jpg"))
  
    print("Starting document flattening process...")
    # Step 23 – Flatten Document (Now processes top_n=3)
    flattened_results = flatten_document(image_path, save_debug=True, top_n=5)
    
    # For backward compatibility: Use the first (largest) for preprocessor
    flattened_doc = flattened_results[0]  # Index 0: Largest
    flattened_path = os.path.join(script_dir, "..", "output", "17_flattened_1.jpg")  # Use _1.jpg
    
    print("Starting paper type detection and preprocessing...")
    # Step 24 – Initialize Preprocessor and Load Flattened Image
    preprocessor = CVImagePreprocessor()
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