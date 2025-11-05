import cv2
import numpy as np
import os

def flatten_document(image_path, save_debug=False):
    # Get the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the debug output folder relative to this script
    output_dir = os.path.normpath(os.path.join(script_dir, "..", "output"))

    # Ensure the directory exists
    if save_debug and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "gray.jpg"), gray)
        cv2.imwrite(os.path.join(output_dir, "canny.jpg"), edges)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, i = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) > 1000] # Filter contour to ignore random small shapes

    if save_debug:
        all_contours_img = image.copy()
        cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "all_contours.jpg"), all_contours_img)
    
    if save_debug:
        for i, contour in enumerate(contours[:5]):
            debug_img = image.copy()
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, f"all_counter_{i+1}_contour.jpg"), debug_img)

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
        contour_img = image.copy()
        cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "contour.jpg"), contour_img)

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

    # Transformation matrix + warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    if save_debug:
        cv2.imwrite(os.path.join(output_dir, "flattened.jpg"), warped)

    return warped


# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the correct path to dataset
image_path = os.path.join(script_dir, "..", "dataset", "contour_5.jpg")
# Normalize to absolute path
image_path = os.path.normpath(image_path)

flattened_doc = flatten_document(image_path,save_debug=True)
if flattened_doc is not None:
    cv2.imshow("Flattened Document", flattened_doc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()