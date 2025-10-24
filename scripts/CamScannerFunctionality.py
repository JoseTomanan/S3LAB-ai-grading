import cv2
import numpy as np
import os

def flatten_document(image_path, save_debug=False):
    # -Load and preprocess
    image = cv2.imread(image_path)
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Save intermediate images
    # if save_debug:
    #     cv2.imwrite("../output/gray.jpg", gray)
    #     cv2.imwrite("../output/canny.jpg", edges)

    # Find contours
    contours, i = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        raise Exception("Could not find a 4-cornered contour (document boundary).")

    if save_debug:
        contour_img = image.copy()
        cv2.drawContours(contour_img, [doc_contour], -1, (0, 255, 0), 2)
        cv2.imwrite("../output/contour.jpg", contour_img)

    # Order the corner points
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    # Compute perspective transform
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
        cv2.imwrite("../output/flattened.jpg", warped)

    return warped

# Get the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the correct path to dataset
image_path = os.path.join(script_dir, "..", "dataset", "contour_3.jpg")
# Normalize to absolute path
image_path = os.path.normpath(image_path)

flattened_doc = flatten_document(image_path,save_debug=True)
if flattened_doc is not None:
    cv2.imshow("Flattened Document", flattened_doc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()