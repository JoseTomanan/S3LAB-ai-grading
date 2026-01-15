import cv2
import numpy as np
import math


def mapp(h):
	"""
	TEMPORARY SEGREGATION!!! TO BE FIXED!!!
	"""
	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]

	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew


def get_robust_aspect_ratio(coords):
	"""
	Calculates the ratio using the 'Minimum Area' logic.
	Works better for trapezoids and skewed shapes.
	"""
	pts = np.array(coords)

	# 1. Find the center of the shape
	center = np.mean(pts, axis=0)

	# 2. Sort points to ensure they are in order (Clockwise)
	angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
	pts = pts[np.argsort(angles)]

	# 3. Calculate side vectors
	# Side A (Top/Bottom) and Side B (Left/Right)
	side_a1 = pts[1] - pts[0]
	side_a2 = pts[3] - pts[2]
	side_b1 = pts[2] - pts[1]
	side_b2 = pts[0] - pts[3]

	# 4. Use the magnitude of the vectors
	# This effectively treats the shape as a rectangle 
	# based on its primary spans.
	w = (np.linalg.norm(side_a1) + np.linalg.norm(side_a2)) / 2
	h = (np.linalg.norm(side_b1) + np.linalg.norm(side_b2)) / 2

	return max(w,h) / min(w,h)


def find_first_box(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
	"""
	TEMPORARY SEGREGATION!!! TO BE FIXED!!!
	"""
	image_h, image_w, _ = image.shape
	image_ratio = image_w/image_h
	print("IMAGE RATIO:", 800*image_ratio)

	# Temporary; resizing because opencv does not work well with bigger images
	image = cv2.resize(image, (int(800*(image_ratio**2)), 800))
	orig = image.copy()
	cv2.imshow("Elongated original image", image)

	# STEP 1 : MAKE GRAYSCALE (TEMPORARILY)
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# STEP 2: CREATE HORIZONTAL KERNEL (for morphological operations)
	horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))

	# STEP 3: DETECT HORIZONTAL LINES
	detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
	cv2.imshow("Detected lines", detected_lines)

	# STEP 4: SUBTRACT LINES FROM ORIGINAL
	gray_no_lines = cv2.subtract(gray, detected_lines)
	cv2.imshow("Without Lines", gray_no_lines)
	
	# STEP 4.1: ADAPTIVE THRESHOLDING (Not used)
	# TODO: FIX PARAMETER VALUES
	thresh = cv2.adaptiveThreshold(gray_no_lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY, 3, 2)
	cv2.imshow("Adaptive threshold", thresh)

	# STEP 5: APPLY GAUSSIAN BLUR
	blurred = cv2.GaussianBlur(gray_no_lines, (5,5), 0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
	cv2.imshow("Blur",blurred)

	# STEP 6: SEGMENT CANNY LINES
	edged = cv2.Canny(blurred, 30, 50)  #30 MinThreshold and 50 is the MaxThreshold
	cv2.imshow("Canny",edged)


	# STEP 7: CONTOURING
	contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			#retrieve the contours as a list, with simple apprximation model
	contours = sorted(contours,key=cv2.contourArea,reverse=True)

	for c in contours:	#the loop extracts the boundary contours of the page
		p = cv2.arcLength(c,True)
		approx = cv2.approxPolyDP(c,0.02*p,True)
		if len(approx) == 4:
			target=approx
			break

	approx = mapp(target)
	box_ratio = math.sqrt(get_robust_aspect_ratio(approx))
	box_length = int(800*math.sqrt(box_ratio))

	print("BOX RATIO:", box_ratio)

	pts = np.float32(np.array([[0,0], [box_length,0], [box_length,800], [0,800]]))

	# STEP 8: PERSPECTIVE TRANSFORM
	op = cv2.getPerspectiveTransform(approx, pts) #pyright: ignore
	dst = cv2.warpPerspective(orig, op, (int(800*box_ratio), 800))

	# STEP 9: SHOW FINAL RESULT!
	cv2.destroyAllWindows()
	cv2.imshow("Detected Segmented Image", dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return dst


if __name__ == "__main__":
	image = cv2.imread("dataset/contour_15.jpg")
	assert image is not None
	find_first_box(image)