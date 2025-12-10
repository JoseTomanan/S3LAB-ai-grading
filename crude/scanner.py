import cv2
import numpy as np



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

def find_first_box(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
	"""
	TEMPORARY SEGREGATION!!! TO BE FIXED!!!
	"""
	# Temporary; resizing because opencv does not work well with bigger images
	image=cv2.resize(image,(1300,800)) 
	orig=image.copy()

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
	
	# STEP 4.1: ADAPTIVE THRESHOLDING
	thresh = cv2.adaptiveThreshold(gray_no_lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
								cv2.THRESH_BINARY, 3, 2)
	cv2.imshow("Adaptive threshold", thresh)

	# STEP 5: APPLY GAUSSIAN BLUR
	blurred = cv2.GaussianBlur(gray_no_lines, (5,5), 0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
	cv2.imshow("Blur",blurred)

	# STEP 6: SEGMENT CANNY LINES
	edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
	cv2.imshow("Canny",edged)


	# STEP 7: CONTOURING
	contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
	contours=sorted(contours,key=cv2.contourArea,reverse=True)

	for c in contours:	#the loop extracts the boundary contours of the page
		p=cv2.arcLength(c,True)
		approx=cv2.approxPolyDP(c,0.02*p,True)
		if len(approx)==4:
			target=approx
			break

	approx=mapp(target)

	pts=np.float32(np.array([[0,0],[800,0],[800,800],[0,800]]))  #map to 800*800 target window

	# STEP 8: PERSPECTIVE TRANSFORM
	op=cv2.getPerspectiveTransform(approx,pts) #pyright: ignore
	dst=cv2.warpPerspective(orig,op,(800,800))

	# STEP 9: SHOW FINAL RESULT!
	cv2.imshow("FINAL SCANNED",dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return dst
