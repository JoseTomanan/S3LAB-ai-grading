import cv2
import numpy as np
import mapper


IS_LINED = False


if __name__ == "__main__":
    image = cv2.imread("ignoreable/white4croppedB.jpg")	# read in the image
    assert image is not None

    image=cv2.resize(image,(800,1300)) 	# resizing because opencv does not work well with bigger images
    orig=image.copy()

	# STEP 1 : MAKE GRAYSCALE (TEMPORARILY)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    if IS_LINED:
        # STEP 2: CREATE HORIZONTAL KERNEL (for morphological operations)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))

        # STEP 3: DETECT HORIZONTAL LINES
        detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        cv2.imshow("Detected lines", detected_lines)

        # STEP 4: SUBTRACT LINES FROM ORIGINAL
        gray_no_lines = cv2.subtract(gray, detected_lines)
        cv2.imshow("Without Lines", gray_no_lines)
        
        # STEP 4.1: ADAPTIVE THRESHOLDING
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Adaptive threshold", thresh)

    # STEP 5: APPLY GAUSSIAN BLUR
    blurred = cv2.GaussianBlur(thresh if IS_LINED else gray, (5,5), 0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
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
    approx=mapper.mapp(target) #find endpoints of the sheet

    pts=np.float32(np.array([[0,0],[800,0],[800,800],[0,800]]))  #map to 800*800 target window

	# STEP 8: PERSPECTIVE TRANSFORM
    op=cv2.getPerspectiveTransform(approx,pts) #pyright: ignore
    dst=cv2.warpPerspective(orig,op,(800,800))

	# STEP 9: SHOW FINAL RESULT!
    cv2.imshow("FINAL SCANNED",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
