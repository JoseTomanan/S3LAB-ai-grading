import cv2
import numpy as np
import mapper



if __name__ == "__main__":
    image=cv2.imread("ignoreable/ruled1.jpg")   #read in the image
    assert image is not None

    image=cv2.resize(image,(1300,800)) #resizing because opencv does not work well with bigger images
    orig=image.copy()

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #RGB To Gray Scale

    # NEW: Remove horizontal lines
    # Create a horizontal kernel for morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (400, 1))

    # Detect horizontal lines using morphology
    detected_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cv2.imshow("Detected lines", detected_lines)

    # Subtract the lines from the original
    gray_no_lines = cv2.subtract(gray, detected_lines)

    # Optional: Apply adaptive thresholding to enhance content
    # thresh = cv2.adaptiveThreshold(gray_no_lines, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY, 11, 2)
    cv2.imshow("Without Lines", gray_no_lines)

    # STEP: APPLY GAUSSIAN BLUR
    blurred = cv2.GaussianBlur(gray_no_lines, (5,5), 0)  # (5,5) is the kernel size and 0 is sigma that determines the amount of blur
    cv2.imshow("Blur",blurred)

    # STEP: SEGMENT CANNY LINES
    edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
    cv2.imshow("Canny",edged)


    contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
    contours=sorted(contours,key=cv2.contourArea,reverse=True)


    #the loop extracts the boundary contours of the page
    for c in contours:
        p=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*p,True)

        if len(approx)==4:
            target=approx
            break
    approx=mapper.mapp(target) #find endpoints of the sheet

    pts=np.float32(np.array([[0,0],[800,0],[800,800],[0,800]]))  #map to 800*800 target window

    op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
    dst=cv2.warpPerspective(orig,op,(800,800))


    cv2.imshow("FINAL SCANNED",dst)
    # press q or Esc to close
    cv2.waitKey(0)
    cv2.destroyAllWindows()
