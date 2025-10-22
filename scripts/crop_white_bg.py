import numpy as np
import cv2


def crop_white_background(input_path: str, output_path: str, line_thickness: int=5, color_tolerance: int=10):
	"""
	Finds a thick, continuous, single-color vertical line to use as a divider.
	Then, it crops the largest content block to the right of that line.

	Args:
		input_path (str): The path to the input image file.
		output_path (str): The path to save the cropped image file.
		padding (int): Pixels to add as a border around the final crop.
		line_thickness (int): The approximate thickness of the divider line to find.
		color_tolerance (int): How much a pixel's color can deviate from the line's
								main color (0-255). A small value handles JPEG artifacts.
	"""
	try:
		original_img = cv2.imread(input_path)
		if original_img is None:
			raise FileNotFoundError
		height, width, _ = original_img.shape
	except (FileNotFoundError, AttributeError):
		print(f"Error: The file '{input_path}' was not found or is not a valid image.")
		return

	divider_x = None

	# --- Step 1: Scan for the Vertical Line ---
	# Iterate through each possible starting column of the line
	for x in range(width - line_thickness):
		# Extract a vertical strip of the specified thickness
		strip = original_img[:, x : x + line_thickness]

		# Take a sample color from the top-center of the strip
		target_color = strip[0, line_thickness // 2]

		# --- Step 2: Check for Color Consistency ---
		# Calculate the color difference for the entire strip at once.
		# We use absolute difference and check if any pixel deviates too much.
		# Using int16 to prevent overflow errors during subtraction.
		diff = np.abs(strip.astype(np.int16) - target_color.astype(np.int16))

		# If the maximum difference in the entire strip is within our tolerance,
		# we've found our continuous, consistent-colored line.
		if np.max(diff) <= color_tolerance:
			# The divider is the center of the strip
			divider_x = x + line_thickness // 2
			print(f"Detected {line_thickness}px thick vertical line at x={divider_x}")
			break

	if divider_x is None:
		print("Could not detect a continuous vertical line with the specified thickness and color consistency.")
		return

	# --- Step 3: Isolate, Threshold, and Crop (Same as before) ---
	gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
	right_side_gray = gray_img[:, divider_x:]
	_, mask = cv2.threshold(right_side_gray, 220, 255, cv2.THRESH_BINARY_INV)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x_rel, y_rel, w, h = cv2.boundingRect(largest_contour)

		# Calculate final crop coordinates
		x1 = max(0, divider_x + x_rel)
		y1 = max(0, y_rel)
		x2 = min(width, divider_x + x_rel + w)
		y2 = min(height, y_rel + h)

		cropped_img = original_img[y1:y2, x1:x2]
		cv2.imwrite(output_path, cropped_img)
		print(f"Successfully cropped and saved image to '{output_path}'")
	else:
		print(f"Found divider but no content to the right of it.")



if __name__ == "__main__":
	for i in range(2030):
		input_file = f"dataset/DrawEduMath/Before/image_{i}.png"
		output_path = f"dataset/DrawEduMath/Postprocessing/{i}.png"
		crop_white_background(input_file, output_path)