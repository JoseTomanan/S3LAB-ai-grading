## For blob_detector.py, document_scanner.py, box_segmenter.py
NORMAL_SIZE = 2048
AREA = NORMAL_SIZE ** 2
MAX_ASPECT_RATIO = 9.0

## For box_segmenter.py
MIN_AREA = AREA * 0.04
MAX_AREA = AREA * 0.90

## For document_scanner.py
MIN_PAGE_AREA = AREA * 0.30
MAX_PAGE_AREA = AREA * 0.95
BORDER_MARGIN_RATIO = 0.02

## For box segmentation utility functions (as found in utils)
MAX_SKEW_DEG = 5.0
MAX_TILT_DEG = 15.0
SECTION_CORNER_ANGLE_TOL = 12.0
