## For blob_detector.py, document_scanner.py, box_segmenter.py
NORMAL_SIZE = 2048

## Derived from NORMAL_SIZE; used to compute area thresholds below
AREA = NORMAL_SIZE ** 2

## For box_segmenter.py
MIN_AREA = AREA * 0.01
MAX_AREA = AREA * 0.90

## For document_scanner.py
MIN_PAGE_AREA = AREA * 0.30
MAX_PAGE_AREA = AREA * 0.90
BORDER_MARGIN_RATIO = 0.02

## For document_scanner.py, box_segmenter.py
MAX_ASPECT_RATIO = 9.0

## For utils/__init__.py
MAX_SKEW_DEG = 5.0
MAX_TILT_DEG = 15.0
SECTION_CORNER_ANGLE_TOL = 12.0
