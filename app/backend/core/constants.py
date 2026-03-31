import math


## For blob_detector.py, document_scanner.py, box_segmenter.py
NORMAL_SIZE = 2048
AREA = NORMAL_SIZE ** 2
MAX_ASPECT_RATIO = 9.0

## For box_segmenter.py
MIN_AREA = AREA * 0.02
MAX_AREA = AREA * 0.90

## For document_scanner.py
MIN_PAGE_AREA = AREA * 0.30
MAX_PAGE_AREA = AREA * 0.95
BORDER_MARGIN_RATIO = 0.02

## For box_segmenter utility functions (as found in utils)
## Values are overridden by document_scanner path
MAX_SKEW_DEG = 5.0
MAX_TILT_DEG = 15.0
SECTION_CORNER_ANGLE_TOL = 9.0

## For blob_detector.py
AREA_FACTOR = (NORMAL_SIZE/1000)**2
MIN_DOT_AREA = AREA_FACTOR * 50
MAX_DOT_AREA = AREA_FACTOR * 8000
MIN_CIRCULARITY = 0.35  # TEMPORARY REDUCTION FROM 0.40; TODO: Add antialiasing then bring back orig value
MIN_SOLIDITY = 0.65
MIN_BBOX_ASPECT = 0.35
DOT_DEDUP_DIST = math.sqrt(MAX_DOT_AREA) / 2.5  # ~73px at NORMAL_SIZE=2048
MAX_RING_DENSITY = 0.30  # max white-pixel density in ring around dot (filters handwriting clutter)
