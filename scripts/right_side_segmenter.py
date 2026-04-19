import cv2
import numpy as np
from pathlib import Path


FOLDER = Path(__file__).parent.parent / "dataset" / "DrawEduMath" / "Before"
OUT_FOLDER = Path(__file__).parent.parent / "dataset" / "DrawEduMath" / "Claude_Postprocessing"
DIVIDER_GREY_LO = 110   # lower bound of divider grey value
DIVIDER_GREY_HI = 145   # upper bound of divider grey value
DIVIDER_MIN_X = 0.15    # fraction of image width
DIVIDER_MAX_X = 0.85
WHITE_THRESH = 245      # pixels >= this are considered whitespace
PAD = 8                 # padding around tight crop


def find_divider(gray: np.ndarray) -> int | None:
    """Find the x-column of the solid grey vertical divider.

    The divider is a narrow strip of uniform constant-grey pixels (value ~124-130)
    that spans the full image height, making it uniquely identifiable by:
      - 100% coverage (every pixel in the column is non-white)
      - Very low variance (the same grey value throughout)
      - Grey value within the expected range
    """
    h, w = gray.shape
    min_x = int(w * DIVIDER_MIN_X)
    max_x = int(w * DIVIDER_MAX_X)

    full_coverage = (gray < 250).sum(axis=0) == h
    in_grey_range = (gray.min(axis=0) >= DIVIDER_GREY_LO) & (gray.max(axis=0) <= DIVIDER_GREY_HI)

    candidates = np.where(full_coverage & in_grey_range)[0]
    candidates = candidates[(candidates >= min_x) & (candidates <= max_x)]

    if len(candidates) == 0:
        return None

    return int(candidates[0])


def tight_crop(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray < WHITE_THRESH
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = img.shape[:2]
    r0 = max(0, rmin - PAD)
    r1 = min(h, rmax + PAD + 1)
    c0 = max(0, cmin - PAD)
    c1 = min(w, cmax + PAD + 1)
    return img[r0:r1, c0:c1]


def process(path: Path) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  SKIP (unreadable): {path.name}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    div_x = find_divider(gray)
    if div_x is None:
        print(f"  SKIP (no divider): {path.name}")
        return False

    right = img[:, div_x + 2:]
    cropped = tight_crop(right)

    stem = path.stem.removeprefix("image_")
    out_path = OUT_FOLDER / (stem + path.suffix)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(out_path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        cv2.imwrite(str(out_path), cropped)

    print(f"  OK  {path.name} -> {out_path.name}  (divider @ x={div_x})")
    return True


def main():
    files = sorted(FOLDER.glob("image_*"))
    image_files = [f for f in files if f.suffix.lower() in (".jpeg", ".jpg", ".png")]
    print(f"Processing {len(image_files)} images in {FOLDER}")
    ok = skipped = 0
    for f in image_files:
        if process(f):
            ok += 1
        else:
            skipped += 1
    print(f"\nDone. {ok} saved, {skipped} skipped.")


if __name__ == "__main__":
    main()
