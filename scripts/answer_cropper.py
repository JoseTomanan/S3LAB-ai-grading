import cv2
import numpy as np
from pathlib import Path

FOLDER = Path(__file__).parent.parent / "dataset" / "DrawEduMath" / "Claude_Postprocessing"
PAD = 8
GAP_MIN_ROWS = 8  # consecutive near-white rows that count as the header/content separator
DARK_THRESH = 200  # pixel is "dark" if gray < this value (ignores JPEG/PNG noise)


def find_answer_start(gray: np.ndarray) -> int:
    """Return the first row of answer content, skipping the 'Student Response' header.

    Scans for the first significant whitespace gap (>= GAP_MIN_ROWS consecutive
    rows with few dark pixels) after the header, then returns the first content
    row after that gap. Uses dark-pixel count to ignore JPEG compression noise.
    """
    h, w = gray.shape
    # A row is "content" if it has enough genuinely dark pixels (ignores JPEG noise)
    dark_count = (gray < 200).sum(axis=1)
    is_content = dark_count > max(3, w // 80)

    first_content = next((r for r in range(h) if is_content[r]), None)
    if first_content is None:
        return 0

    in_gap = False
    gap_len = 0
    for r in range(first_content, h):
        if not is_content[r]:
            in_gap = True
            gap_len += 1
        else:
            if in_gap and gap_len >= GAP_MIN_ROWS:
                return r
            in_gap = False
            gap_len = 0

    return first_content


def tight_crop(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    dark = gray < DARK_THRESH
    row_dark = dark.sum(axis=1)
    col_dark = dark.sum(axis=0)
    content_rows = np.where(row_dark > max(3, w // 80))[0]
    content_cols = np.where(col_dark > max(3, h // 80))[0]
    if len(content_rows) == 0:
        return img
    rmin, rmax = content_rows[0], content_rows[-1]
    cmin, cmax = content_cols[0], content_cols[-1]
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
    answer_row = find_answer_start(gray)

    answer_img = img[answer_row:, :]
    cropped = tight_crop(answer_img)

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), cropped, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        cv2.imwrite(str(path), cropped)

    print(f"  OK  {path.name}  (answer @ row {answer_row})")
    return True


def main():
    files = sorted(FOLDER.glob("*"))
    image_files = [
        f for f in files
        if f.suffix.lower() in (".jpeg", ".jpg", ".png")
        and not f.stem.startswith("image_")
    ]
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
