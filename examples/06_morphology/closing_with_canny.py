"""Close gaps in Canny edges by dilating, then eroding (morphological closing)."""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 200
KERNEL = np.ones((5, 5), np.uint8)


def main():
    image = cv.imread(str(IMAGES / "documentscanner2.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/documentscanner2.jpg")
    image = cv.resize(image, (500, 600))

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)
    edges = cv.Canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)

    dilated = cv.dilate(edges, KERNEL, iterations=1)
    closed = cv.erode(dilated, KERNEL, iterations=1)

    cv.imshow("Original", image)
    cv.imshow("Canny edges", edges)
    cv.imshow("Dilated", dilated)
    cv.imshow("Dilated then eroded (closing)", closed)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
