"""Crop a region of interest (ROI) out of an image using array slicing."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

# ROI corners: (x1, y1) top-left, (x2, y2) bottom-right
X1, Y1 = 100, 100
X2, Y2 = 400, 300


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    cropped = image[Y1:Y2, X1:X2]

    cv.imshow("Original", image)
    cv.imshow("Cropped", cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
