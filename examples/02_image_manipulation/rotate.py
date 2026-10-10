"""Rotate an image around its center using an affine transform."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

ANGLE = 45  # degrees; in OpenCV a positive angle is counter-clockwise


def rotate(image, angle):
    """Rotate `image` around its center by `angle` degrees."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(image, matrix, (width, height))


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    image = cv.resize(image, (300, 300))

    cv.imshow("Original", image)
    cv.imshow("Clockwise", rotate(image, -ANGLE))
    cv.imshow("Counter-clockwise", rotate(image, ANGLE))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
