"""Warp a region of an image to a top-down view using four hand-picked points."""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

# Corners of the region in the source image, in the order:
# top-left, top-right, bottom-left, bottom-right
SOURCE_POINTS = np.float32([[110, 221], [288, 189], [154, 482], [351, 441]])

OUTPUT_WIDTH, OUTPUT_HEIGHT = 300, 400


def main():
    image = cv.imread(str(IMAGES / "cards.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/cards.jpg")

    destination_points = np.float32(
        [[0, 0], [OUTPUT_WIDTH, 0], [0, OUTPUT_HEIGHT], [OUTPUT_WIDTH, OUTPUT_HEIGHT]]
    )
    matrix = cv.getPerspectiveTransform(SOURCE_POINTS, destination_points)
    warped = cv.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    cv.imshow("Original", image)
    cv.imshow("Warped", warped)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
