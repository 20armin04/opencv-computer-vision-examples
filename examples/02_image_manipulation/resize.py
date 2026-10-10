"""Resize an image to a fixed width and height."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

NEW_SIZE = (350, 350)  # (width, height)


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    resized = cv.resize(image, NEW_SIZE)

    cv.imshow("Original", image)
    cv.imshow("Resized", resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
