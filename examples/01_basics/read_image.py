"""Read an image from disk and display it."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
