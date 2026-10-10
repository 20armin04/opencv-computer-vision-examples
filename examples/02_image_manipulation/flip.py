"""Flip an image horizontally, vertically, or both."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    flips = {
        "Original": image,
        "Horizontal flip": cv.flip(image, 1),
        "Vertical flip": cv.flip(image, 0),
        "Horizontal + vertical flip": cv.flip(image, -1),
    }
    for title, result in flips.items():
        cv.imshow(title, result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
