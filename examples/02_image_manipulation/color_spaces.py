"""Convert an image between color spaces: BGR, grayscale, RGB and HSV."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    conversions = {
        "Original (BGR)": image,
        "Grayscale": cv.cvtColor(image, cv.COLOR_BGR2GRAY),
        # OpenCV displays images as BGR, so the RGB version looks color-swapped.
        "RGB": cv.cvtColor(image, cv.COLOR_BGR2RGB),
        "HSV": cv.cvtColor(image, cv.COLOR_BGR2HSV),
    }
    for title, result in conversions.items():
        cv.imshow(title, result)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
