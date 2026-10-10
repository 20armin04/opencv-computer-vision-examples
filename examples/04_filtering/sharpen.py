"""Sharpen an image with a 3x3 convolution kernel."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

SHARPEN_KERNEL = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
)


def main():
    image = cv.imread(str(IMAGES / "sharp.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/sharp.png")

    sharpened = cv.filter2D(image, -1, SHARPEN_KERNEL)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(sharpened, cv.COLOR_BGR2RGB))
    plt.title("Sharpened")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
