"""Morphological dilation: grows bright regions."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

KERNEL = np.ones((10, 10), np.uint8)
ITERATIONS = 3


def main():
    gray = cv.imread(str(IMAGES / "morph1.png"), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("Image not found: data/images/morph1.png")

    dilated = cv.dilate(gray, KERNEL, iterations=ITERATIONS)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dilated, cmap="gray")
    plt.title("Dilated")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
