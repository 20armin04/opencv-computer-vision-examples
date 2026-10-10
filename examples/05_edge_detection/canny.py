"""Canny edge detection on a blurred grayscale image."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

BLUR_KERNEL = (3, 3)
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 100


def main():
    image = cv.imread(str(IMAGES / "edge2.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/edge2.jpg")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, BLUR_KERNEL, 1)
    edges = cv.Canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny edges")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
