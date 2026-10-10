"""Sobel edge detection: horizontal, vertical and combined gradients."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

KERNEL_SIZE = 5


def main():
    image = cv.imread(str(IMAGES / "edge2.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/edge2.jpg")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    results = [
        ("Original", cv.cvtColor(image, cv.COLOR_BGR2RGB)),
        ("Sobel X", cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=KERNEL_SIZE)),
        ("Sobel Y", cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=KERNEL_SIZE)),
        ("Sobel XY", cv.Sobel(gray, cv.CV_64F, 1, 1, ksize=KERNEL_SIZE)),
    ]

    plt.figure(figsize=(10, 10))
    for i, (title, result) in enumerate(results, start=1):
        plt.subplot(2, 2, i)
        plt.imshow(result, cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
