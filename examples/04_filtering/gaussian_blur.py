"""Gaussian blur with increasing kernel sizes."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

KERNEL_SIZES = (5, 15, 51)  # must be odd
SIGMA = 1


def main():
    image = cv.imread(str(IMAGES / "gaussian.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/gaussian.png")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # matplotlib expects RGB

    results = [("Original", image)]
    results += [(f"Gaussian {k}x{k}", cv.GaussianBlur(image, (k, k), SIGMA)) for k in KERNEL_SIZES]

    plt.figure(figsize=(10, 10))
    for i, (title, result) in enumerate(results, start=1):
        plt.subplot(2, 2, i)
        plt.imshow(result)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
