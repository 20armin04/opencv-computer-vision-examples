"""Histogram equalization, with the cumulative distribution function (CDF) overlaid."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
from plotting import plot_image_and_histogram

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"


def main():
    gray = cv.imread(str(IMAGES / "hist_str_2.png"), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("Image not found: data/images/hist_str_2.png")

    equalized = cv.equalizeHist(gray)

    _, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_image_and_histogram(axes[0, 0], axes[0, 1], gray, "Original", show_cdf=True)
    plot_image_and_histogram(axes[1, 0], axes[1, 1], equalized, "Equalized", show_cdf=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
