"""Contrast stretching: linearly map an intensity range onto 0-255.

Two variants are shown:
  * stretching - uses the image's own minimum and maximum intensity
  * clipping   - uses a narrower, hand-picked range (values outside are clipped)
"""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from plotting import plot_image_and_histogram

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

CLIP_RANGE = (100, 145)  # (low, high) intensities for the clipping variant


def stretch(gray, low, high):
    """Map intensities in [low, high] to [0, 255]; values outside are clipped."""
    scaled = (gray.astype(np.float32) - low) / max(high - low, 1) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)


def main():
    gray = cv.imread(str(IMAGES / "hist_str_2.png"), cv.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit("Image not found: data/images/hist_str_2.png")

    results = [
        ("Original", gray),
        ("Stretched", stretch(gray, int(gray.min()), int(gray.max()))),
        ("Clipped", stretch(gray, *CLIP_RANGE)),
    ]

    _, axes = plt.subplots(len(results), 2, figsize=(10, 12))
    for (title, image), (ax_image, ax_hist) in zip(results, axes):
        plot_image_and_histogram(ax_image, ax_hist, image, title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
