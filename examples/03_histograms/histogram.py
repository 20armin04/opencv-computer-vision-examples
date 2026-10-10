"""Compare the grayscale histograms of dark, light, low- and high-contrast images."""

from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
from plotting import plot_image_and_histogram

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

SAMPLES = {
    "Dark": "hist_dark.png",
    "Light": "hist_light.png",
    "Low contrast": "hist_low_con.png",
    "High contrast": "hist_high_con.png",
}


def main():
    _, axes = plt.subplots(2, 4, figsize=(14, 7))

    for i, (title, filename) in enumerate(SAMPLES.items()):
        image = cv.imread(str(IMAGES / filename), cv.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Image not found: data/images/{filename}")

        row, col = divmod(i, 2)
        plot_image_and_histogram(axes[row, col * 2], axes[row, col * 2 + 1], image, title)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
