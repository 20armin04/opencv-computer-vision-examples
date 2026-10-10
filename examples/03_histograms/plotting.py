"""Shared plotting helper for the histogram examples."""

import cv2 as cv


def plot_image_and_histogram(ax_image, ax_hist, gray, title, show_cdf=False):
    """Draw a grayscale image on `ax_image` and its histogram on `ax_hist`."""
    ax_image.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax_image.set_title(title)
    ax_image.axis("off")

    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    ax_hist.plot(hist, color="black", label="histogram")

    if show_cdf:
        cdf = hist.cumsum()
        cdf_scaled = cdf * (hist.max() / cdf.max())
        ax_hist.plot(cdf_scaled, color="red", label="cdf")
        ax_hist.legend()

    ax_hist.set_title(f"{title} histogram")
    ax_hist.set_xlabel("Intensity value")
    ax_hist.set_ylabel("Pixel count")
    ax_hist.set_xlim([0, 256])
