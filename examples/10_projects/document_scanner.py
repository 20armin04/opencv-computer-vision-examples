"""Document scanner: find the page in a photo and warp it to a flat, top-down view."""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

LOW_THRESHOLD = 150
HIGH_THRESHOLD = 250
MIN_AREA = 5000  # ignore small contours (noise)
KERNEL = np.ones((5, 5), np.uint8)


def preprocess(image):
    """Return a binary edge map with thickened borders."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, LOW_THRESHOLD, HIGH_THRESHOLD)
    dilated = cv.dilate(edges, KERNEL, iterations=2)
    return cv.erode(dilated, KERNEL, iterations=1)


def find_document_corners(edges):
    """Return the four corners of the largest quadrilateral, or None if not found."""
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best_corners = None
    best_area = MIN_AREA
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= best_area:
            continue

        perimeter = cv.arcLength(contour, True)
        corners = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(corners) == 4:
            best_corners, best_area = corners, area

    return best_corners


def order_corners(corners):
    """Order corners as: top-left, top-right, bottom-left, bottom-right."""
    points = corners.reshape((4, 2))

    by_y = points[np.argsort(points[:, 1])]
    top, bottom = by_y[:2], by_y[2:]

    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]

    return np.float32([top_left, top_right, bottom_left, bottom_right])


def warp_to_top_down(image, corners):
    """Warp the quadrilateral given by `corners` into an upright rectangle."""
    source = order_corners(corners)
    _, _, w, h = cv.boundingRect(source)
    destination = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    matrix = cv.getPerspectiveTransform(source, destination)
    return cv.warpPerspective(image, matrix, (w, h))


def main():
    image = cv.imread(str(IMAGES / "documentscanner.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/documentscanner.jpg")
    image = cv.resize(image, (480, 640))

    edges = preprocess(image)
    corners = find_document_corners(edges)
    if corners is None:
        raise SystemExit("No document found in the image")

    outlined = image.copy()
    cv.drawContours(outlined, [corners], -1, (0, 255, 0), 3)

    cv.imshow("Original", image)
    cv.imshow("Edges", edges)
    cv.imshow("Detected document", outlined)
    cv.imshow("Scanned document", warp_to_top_down(image, corners))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
