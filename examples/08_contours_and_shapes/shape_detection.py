"""Detect and label basic shapes (triangle, square, rectangle, circle) from contours."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150
MIN_AREA = 1000  # ignore small contours (noise)
SQUARE_ASPECT_RANGE = (0.98, 1.03)


def classify_shape(contour):
    """Return (label, bounding box) for a contour based on its number of corners."""
    perimeter = cv.arcLength(contour, True)
    corners = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    x, y, w, h = cv.boundingRect(corners)

    if len(corners) == 3:
        label = "triangle"
    elif len(corners) == 4:
        aspect_ratio = w / float(h)
        is_square = SQUARE_ASPECT_RANGE[0] < aspect_ratio < SQUARE_ASPECT_RANGE[1]
        label = "square" if is_square else "rectangle"
    elif len(corners) > 4:
        label = "circle"
    else:
        label = "unknown"

    return label, (x, y, w, h)


def main():
    image = cv.imread(str(IMAGES / "shapes.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/shapes.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)
    edges = cv.Canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f"Contours found: {len(contours)}")

    annotated = image.copy()
    for contour in contours:
        area = cv.contourArea(contour)
        if area <= MIN_AREA:
            continue

        label, (x, y, w, h) = classify_shape(contour)
        print(f"{label}: area={area:.0f}")

        cv.drawContours(annotated, [contour], -1, (255, 0, 0), 4)
        cv.putText(
            annotated,
            label,
            (x + w // 2 - 10, y + h // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

    cv.imshow("Original", image)
    cv.imshow("Gray", gray)
    cv.imshow("Blurred", blurred)
    cv.imshow("Canny edges", edges)
    cv.imshow("Detected shapes", annotated)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
