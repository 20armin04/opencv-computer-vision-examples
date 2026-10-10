"""Find and draw external contours: gray -> blur -> Canny -> findContours."""

from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150


def main():
    image = cv.imread(str(IMAGES / "shapes.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/shapes.png")
    image = cv.resize(image, (500, 500))

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)
    edges = cv.Canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f"Contours found: {len(contours)}")

    annotated = image.copy()
    cv.drawContours(annotated, contours, -1, (0, 255, 0), 4)

    cv.imshow("Original", image)
    cv.imshow("Gray", gray)
    cv.imshow("Blurred", blurred)
    cv.imshow("Canny edges", edges)
    cv.imshow("Contours", annotated)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
