"""Count coins: threshold -> erode -> Canny -> contours, then number each coin."""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

ERODE_KERNEL = np.ones((20, 20), np.uint8)  # large kernel separates touching coins
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 150


def main():
    image = cv.imread(str(IMAGES / "coin.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/coin.png")

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)

    # Otsu picks the threshold automatically; invert so coins are white.
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary = cv.bitwise_not(binary)

    eroded = cv.erode(binary, ERODE_KERNEL, iterations=1)
    edges = cv.Canny(eroded, LOW_THRESHOLD, HIGH_THRESHOLD)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f"Coins found: {len(contours)}")

    annotated = image.copy()
    for number, contour in enumerate(contours, start=1):
        x, y, w, h = cv.boundingRect(contour)
        cv.putText(
            annotated,
            str(number),
            (x + w // 2 - 10, y + h // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )

    cv.imshow("Original", image)
    cv.imshow("Binary", binary)
    cv.imshow("Eroded", eroded)
    cv.imshow("Canny edges", edges)
    cv.imshow("Counted coins", annotated)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
