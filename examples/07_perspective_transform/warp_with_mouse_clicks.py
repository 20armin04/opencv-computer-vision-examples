"""Pick four corners with the mouse and warp that region to a top-down view.

Click the corners in this order: top-left, top-right, bottom-left, bottom-right.
Press 'q' to quit.
"""

from pathlib import Path

import cv2 as cv
import numpy as np

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"

OUTPUT_WIDTH, OUTPUT_HEIGHT = 400, 500
WINDOW = "Original (click 4 corners)"


def main():
    image = cv.imread(str(IMAGES / "cards.jpg"))
    if image is None:
        raise SystemExit("Image not found: data/images/cards.jpg")

    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    cv.namedWindow(WINDOW)
    cv.setMouseCallback(WINDOW, on_mouse)

    destination_points = np.float32(
        [[0, 0], [OUTPUT_WIDTH, 0], [0, OUTPUT_HEIGHT], [OUTPUT_WIDTH, OUTPUT_HEIGHT]]
    )

    while True:
        canvas = image.copy()
        for point in points:
            cv.circle(canvas, point, 5, (0, 255, 0), -1)
        cv.imshow(WINDOW, canvas)

        if len(points) == 4:
            matrix = cv.getPerspectiveTransform(np.float32(points), destination_points)
            warped = cv.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            cv.imshow("Warped", warped)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
