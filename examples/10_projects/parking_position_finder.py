"""Mark parking spots on an image and save them to parking_positions.json.

Left-click   add a spot (the click is its top-left corner)
Right-click  remove the spot under the cursor
Press 'q'    quit

Run this first, then run parking_space_counter.py.
"""

import json
from pathlib import Path

import cv2 as cv

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"
POSITIONS_FILE = Path(__file__).with_name("parking_positions.json")

SPOT_WIDTH, SPOT_HEIGHT = 108, 46


def load_positions():
    try:
        return json.loads(POSITIONS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_positions(positions):
    POSITIONS_FILE.write_text(json.dumps(positions))


def main():
    image = cv.imread(str(IMAGES / "parking.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/parking.png")

    positions = load_positions()

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            positions.append([x, y])
            save_positions(positions)
        elif event == cv.EVENT_RBUTTONDOWN:
            for i, (px, py) in enumerate(positions):
                if px < x < px + SPOT_WIDTH and py < y < py + SPOT_HEIGHT:
                    positions.pop(i)
                    save_positions(positions)
                    break

    cv.namedWindow("Parking")
    cv.setMouseCallback("Parking", on_mouse)

    while True:
        canvas = image.copy()
        for px, py in positions:
            cv.rectangle(canvas, (px, py), (px + SPOT_WIDTH, py + SPOT_HEIGHT), (255, 0, 0), 2)

        cv.imshow("Parking", canvas)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
