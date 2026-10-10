"""Count free parking spots in a video using the spots saved by parking_position_finder.py.

Each frame is converted to a binary edge-like image; a spot with few white
pixels is considered free. Press 'q' to quit.
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"
POSITIONS_FILE = Path(__file__).with_name("parking_positions.json")

SPOT_WIDTH, SPOT_HEIGHT = 108, 46
FREE_THRESHOLD = 800  # fewer white pixels than this -> the spot is free
FREE_COLOR = (0, 255, 0)
OCCUPIED_COLOR = (0, 0, 255)


def preprocess(frame):
    """Return a binary image where busy (occupied) areas contain many white pixels."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)
    binary = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 25, 16
    )
    denoised = cv.medianBlur(binary, 5)  # removes salt-and-pepper noise
    return cv.dilate(denoised, np.ones((3, 3), np.uint8), iterations=1)


def draw_spots(processed, frame, positions):
    """Draw every spot on `frame` (green = free, red = occupied) and a free counter."""
    free_spots = 0

    for x, y in positions:
        spot = processed[y : y + SPOT_HEIGHT, x : x + SPOT_WIDTH]
        is_free = cv.countNonZero(spot) < FREE_THRESHOLD
        free_spots += is_free

        color = FREE_COLOR if is_free else OCCUPIED_COLOR
        cv.rectangle(frame, (x, y), (x + SPOT_WIDTH, y + SPOT_HEIGHT), color, 2)

    cv.rectangle(frame, (50, 20), (270, 70), (255, 255, 255), -1)
    label = f"Free: {free_spots}/{len(positions)}"
    cv.putText(frame, label, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def main():
    try:
        positions = json.loads(POSITIONS_FILE.read_text())
    except FileNotFoundError:
        raise SystemExit("No parking_positions.json found. Run parking_position_finder.py first.")

    cap = cv.VideoCapture(str(VIDEOS / "parking.mp4"))
    if not cap.isOpened():
        raise SystemExit("Video not found: data/videos/parking.mp4")

    while True:
        # Loop the video: rewind when it reaches the end.
        if cap.get(cv.CAP_PROP_POS_FRAMES) == cap.get(cv.CAP_PROP_FRAME_COUNT):
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)

        success, frame = cap.read()
        if not success:
            break

        draw_spots(preprocess(frame), frame, positions)

        cv.imshow("Parking spaces", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
