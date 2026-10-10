"""Detect license plates in a video using only edges and contour geometry.

A contour counts as a plate if it has four corners, a wide aspect ratio and a
plausible area. Press 'q' to quit.
"""

from pathlib import Path

import cv2 as cv

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"

FRAME_SIZE = (640, 480)
LOW_THRESHOLD = 100
HIGH_THRESHOLD = 210
ASPECT_RATIO_RANGE = (3, 5)
AREA_RANGE = (1000, 3000)
BOX_SCALE = 1.5  # enlarge the drawn box so it covers the whole plate


def is_license_plate(contour):
    """Return True if the contour looks like a license plate."""
    perimeter = cv.arcLength(contour, True)
    corners = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(corners) != 4:
        return False

    _, _, w, h = cv.boundingRect(contour)
    aspect_ratio = w / float(h)
    area = cv.contourArea(contour)

    return (
        ASPECT_RATIO_RANGE[0] < aspect_ratio < ASPECT_RATIO_RANGE[1]
        and AREA_RANGE[0] < area < AREA_RANGE[1]
    )


def main():
    cap = cv.VideoCapture(str(VIDEOS / "demo.mp4"))
    if not cap.isOpened():
        raise SystemExit("Video not found: data/videos/demo.mp4")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv.resize(frame, FRAME_SIZE)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 1)
        edges = cv.Canny(blurred, LOW_THRESHOLD, HIGH_THRESHOLD)

        contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if not is_license_plate(contour):
                continue

            x, y, w, h = cv.boundingRect(contour)
            new_w, new_h = int(w * BOX_SCALE), int(h * BOX_SCALE)
            new_x = x - (new_w - w) // 2
            new_y = y - (new_h - h) // 2
            cv.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 3)

        cv.imshow("License plates", frame)
        cv.imshow("Canny edges", edges)
        if cv.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
