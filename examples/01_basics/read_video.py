"""Play a video file frame by frame. Press 'q' to quit."""

from pathlib import Path

import cv2 as cv

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"


def main():
    cap = cv.VideoCapture(str(VIDEOS / "bikes.mp4"))
    if not cap.isOpened():
        raise SystemExit("Video not found: data/videos/bikes.mp4")

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv.imshow("Video", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
