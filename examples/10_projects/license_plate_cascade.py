"""Detect license plates in a video with a Haar cascade. Press 'q' to quit.

Note: the cascade bundled with OpenCV was trained on Russian plates, so results
depend heavily on the video.
"""

from pathlib import Path

import cv2 as cv

VIDEOS = Path(__file__).resolve().parents[2] / "data" / "videos"

FRAME_SIZE = (640, 480)


def main():
    cap = cv.VideoCapture(str(VIDEOS / "demo.mp4"))
    if not cap.isOpened():
        raise SystemExit("Video not found: data/videos/demo.mp4")

    cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_russian_plate_number.xml")
    if cascade.empty():
        raise SystemExit("Could not load the license plate cascade")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv.resize(frame, FRAME_SIZE)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for x, y, w, h in cascade.detectMultiScale(gray, 1.1, 12):
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv.imshow("License plates", frame)
        if cv.waitKey(15) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
