"""Detect faces, eyes and smiles from the webcam in real time. Press 'q' to quit."""

import cv2 as cv
from haar_utils import FaceDetector


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")

    detector = FaceDetector(
        face_params={"scaleFactor": 1.1, "minNeighbors": 5},
        eye_params={"scaleFactor": 1.1, "minNeighbors": 5},
        smile_params={"scaleFactor": 1.2, "minNeighbors": 15},
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv.imshow("Face detection", detector.annotate(frame))
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
