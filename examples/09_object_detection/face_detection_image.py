"""Detect faces, eyes and smiles in a still image with Haar cascades."""

from pathlib import Path

import cv2 as cv
from haar_utils import FaceDetector

IMAGES = Path(__file__).resolve().parents[2] / "data" / "images"


def main():
    image = cv.imread(str(IMAGES / "face.png"))
    if image is None:
        raise SystemExit("Image not found: data/images/face.png")

    detector = FaceDetector(
        face_params={"scaleFactor": 1.5, "minNeighbors": 10},
        eye_params={"scaleFactor": 1.5, "minNeighbors": 8},
        smile_params={"scaleFactor": 1.2, "minNeighbors": 10},
        thickness=4,
    )

    cv.imshow("Face detection", detector.annotate(image))
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
