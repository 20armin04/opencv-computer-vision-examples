"""Shared helpers for the Haar-cascade face detection examples."""

import cv2 as cv

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def load_cascade(filename):
    """Load one of the Haar cascades that ship with OpenCV."""
    cascade = cv.CascadeClassifier(cv.data.haarcascades + filename)
    if cascade.empty():
        raise SystemExit(f"Could not load cascade: {filename}")
    return cascade


class FaceDetector:
    """Detects faces, and eyes and smiles inside each face, and draws them."""

    def __init__(self, face_params, eye_params, smile_params, thickness=2):
        self.face_cascade = load_cascade("haarcascade_frontalface_default.xml")
        self.eye_cascade = load_cascade("haarcascade_eye.xml")
        self.smile_cascade = load_cascade("haarcascade_smile.xml")
        self.face_params = face_params
        self.eye_params = eye_params
        self.smile_params = smile_params
        self.thickness = thickness

    def annotate(self, frame):
        """Draw detections on `frame` in place."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for x, y, w, h in self.face_cascade.detectMultiScale(gray, **self.face_params):
            cv.rectangle(frame, (x, y), (x + w, y + h), RED, self.thickness)

            face_gray = gray[y : y + h, x : x + w]
            # A slice is a view, so drawing on it also draws on `frame`.
            face_color = frame[y : y + h, x : x + w]

            for ex, ey, ew, eh in self.eye_cascade.detectMultiScale(face_gray, **self.eye_params):
                cv.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), GREEN, self.thickness)

            for sx, sy, sw, sh in self.smile_cascade.detectMultiScale(
                face_gray, **self.smile_params
            ):
                cv.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), BLUE, self.thickness)

        return frame
