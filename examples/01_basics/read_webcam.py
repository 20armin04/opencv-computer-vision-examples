"""Show the live webcam feed. Press 'q' to quit."""

import cv2 as cv


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam")

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 440)
    cap.set(cv.CAP_PROP_BRIGHTNESS, 100)

    while True:
        success, frame = cap.read()
        if not success:
            break

        cv.imshow("Webcam", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
