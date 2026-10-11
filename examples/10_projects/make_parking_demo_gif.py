"""Create docs/parking_demo.gif: the raw video (before) next to the result (after).

Run it once after adding data/videos/parking.mp4:
    python examples/10_projects/make_parking_demo_gif.py
"""

import json
from pathlib import Path

import cv2 as cv
from parking_space_counter import POSITIONS_FILE, VIDEOS, draw_spots, preprocess
from PIL import Image

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "parking_demo.gif"

PANEL_WIDTH = 480  # width of each side; the GIF is twice this wide
FPS = 10
MAX_SECONDS = 8
PALETTE_COLORS = 128  # fewer colors -> smaller file


def add_label(panel, text):
    cv.rectangle(panel, (0, 0), (110, 30), (255, 255, 255), -1)
    cv.putText(panel, text, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    positions = json.loads(POSITIONS_FILE.read_text())

    cap = cv.VideoCapture(str(VIDEOS / "parking.mp4"))
    if not cap.isOpened():
        raise SystemExit("Video not found: data/videos/parking.mp4")

    step = max(round((cap.get(cv.CAP_PROP_FPS) or 30) / FPS), 1)
    frames = []
    index = 0

    while len(frames) < FPS * MAX_SECONDS:
        success, frame = cap.read()
        if not success:
            break

        if index % step == 0:
            before = frame.copy()
            draw_spots(preprocess(frame), frame, positions)  # draws on `frame`

            scale = PANEL_WIDTH / frame.shape[1]
            before, after = (
                cv.resize(p, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
                for p in (before, frame)
            )
            add_label(before, "Before")
            add_label(after, "After")

            rgb = cv.cvtColor(cv.hconcat([before, after]), cv.COLOR_BGR2RGB)
            frames.append(
                Image.fromarray(rgb).convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=PALETTE_COLORS
                )
            )
        index += 1

    cap.release()
    if not frames:
        raise SystemExit("No frames could be read from the video")

    OUTPUT.parent.mkdir(exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=True,
    )
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Saved {OUTPUT.relative_to(OUTPUT.parents[1])} ({len(frames)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
