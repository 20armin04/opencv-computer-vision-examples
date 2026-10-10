# Sample data

The examples read their inputs from this folder. Put the files below in
`images/` and `videos/` (keep the exact names), or edit the file names at the
top of the scripts to use your own media.

Only commit media you own or that is licensed for redistribution, and credit
the sources in the last column.

## Images (`data/images/`)

| File | Used by | Source / credit |
| --- | --- | --- |
| `face.png` | `read_image`, `color_spaces`, `crop`, `resize`, `rotate`, `flip`, `box_blur`, `face_detection_image` | |
| `hist_dark.png`, `hist_light.png`, `hist_low_con.png`, `hist_high_con.png` | `histogram` | |
| `hist_str_2.png` | `contrast_stretching`, `histogram_equalization` | |
| `gaussian.png` | `gaussian_blur` | |
| `median.bmp` | `median_blur` | |
| `sharp.png` | `sharpen` | |
| `edge2.jpg` | `sobel`, `canny` | |
| `morph1.png` | `erosion`, `dilation` | |
| `documentscanner.jpg` | `document_scanner` | |
| `documentscanner2.jpg` | `closing_with_canny` | |
| `cards.jpg` | `warp_perspective`, `warp_with_mouse_clicks` | |
| `shapes.png` | `find_contours`, `shape_detection` | |
| `coin.png` | `coin_counting` | |
| `parking.png` | `parking_position_finder` | |

## Videos (`data/videos/`)

| File | Used by | Source / credit |
| --- | --- | --- |
| `bikes.mp4` | `read_video` | |
| `demo.mp4` | `license_plate_cascade`, `license_plate_contours` | |
| `parking.mp4` | `parking_space_counter` | |

## Notes

- `warp_perspective` uses hard-coded corner coordinates that match `cards.jpg`.
- The parking spots in `examples/10_projects/parking_positions.json` match
  `parking.png` / `parking.mp4`. For different footage, re-run
  `parking_position_finder.py` to mark new spots.
