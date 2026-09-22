"""T3 -- annotate course plates: 4 corners + plate text + tags, one row per plate.

Run from the repo root (Repte1MAPSIV):
    python ground_truth_maker/truth_maker.py --annotator <your_name>

Requirements: pip install opencv-python numpy

Images are read from data_raw/cv/Frontal and data_raw/cv/Lateral.
Everyone shares one file, ground_truth_maker/ground_truth_cv.csv. If more than
one person is annotating, agree out loud who does which images first (e.g.
split by folder or index range) -- the tool can't coordinate that for you,
it only skips images already in the file at the moment it starts.

Keys:
  click     add a corner (up to 4, any order)
  u         undo last clicked point
  r         reset all clicked points for the current plate
  Enter     save this plate, move to the NEXT image
  a         save this plate, start ANOTHER plate on the SAME image
  n         no readable plate in this image (still asks for lighting)
  p         go back and redo the previous image (this session only)
  1/2/3     lighting: daylight / artificial indoor / night
  k         toggle "bent plate" flag
  t         type/correct the plate text in the terminal (only way text is ever typed)
  q         quit
"""
import argparse
import csv
import datetime
import os
import re
from pathlib import Path

import cv2
import numpy as np

from geometry import order_points_robust

PLATE_REGEX = re.compile(r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$")

ROOT = Path(__file__).resolve().parents[1]  # Repte1MAPSIV/
IMAGES_DIR = ROOT / "data_raw" / "cv"
IMAGE_SUBFOLDERS = ["Frontal", "Lateral"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUT_CSV = Path(__file__).resolve().parent / "ground_truth_cv.csv"

MAX_W, MAX_H = 1400, 900
LIGHTING_KEYS = {ord("1"): "daylight", ord("2"): "artificial", ord("3"): "night"}
COLUMNS = [
    "image_path", "source", "plate_id", "plate_text", "text_source", "vehicle_id",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "corner_order_ambiguous",
    "lighting", "bent", "annotator", "timestamp",
]


def load_course_images():
    """Image paths relative to the repo root, with forward slashes
    (e.g. data_raw/cv/Frontal/0216KZP.jpg), so the CSV is the same on
    Windows, macOS and Linux."""
    paths = []
    for sub in IMAGE_SUBFOLDERS:
        folder = IMAGES_DIR / sub
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                paths.append(p.relative_to(ROOT).as_posix())
    return paths


def already_annotated(path):
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image_path"] for row in csv.DictReader(f)}


def open_csv_for_append(path):
    is_new = not path.exists() or path.stat().st_size == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    if is_new:
        writer.writeheader()
        f.flush()
    return f, writer


def write_row(f, writer, row):
    writer.writerow(row)
    f.flush()


class PlateState:
    def __init__(self, plate_text="", text_source="filename"):
        self.points = []  # original-pixel (x, y), up to 4
        self.plate_text = plate_text
        self.text_source = text_source
        self.lighting = None
        self.bent = False

    def ready_to_save(self):
        return len(self.points) == 4 and self.lighting is not None and self.plate_text


def prefill_plate_text(image_path):
    """Never blocks. Pre-fills from the filename if it matches the plate
    regex; otherwise leaves it empty for 't' to fill in later."""
    stem = Path(image_path).stem
    if PLATE_REGEX.match(stem):
        return stem, "filename"
    return "", "manual"


def prompt_text(prompt):
    return input(prompt).strip()


def draw_overlay(display_img, state, img_idx, n_images, image_path, scale, disp_w):
    for i, p in enumerate(state.points):
        dp = (int(p[0] * scale), int(p[1] * scale))
        cv2.circle(display_img, dp, 5, (0, 255, 255), -1)
        cv2.putText(display_img, str(i + 1), (dp[0] + 6, dp[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    ambiguous = None
    if len(state.points) == 4:
        ordered, ambiguous = order_points_robust(np.array(state.points, dtype=np.float64))
        disp_quad = (ordered * scale).astype(np.int32)
        cv2.polylines(display_img, [disp_quad], True, (0, 0, 255), 2)
        tl = tuple(disp_quad[0])
        cv2.circle(display_img, tl, 8, (255, 0, 255), 2)
        cv2.putText(display_img, "TL", (tl[0] + 10, tl[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    plate_line = (f"Plate: {state.plate_text} [{state.text_source}]" if state.plate_text
                  else "PRESS t TO ENTER PLATE TEXT")
    lines = [
        f"[{img_idx + 1}/{n_images}] {Path(image_path).name}",
        plate_line,
        f"Lighting: {state.lighting or 'NOT SET (press 1/2/3)'}   Bent: {state.bent}",
        f"Points: {len(state.points)}/4" + ("   AMBIGUOUS ORDERING" if ambiguous else ""),
        "u=undo r=reset Enter=save+next a=save+another n=no-plate p=prev q=quit "
        "1/2/3=lighting k=bent t=edit-text",
    ]

    font_scale = round(0.8 * (disp_w / MAX_W), 2)
    thickness = 2
    line_height = int(40 * font_scale)
    banner_h = 20 + len(lines) * line_height
    cv2.rectangle(display_img, (0, 0), (disp_w, banner_h), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        y = 25 + i * line_height
        cv2.putText(display_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return ambiguous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    args = parser.parse_args()

    all_images = load_course_images()
    done = already_annotated(OUT_CSV)
    image_paths = [p for p in all_images if p not in done]
    session_start = 0
    print(f"{len(all_images) - len(image_paths)} images already annotated, "
          f"{len(image_paths)} remaining.")

    f, writer = open_csv_for_append(OUT_CSV)
    offsets = {session_start: os.path.getsize(OUT_CSV) if OUT_CSV.exists() else 0}

    idx = session_start
    window = "annotate"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < 4:
            state.points.append((x / scale, y / scale))

    cv2.setMouseCallback(window, on_mouse)

    while idx < len(image_paths):
        image_path = image_paths[idx]
        img = cv2.imread(str(ROOT / image_path))
        if img is None:
            print(f"Could not read {image_path}, skipping.")
            offsets[idx] = os.path.getsize(OUT_CSV)
            idx += 1
            continue
        h, w = img.shape[:2]
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)

        plate_text, text_source = prefill_plate_text(image_path)
        state = PlateState(plate_text, text_source)
        offsets[idx] = os.path.getsize(OUT_CSV)

        advance = None  # "next", "prev", "quit"
        while advance is None:
            base = cv2.resize(img, (disp_w, disp_h))
            display_img = base.copy()
            ambiguous = draw_overlay(display_img, state, idx, len(image_paths), image_path,
                                      scale, disp_w)
            cv2.imshow(window, display_img)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue
            elif key == ord("u"):
                if state.points:
                    state.points.pop()
            elif key == ord("r"):
                state.points = []
            elif key in LIGHTING_KEYS:
                state.lighting = LIGHTING_KEYS[key]
            elif key == ord("k"):
                state.bent = not state.bent
            elif key == ord("t"):
                typed = prompt_text("Type the plate text: ").strip().upper()
                if typed:
                    state.plate_text = typed
                    state.text_source = "manual"
                # loop continues -> top of loop re-renders the window immediately
            elif key == ord("q"):
                advance = "quit"
            elif key == ord("p"):
                if idx > session_start:
                    f.close()
                    with open(OUT_CSV, "r+", newline="", encoding="utf-8") as tf:
                        tf.truncate(offsets[idx - 1])
                    f, writer = open_csv_for_append(OUT_CSV)
                    idx -= 2  # will be incremented back to idx-1 below
                    advance = "prev"
                else:
                    print("Can't go back further than this session's start.")
            elif key == ord("n"):
                lighting = state.lighting
                if lighting is None:
                    print("Pick a lighting tag first (1/2/3).")
                    continue
                vehicle_id = prompt_text("No readable plate. Type a vehicle_id by hand (or leave blank): ").strip()
                row = {
                    "image_path": image_path, "source": "cv", "plate_id": "NONE",
                    "plate_text": "", "text_source": "", "vehicle_id": vehicle_id,
                    "x1": "", "y1": "", "x2": "", "y2": "", "x3": "", "y3": "", "x4": "", "y4": "",
                    "corner_order_ambiguous": "", "lighting": lighting, "bent": "",
                    "annotator": args.annotator, "timestamp": datetime.datetime.now().isoformat(),
                }
                write_row(f, writer, row)
                advance = "next"
            elif key in (13, 10, ord("a")):
                if not state.ready_to_save():
                    print("Need 4 points, a lighting tag, and plate text before saving.")
                    continue
                ordered, ambiguous = order_points_robust(np.array(state.points, dtype=np.float64))
                row = {
                    "image_path": image_path, "source": "cv",
                    "plate_id": state.plate_text, "plate_text": state.plate_text,
                    "text_source": state.text_source,
                    "vehicle_id": state.plate_text,
                    "x1": round(float(ordered[0][0]), 1), "y1": round(float(ordered[0][1]), 1),
                    "x2": round(float(ordered[1][0]), 1), "y2": round(float(ordered[1][1]), 1),
                    "x3": round(float(ordered[2][0]), 1), "y3": round(float(ordered[2][1]), 1),
                    "x4": round(float(ordered[3][0]), 1), "y4": round(float(ordered[3][1]), 1),
                    "corner_order_ambiguous": bool(ambiguous),
                    "lighting": state.lighting, "bent": state.bent,
                    "annotator": args.annotator, "timestamp": datetime.datetime.now().isoformat(),
                }
                write_row(f, writer, row)
                if key == ord("a"):
                    # a second plate on the same image is never the filename's
                    # plate (a filename can only match one plate) -- force
                    # explicit entry via 't' instead of re-prefilling.
                    lighting = state.lighting
                    state = PlateState(plate_text="", text_source="manual")
                    state.lighting = lighting
                else:
                    advance = "next"

        if advance == "quit":
            break
        idx += 1

    f.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
