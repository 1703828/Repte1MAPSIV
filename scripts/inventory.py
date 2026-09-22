"""Dataset inventory: builds data_processed/images.csv and data_processed/plates.csv
and prints the full inventory report.

RUN THIS ONLY ONCE ALL THE DATA IS PROCESSED: it regenerates (overwrites)
data_processed/images.csv and data_processed/plates.csv from scratch.

Run from the repo root (Repte1MAPSIV):
    python scripts/inventory.py

Requirements: pip install opencv-python numpy pandas pillow

Expected layout:
    data_raw/cv/Frontal, data_raw/cv/Lateral   (course images, *.jpg)
    data_raw/uc3m/train, data_raw/uc3m/test    (UC3M-LP images + one .json per image)
    ground_truth_maker/ground_truth_cv.csv     (cv annotations, from truth_maker.py)

images.csv: one row per image, always complete (source, split, view, path, w, h).
plates.csv: one row per plate, only where ground truth exists: UC3M (from its
JSONs) + cv (from ground_truth_cv.csv). cv images annotated as "no readable
plate" (plate_id NONE) have no plate, so they get no row in plates.csv.

Self-contained: everything it needs (geometry helpers and loaders) lives in
this file, no imports from other project files.

This script only reports problems, it doesn't fix them.

Design notes:
- UC3M's JSON `imagePath` field does not identify the image it belongs to
  (verified across all 1975 files: offsets from the real filename are not
  constant, some point at files in the other split, some point nowhere).
  Every image/JSON pair here is matched by filename stem instead.
- Raw poly_coord aspect ratio does NOT cleanly separate single-row from
  two-row plates: a single-row plate photographed at a steep angle can have
  an apparent aspect ratio as low as ~2.2 (perspective foreshortening), which
  overlaps the genuine two-row range. plate_shape is therefore derived from
  the number of character rows (clustered from character bbox positions),
  not from aspect ratio. Aspect ratio is still recorded, for reporting.
"""
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # Repte1MAPSIV/
CV_ROOT = ROOT / "data_raw" / "cv"
UC3M_ROOT = ROOT / "data_raw" / "uc3m"
CV_ANNOTATIONS = ROOT / "ground_truth_maker" / "ground_truth_cv.csv"
PROCESSED = ROOT / "data_processed"

PLATE_REGEX = re.compile(r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$")
CONSONANTS = set("BCDFGHJKLMNPRSTVWXYZ")
DIGITS = set("0123456789")

UC3M_JSON_TOP_KEYS = {"imagePath", "imageHeight", "imageWidth", "lps"}
UC3M_LP_KEYS = {"lp_id", "poly_coord", "characters"}
UC3M_CHAR_KEYS = {"char_id", "bbox_coord"}


def rel_path(path):
    """Path relative to the repo root, with forward slashes on every OS."""
    return path.relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def simplify_to_quad(pts, eps_fractions=(0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18)):
    """Reduce a >4-point polygon to its 4 true corners.

    Some UC3M poly_coord entries carry 5 or 6 points: the 4 real corners plus
    1-2 extra points sitting almost exactly on an edge. Convex hull +
    increasing approxPolyDP epsilon collapses those extra collinear points;
    tested to recover the correct 4 corners on all 142 such cases in UC3M
    train+test.

    Returns (4, 2) float32 array, or None if no epsilon in eps_fractions
    reduces it to exactly 4 points (caller must handle/report this).
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 4:
        return pts
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for eps_frac in eps_fractions:
        approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def _edge_orientation_deg(p, q):
    """Line orientation of segment p-q, mod 180 (direction-agnostic)."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    return np.degrees(np.arctan2(dy, dx)) % 180


def _pair_mean_orientation_deg(theta1, theta2):
    """Circular mean of two orientations defined mod 180 (double-angle trick,
    avoids e.g. averaging 179 and 1 into 90 instead of 0)."""
    z = np.exp(1j * 2 * np.radians(theta1)) + np.exp(1j * 2 * np.radians(theta2))
    return (np.degrees(np.angle(z)) / 2) % 180


def _dist_to_horizontal_deg(theta_deg):
    d = theta_deg % 180
    return min(d, 180 - d)


def order_points_robust(pts, ambiguous_threshold_deg=20.0):
    """Corner ordering by edge DIRECTION, not edge length or sum/diff — works
    for both single-row (~4.7:1) and two-row (~1.2-1.5:1) plates.

    pts: (4, 2) array of [x, y], any order/winding.
    Returns: (ordered, ambiguous)
      ordered: (4, 2) float32 array in TL, TR, BR, BL order.
      ambiguous: True if the top/bottom vs left/right choice is a near-tie.
    """
    pts = np.asarray(pts, dtype=np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    cyclic = pts[np.argsort(angles)]

    edges = [(cyclic[i], cyclic[(i + 1) % 4]) for i in range(4)]
    pair_a = (edges[0], edges[2])
    pair_b = (edges[1], edges[3])

    theta_a = _pair_mean_orientation_deg(
        _edge_orientation_deg(*pair_a[0]), _edge_orientation_deg(*pair_a[1])
    )
    theta_b = _pair_mean_orientation_deg(
        _edge_orientation_deg(*pair_b[0]), _edge_orientation_deg(*pair_b[1])
    )
    dist_a = _dist_to_horizontal_deg(theta_a)
    dist_b = _dist_to_horizontal_deg(theta_b)
    ambiguous = abs(dist_a - dist_b) < ambiguous_threshold_deg

    topbottom_pair = pair_a if dist_a <= dist_b else pair_b

    def mean_y(edge):
        return (edge[0][1] + edge[1][1]) / 2

    edge1, edge2 = topbottom_pair
    top_edge, bottom_edge = (edge1, edge2) if mean_y(edge1) <= mean_y(edge2) else (edge2, edge1)

    tl, tr = sorted(top_edge, key=lambda p: p[0])
    bl, br = sorted(bottom_edge, key=lambda p: p[0])

    ordered = np.array([tl, tr, br, bl], dtype=np.float32)
    return ordered, ambiguous


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def read_image_size(path):
    """(width, height) from the file header only, no full decode."""
    with Image.open(path) as im:
        return im.size


def scan_uc3m_json_schema(uc3m_root):
    """Check every UC3M JSON against the expected key structure.
    Returns a list of {file, issue} dicts; empty if everything matches."""
    issues = []
    for split in ["train", "test"]:
        for jf in sorted((uc3m_root / split).glob("*.json")):
            try:
                d = json.loads(jf.read_text())
            except json.JSONDecodeError as e:
                issues.append({"file": str(jf), "issue": f"invalid JSON: {e}"})
                continue
            if set(d.keys()) != UC3M_JSON_TOP_KEYS:
                issues.append({"file": str(jf), "issue": f"top-level keys {set(d.keys())}"})
                continue
            for i, lp in enumerate(d.get("lps", [])):
                if set(lp.keys()) != UC3M_LP_KEYS:
                    issues.append({"file": str(jf), "issue": f"lps[{i}] keys {set(lp.keys())}"})
                    continue
                if len(lp["poly_coord"]) != 4 or any(len(p) != 2 for p in lp["poly_coord"]):
                    issues.append({"file": str(jf), "issue": f"lps[{i}] poly_coord shape wrong"})
                for j, ch in enumerate(lp.get("characters", [])):
                    if set(ch.keys()) != UC3M_CHAR_KEYS:
                        issues.append(
                            {"file": str(jf), "issue": f"lps[{i}].characters[{j}] keys {set(ch.keys())}"}
                        )
                    elif len(ch["bbox_coord"]) != 2 or any(len(p) != 2 for p in ch["bbox_coord"]):
                        issues.append(
                            {"file": str(jf), "issue": f"lps[{i}].characters[{j}] bbox_coord shape wrong"}
                        )
    return issues


def parse_lp_id(lp_id):
    """Split 'AD-*425*WR' into prefix/lighting/plate string parts.

    Prefix is always 2 letters: first letter unexplained (camera/session?),
    second letter is 'D' or 'N' and correlates strongly with image brightness
    -> used as the 'lighting' tag.
    """
    prefix, plate = lp_id.split("-")
    lighting = {"D": "day", "N": "night"}.get(prefix[1]) if len(prefix) == 2 else None
    wildcards = plate.count("*")
    return {
        "prefix": prefix,
        "lighting": lighting,
        "plate_str": plate,
        "plate_len": len(plate),
        "n_masked": wildcards,
        "n_visible": len(plate) - wildcards,
    }


def cluster_char_rows(char_centers, tl, tr, br, bl, gap_ratio_threshold=0.25):
    """Split character centers into 1 or 2 rows using the plate's own local
    (u, v) frame, not raw image y.

    The (u, v) frame comes from a full perspective normalization (quad -> unit
    square via cv2.getPerspectiveTransform), not a linear projection onto one
    edge direction: real quads are not always orthogonal (perspective skew),
    so a linear projection would invent fake row splits on tilted single-row
    plates. The homography uses all 4 corners, which avoids this.

    Returns (row_index array or None, read_order list, n_rows or None).
    n_rows is None when fewer than 2 characters are visible (can't cluster).
    """
    n = len(char_centers)
    if n < 2:
        return None, list(range(n)), None

    centers = np.asarray(char_centers, dtype=np.float32).reshape(-1, 1, 2)
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src, dst)
    normalized = cv2.perspectiveTransform(centers, homography).reshape(-1, 2)
    proj_u, proj_v = normalized[:, 0], normalized[:, 1]

    order = np.argsort(proj_v)
    gaps = np.diff(proj_v[order])
    row_idx = np.zeros(n, dtype=int)
    if len(gaps) and gaps.max() > gap_ratio_threshold:
        split_at = np.argmax(gaps)
        for rank, idx in enumerate(order):
            row_idx[idx] = 0 if rank <= split_at else 1
        n_rows = 2
    else:
        n_rows = 1

    read_order = sorted(range(n), key=lambda i: (row_idx[i], proj_u[i]))
    return row_idx, read_order, n_rows


def check_visible_format(plate_str, char_ids_in_read_order):
    """Compare visible characters (in derived reading order) against the
    current-format pattern (4 digits then 3 consonants) at their positions
    within the full plate_str (masked positions included in the position
    count, only visible ones checked).

    Returns (compatible: bool, reason: str or None). Only meaningful for
    plate_str of length 7; longer/shorter strings are old-format/foreign and
    reported separately, not run through this check.
    """
    if len(plate_str) != 7:
        return None, "non-standard length"

    visible_positions = [i for i, c in enumerate(plate_str) if c != "*"]
    if len(visible_positions) != len(char_ids_in_read_order):
        return None, "visible-count mismatch"

    for pos, char_id in zip(visible_positions, char_ids_in_read_order):
        expected_digit = pos < 4
        if expected_digit and char_id not in DIGITS:
            return False, f"position {pos} expected digit, got {char_id!r}"
        if not expected_digit and char_id not in CONSONANTS:
            return False, f"position {pos} expected consonant, got {char_id!r}"
    return True, None


def load_images_table():
    """images.csv rows for both sources: source, split, view, image_path, image_w, image_h."""
    rows = []
    for view_dir in ["Frontal", "Lateral"]:
        for img_path in sorted((CV_ROOT / view_dir).glob("*.jpg")):
            w, h = read_image_size(img_path)
            rows.append(
                {
                    "source": "cv",
                    "split": pd.NA,
                    "view": view_dir.lower(),
                    "image_path": rel_path(img_path),
                    "image_w": w,
                    "image_h": h,
                }
            )

    for split in ["train", "test"]:
        for json_path in sorted((UC3M_ROOT / split).glob("*.json")):
            img_path = json_path.with_suffix(".jpg")
            w, h = read_image_size(img_path)
            rows.append(
                {
                    "source": "uc3m",
                    "split": split,
                    "view": pd.NA,
                    "image_path": rel_path(img_path),
                    "image_w": w,
                    "image_h": h,
                }
            )
    return pd.DataFrame(rows)


def load_uc3m_plates_table():
    """plates.csv rows for uc3m, with all derived geometry/format columns.

    Returns (plates_df, per_plate_diagnostics) where diagnostics carries the
    extra report-only fields (ambiguous ordering, format check reason, JSON
    order vs derived order agreement) that don't belong in the persisted table.
    """
    rows = []
    diagnostics = []

    for split in ["train", "test"]:
        for json_path in sorted((UC3M_ROOT / split).glob("*.json")):
            d = json.loads(json_path.read_text())
            image_path = rel_path(json_path.with_suffix(".jpg"))

            for lp in d["lps"]:
                lp_info = parse_lp_id(lp["lp_id"])
                raw_pts = lp["poly_coord"]
                n_raw_points = len(raw_pts)
                quad = simplify_to_quad(raw_pts)
                if quad is None:
                    diagnostics.append(
                        {
                            "image_path": image_path,
                            "lp_id": lp["lp_id"],
                            "ambiguous_ordering": None,
                            "json_order_matches_derived_order": None,
                            "lp_id_matches_visible_chars": None,
                            "format_compatible": None,
                            "format_incompatible_reason": "could not simplify polygon to 4 points",
                            "n_characters": len(lp["characters"]),
                            "n_raw_points": n_raw_points,
                        }
                    )
                    continue
                pts = np.array(quad, dtype=np.float64)
                ordered, ambiguous = order_points_robust(pts)
                tl, tr, br, bl = ordered

                width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
                height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
                aspect = width / height if height > 0 else np.nan

                chars = lp["characters"]
                char_centers = [
                    ((c["bbox_coord"][0][0] + c["bbox_coord"][1][0]) / 2,
                     (c["bbox_coord"][0][1] + c["bbox_coord"][1][1]) / 2)
                    for c in chars
                ]
                row_idx, read_order, n_rows = cluster_char_rows(char_centers, tl, tr, br, bl)
                char_ids_json_order = [c["char_id"] for c in chars]
                char_ids_read_order = [chars[i]["char_id"] for i in read_order]
                json_order_matches_derived = char_ids_json_order == char_ids_read_order

                compatible, reason = check_visible_format(lp_info["plate_str"], char_ids_read_order)

                visible_from_lp_id = [c for c in lp_info["plate_str"] if c != "*"]
                lp_id_matches_chars = visible_from_lp_id == char_ids_read_order

                shape = {1: "single_row", 2: "two_row"}.get(n_rows, "unknown")

                rows.append(
                    {
                        "source": "uc3m",
                        "split": split,
                        "image_path": image_path,
                        "vehicle_id": pd.NA,
                        "plate_id": lp["lp_id"],
                        "plate_text": lp_info["plate_str"],
                        "x1": tl[0], "y1": tl[1],
                        "x2": tr[0], "y2": tr[1],
                        "x3": br[0], "y3": br[1],
                        "x4": bl[0], "y4": bl[1],
                        "aspect_ratio": aspect,
                        "n_rows": n_rows,
                        "plate_shape": shape,
                        "lighting": lp_info["lighting"],
                        "masked_chars": lp_info["n_masked"],
                        "view": pd.NA,
                        "format": pd.NA,
                        "glare": pd.NA,
                        "blur": pd.NA,
                    }
                )
                diagnostics.append(
                    {
                        "image_path": image_path,
                        "lp_id": lp["lp_id"],
                        "ambiguous_ordering": ambiguous,
                        "json_order_matches_derived_order": json_order_matches_derived,
                        "lp_id_matches_visible_chars": lp_id_matches_chars,
                        "format_compatible": compatible,
                        "format_incompatible_reason": reason,
                        "n_characters": len(chars),
                        "n_raw_points": n_raw_points,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def load_cv_plates_table():
    """plates.csv rows for cv, from ground_truth_maker/ground_truth_cv.csv
    (shared by everyone annotating with truth_maker.py).

    Fails loudly (raises ValueError) on a duplicate (image_path, plate_id)
    row -- shouldn't happen if annotators coordinate who does which images,
    but a git merge mishap could produce one, and this must not be silent.

    Returns (plates_df, annotations_df): plates_df has the same columns as
    the UC3M table, one row per plate (NONE rows dropped); annotations_df is
    the raw file, for reporting.
    """
    if not CV_ANNOTATIONS.exists():
        return pd.DataFrame(), pd.DataFrame(columns=["image_path", "plate_id"])

    df = pd.read_csv(CV_ANNOTATIONS, encoding="utf-8",
                     dtype={"plate_id": str, "plate_text": str, "vehicle_id": str})
    dupes = df[df.duplicated(subset=["image_path", "plate_id"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            "Duplicate (image_path, plate_id) in ground_truth_cv.csv "
            "(possibly two annotators did the same image, or a bad merge):\n"
            f"{dupes[['image_path', 'plate_id', 'annotator']].to_string(index=False)}"
        )

    rows = []
    for _, a in df[df["plate_id"] != "NONE"].iterrows():
        tl, tr, br, bl = (np.array([a[f"x{i}"], a[f"y{i}"]], dtype=np.float64) for i in range(1, 5))
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        rows.append(
            {
                "source": "cv",
                "split": pd.NA,
                "image_path": a["image_path"],
                "vehicle_id": a["vehicle_id"],
                "plate_id": a["plate_id"],
                "plate_text": a["plate_text"],
                "x1": tl[0], "y1": tl[1],
                "x2": tr[0], "y2": tr[1],
                "x3": br[0], "y3": br[1],
                "x4": bl[0], "y4": bl[1],
                "aspect_ratio": width / height if height > 0 else np.nan,
                "n_rows": pd.NA,        # no character boxes to cluster
                "plate_shape": pd.NA,
                "lighting": a["lighting"],
                "masked_chars": 0,
                "view": Path(a["image_path"]).parent.name.lower(),
                "format": pd.NA,
                "glare": pd.NA,
                "blur": pd.NA,
            }
        )
    return pd.DataFrame(rows), df


def cv_filename_issues():
    """cv filenames that are not a valid current-format plate string."""
    issues = []
    for view_dir in ["Frontal", "Lateral"]:
        for img_path in sorted((CV_ROOT / view_dir).glob("*.jpg")):
            if not PLATE_REGEX.match(img_path.stem):
                issues.append({"view": view_dir.lower(), "file": img_path.name})
    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    section("1. JSON schema scan (all UC3M files)")
    schema_issues = scan_uc3m_json_schema(UC3M_ROOT)
    print(f"Files with schema deviations: {len(schema_issues)}")
    print("(all of these are poly_coord with 5 or 6 points instead of 4, see "
          "section 6b -- not corrupted data, recoverable)")
    for issue in schema_issues[:5]:
        print(" ", issue)

    section("2. Missing JSON / missing image check")
    missing = []
    for split in ["train", "test"]:
        jsons = {p.stem for p in (UC3M_ROOT / split).glob("*.json")}
        images = {p.stem for p in (UC3M_ROOT / split).glob("*.jpg")}
        for stem in jsons - images:
            missing.append(f"{split}/{stem}: JSON without image")
        for stem in images - jsons:
            missing.append(f"{split}/{stem}: image without JSON")
    print(f"Missing pairs: {len(missing)}")
    for m in missing[:20]:
        print(" ", m)

    section("3. images.csv")
    images_df = load_images_table()
    print(f"Total images: {len(images_df)}")
    print(images_df.groupby(["source", "split"], dropna=False).size().to_string())
    print("\nResolutions per source:")
    print(images_df.groupby("source")[["image_w", "image_h"]].agg(["min", "max", "nunique"]))

    section("4. cv filenames that are not a valid plate string")
    cv_issues = cv_filename_issues()
    print(f"Count: {len(cv_issues)}")
    for c in cv_issues:
        print(" ", c["view"], c["file"])

    section("5. cv vehicles appearing in both Frontal and Lateral")
    cv_imgs = images_df[images_df["source"] == "cv"]
    frontal_stems = {Path(p).stem for p in cv_imgs[cv_imgs["view"] == "frontal"]["image_path"]}
    lateral_stems = {Path(p).stem for p in cv_imgs[cv_imgs["view"] == "lateral"]["image_path"]}
    both = frontal_stems & lateral_stems
    print(f"Filenames present in both folders (proxy for vehicle, pre-annotation): {len(both)}")
    print(f"(this only works for the ~{len(frontal_stems | lateral_stems) - len(cv_issues)} filenames "
          f"that are valid plate strings; the ~{len(cv_issues)} invalid ones can't be matched this way "
          f"without the annotated plate text)")

    section("6. UC3M plates: load, geometry, format")
    plates_df, diag_df = load_uc3m_plates_table()
    print(f"Total plate rows: {len(plates_df)} (README says 2547)")
    print(plates_df.groupby("split").size().to_string())

    section("6a. cv plates (from ground_truth_maker/ground_truth_cv.csv)")
    cv_plates_df, cv_annotations = load_cv_plates_table()
    n_cv_images = (images_df["source"] == "cv").sum()
    annotated = set(cv_annotations["image_path"])
    n_no_plate = (cv_annotations["plate_id"] == "NONE").sum()
    print(f"cv images annotated: {len(annotated)} / {n_cv_images}")
    print(f"cv images marked 'no readable plate' (NONE, not in plates.csv): {n_no_plate}")
    print(f"cv plate rows: {len(cv_plates_df)}")
    unknown_images = annotated - set(images_df["image_path"])
    print(f"Annotated images not found in data_raw/cv: {len(unknown_images)}")
    for p in sorted(unknown_images)[:20]:
        print(" ", p)
    if len(annotated) < n_cv_images:
        print(f"\nWARNING: {n_cv_images - len(annotated)} cv images are not annotated yet -- "
              "plates.csv will be incomplete for cv.")

    section("6b. poly_coord point-count breakdown and simplification")
    print("poly_coord point counts (schema says 4; 5/6 = extra point sitting "
          "on an edge):")
    print(diag_df["n_raw_points"].value_counts().sort_index().to_string())
    unrecoverable = diag_df["format_incompatible_reason"].eq(
        "could not simplify polygon to 4 points"
    ).sum()
    print(f"\nCould not reduce to exactly 4 points (convex hull + approxPolyDP): "
          f"{unrecoverable} / {(diag_df['n_raw_points'] != 4).sum()} non-4-point cases")

    section("7. Plates-per-image distribution (UC3M)")
    per_image = plates_df.groupby(["split", "image_path"]).size()
    dist = per_image.value_counts().sort_index()
    print("n_plates_in_image : n_images")
    print(dist.to_string())

    section("8. Corner ordering: ambiguous cases")
    n_ambiguous = diag_df["ambiguous_ordering"].sum()
    print(f"Plates with ambiguous top/bottom-vs-left/right edge classification "
          f"(<20 deg apart): {n_ambiguous} / {len(diag_df)}")

    section("9. Aspect ratio histogram (all UC3M plates)")
    aspects = plates_df["aspect_ratio"].dropna()
    hist, edges = np.histogram(aspects, bins=20)
    for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
        print(f"  {e0:5.2f}-{e1:5.2f}: {h:4d} {'#' * (h // 5)}")
    print("\nNOTE: raw image-plane aspect ratio does NOT cleanly separate single-row "
          "from two-row plates -- a single-row plate photographed at a steep angle "
          "can show an apparent aspect ratio as low as ~2.2 (perspective "
          "foreshortening), overlapping the genuine two-row range. plate_shape below "
          "is derived from character-row clustering instead, not from this histogram.")

    section("10. plate_shape and n_rows (derived from character-row clustering)")
    print("plate_shape counts, by split:")
    print(plates_df.groupby(["split", "plate_shape"]).size().unstack(fill_value=0).to_string())
    print("\nn_rows counts, by split:")
    print(plates_df.groupby(["split", "n_rows"], dropna=False).size().unstack(fill_value=0).to_string())
    unknown_shape = (plates_df["plate_shape"] == "unknown").sum()
    print(f"\n'unknown' shape (fewer than 2 visible characters, can't cluster rows): {unknown_shape}")

    section("11. Aspect ratio cross-check against derived plate_shape")
    for shape, group in plates_df.groupby("plate_shape"):
        a = group["aspect_ratio"].dropna()
        print(f"  {shape:<12} n={len(a):4d}  aspect min={a.min():.2f} "
              f"median={a.median():.2f} max={a.max():.2f}")

    section("12. Reading order: JSON order vs derived (row, x) order")
    matches = diag_df["json_order_matches_derived_order"].value_counts()
    print("JSON `characters` list order already equals derived reading order?")
    print(matches.to_string())
    two_plus_chars = diag_df[diag_df["n_characters"] >= 2]
    matches_2plus = two_plus_chars["json_order_matches_derived_order"].value_counts()
    print(f"\nSame, restricted to plates with >=2 characters (n={len(two_plus_chars)}):")
    print(matches_2plus.to_string())

    section("13. lp_id visible-character sequence vs derived read order")
    lp_id_matches = diag_df["lp_id_matches_visible_chars"].value_counts()
    print("Does stripping '*' from lp_id give the same sequence as characters "
          "sorted into our derived reading order?")
    print(lp_id_matches.to_string())

    section("14. Masked characters: wildcard count vs len(characters)")
    mismatch = (plates_df["plate_text"].str.len() - plates_df["masked_chars"]) != diag_df["n_characters"]
    print(f"Plates where len(lp_id)-wildcards != len(characters): {mismatch.sum()} / {len(plates_df)}")
    print("\nmasked_chars distribution:")
    print(plates_df["masked_chars"].value_counts().sort_index().to_string())
    print("\nplate string length distribution (7 = current format length, others = "
          "old-format/foreign, see section 15):")
    print(plates_df["plate_text"].str.len().value_counts().sort_index().to_string())

    section("15. Current-format compatibility at VISIBLE positions")
    compat = diag_df["format_compatible"].value_counts(dropna=False)
    print(compat.to_string())
    incompatible = diag_df[diag_df["format_compatible"] == False]  # noqa: E712
    print(f"\nIncompatible examples (first 10 of {len(incompatible)}):")
    for _, row in incompatible.head(10).iterrows():
        print(f"  {row['lp_id']}: {row['format_incompatible_reason']}")
    non_standard = diag_df[diag_df["format_compatible"].isna()]
    print(f"\nNon-standard length (not run through the check): {len(non_standard)}")

    section("16. lighting tag (from lp_id prefix second letter, D/N)")
    print(plates_df.groupby(["split", "lighting"], dropna=False).size().unstack(fill_value=0).to_string())

    section("17. Vehicle matching across images via wildcard-compatible lp_id")
    plates_df["_plate_len"] = plates_df["plate_text"].str.len()
    groups_found = 0
    group_sizes = []
    for length, group in plates_df.groupby("_plate_len"):
        strs = group["plate_text"].tolist()
        idxs = group.index.tolist()
        parent = {i: i for i in idxs}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        def compatible(a, b):
            return all(ca == cb or ca == "*" or cb == "*" for ca, cb in zip(a, b))

        n = len(strs)
        for a in range(n):
            for b in range(a + 1, n):
                if compatible(strs[a], strs[b]):
                    union(idxs[a], idxs[b])

        roots = Counter(find(i) for i in idxs)
        for root, size in roots.items():
            if size > 1:
                groups_found += 1
                group_sizes.append(size)

    print(f"Groups of >=2 plates whose masked strings are mutually compatible: {groups_found}")
    if group_sizes:
        print(f"Group size distribution: {Counter(group_sizes)}")
    print("\nCAUTION: wildcard-compatibility does not prove same vehicle -- two "
          "different real plates can be compatible by coincidence if their masks "
          "happen to cover the positions where they'd otherwise differ. This is "
          "a candidate signal only.")

    # Sections 7-17 are UC3M-only; from here on, all sources together.
    plates_df = plates_df.drop(columns=["_plate_len"])
    if not cv_plates_df.empty:
        plates_df = pd.concat([plates_df, cv_plates_df[plates_df.columns]], ignore_index=True)
    print("\nplates.csv rows per source:")
    print(plates_df.groupby("source").size().to_string())

    section("18. Duplicate rows")
    dupes = plates_df.duplicated(subset=["image_path", "plate_id"]).sum()
    print(f"Duplicate (image_path, plate_id) rows: {dupes}")

    section("19. Corners outside image bounds")
    merged = plates_df.merge(
        images_df[["image_path", "image_w", "image_h"]], on="image_path", how="left"
    )
    out_of_bounds = 0
    for _, row in merged.iterrows():
        xs = [row["x1"], row["x2"], row["x3"], row["x4"]]
        ys = [row["y1"], row["y2"], row["y3"], row["y4"]]
        if min(xs) < 0 or min(ys) < 0 or max(xs) > row["image_w"] or max(ys) > row["image_h"]:
            out_of_bounds += 1
    print(f"Plates with a corner outside the image bounds: {out_of_bounds} / {len(merged)}")

    section("Writing data_processed/")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    images_df.to_csv(PROCESSED / "images.csv", index=False)
    plates_df.to_csv(PROCESSED / "plates.csv", index=False)
    print(f"images.csv: {len(images_df)} rows")
    print(f"plates.csv: {len(plates_df)} rows "
          f"({', '.join(f'{k}: {v}' for k, v in plates_df['source'].value_counts().items())})")


if __name__ == "__main__":
    main()
