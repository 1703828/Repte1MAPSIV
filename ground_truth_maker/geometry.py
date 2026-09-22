"""Corners, angles, IoU, rectification helpers."""
import cv2
import numpy as np


def simplify_to_quad(pts, eps_fractions=(0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18)):
    """Reduce a >4-point polygon to its 4 true corners.

    Some UC3M poly_coord entries carry 5 or 6 points: the 4 real corners plus
    1-2 extra points sitting almost exactly on an edge (annotation artifacts,
    not a different physical shape - verified visually on samples). Convex
    hull + increasing approxPolyDP epsilon collapses those extra collinear
    points; tested to recover the correct 4 corners on all 142 such cases in
    UC3M train+test.

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


def order_points_sumdiff(pts):
    """Classic sum/diff corner ordering. Fails early on elongated quadrilaterals
    rotated far from axis-aligned — see order_points_robust for those.

    pts: (4, 2) array of [x, y], any order/winding.
    Returns: (4, 2) float32 array in TL, TR, BR, BL order.
    """
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]  # BR: largest x+y

    diff = pts[:, 1] - pts[:, 0]  # y - x
    rect[1] = pts[np.argmin(diff)]  # TR: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # BL: largest y-x
    return rect


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

    Method: sort points cyclically around the centroid to find adjacency, split
    the 4 edges into two opposite pairs, classify the pair whose mean direction
    is closer to horizontal as top/bottom, then label by position within each
    edge (smaller y = top, smaller x = left).

    pts: (4, 2) array of [x, y], any order/winding.
    Returns: (ordered, ambiguous)
      ordered: (4, 2) float32 array in TL, TR, BR, BL order.
      ambiguous: True if the two edge pairs are within `ambiguous_threshold_deg`
        of each other in how close they are to horizontal — i.e. the
        top/bottom vs left/right choice is a near-tie (quadrilateral close to
        a rotation where sides are ~45 degrees from horizontal).
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
