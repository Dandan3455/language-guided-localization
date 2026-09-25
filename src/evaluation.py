"""Continuous xyxy coordinates with a top-left origin; no +1 in dimensions. Standard library only."""

import math
from numbers import Real


def validate_box(box):
    """Return four float coordinates; reject invalid formats, nonfinite values, and reversed bounds."""
    try:
        values = tuple(box)
    except TypeError as exc:
        raise ValueError("Box must be a sequence of four numbers [xmin, ymin, xmax, ymax]") from exc
    if len(values) != 4:
        raise ValueError("Box must contain exactly four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise ValueError("Box coordinates must be real numbers, not strings or booleans")
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Box coordinates must be finite, without NaN or Infinity")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("Reversed box coordinates: require xmax >= xmin and ymax >= ymin")
    return values


def evaluate_boxes(ground_truth_box, prediction_box):
    """Return box areas, IoU, and localization correctness using strictly IoU > 0.5."""
    gx1, gy1, gx2, gy2 = validate_box(ground_truth_box)
    px1, py1, px2, py2 = validate_box(prediction_box)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    pred_area = (px2 - px1) * (py2 - py1)
    intersection_width = max(0.0, min(gx2, px2) - max(gx1, px1))
    intersection_height = max(0.0, min(gy2, py2) - max(gy1, py1))
    intersection = intersection_width * intersection_height
    union = gt_area + pred_area - intersection
    if not all(math.isfinite(v) for v in (gt_area, pred_area, intersection, union)):
        raise ValueError("Coordinates caused area overflow; use a smaller coordinate range")
    iou = intersection / union if gt_area > 0 and pred_area > 0 and union > 0 else 0.0
    return {
        "ground_truth_area": gt_area,
        "prediction_area": pred_area,
        "intersection_area": intersection,
        "union_area": union,
        "iou": iou,
        "criterion": "IoU > 0.5",
        "is_correct": iou > 0.5,
    }


def calculate_iou(box_a, box_b):
    """Compute IoU independently of files or prediction sources."""
    return evaluate_boxes(box_a, box_b)["iou"]


def run_checks():
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == 1.0
    assert calculate_iou(box, [20, 20, 30, 30]) == 0.0
    # By hand: intersection = 5*10 = 50; union = 100+100-50 = 150; IoU = 1/3.
    assert math.isclose(calculate_iou(box, [5, 0, 15, 10]), 1 / 3)
    assert calculate_iou(box, [10, 0, 20, 10]) == 0.0  # Boxes only touch at the boundary.
    assert calculate_iou([0, 0, 0, 10], box) == 0.0
    assert calculate_iou(box, [0, 0, 10, 0]) == 0.0
    assert calculate_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert not evaluate_boxes(box, [0, 0, 5, 10])["is_correct"]  # Exactly 0.5 does not pass.
    for invalid in ([10, 0, 0, 10], [0, 10, 10, 0],
                    [0, 0, float("nan"), 10], [0, 0, float("inf"), 10],
                    [0, 0, float("-inf"), 10], [0, 0, 10], [0, 0, "10", 10]):
        for a, b in ((invalid, box), (box, invalid)):
            try:
                calculate_iou(a, b)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid box did not raise an error: {invalid}")
    print("All evaluation checks passed.")


if __name__ == "__main__":
    run_checks()
