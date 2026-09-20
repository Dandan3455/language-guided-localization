"""连续 xyxy 坐标：左上角为原点，宽高不加 1。仅使用标准库。"""

import math
from numbers import Real


def validate_box(box):
    """返回四个浮点坐标；错误格式、非有限数值和颠倒坐标均报错。"""
    try:
        values = tuple(box)
    except TypeError as exc:
        raise ValueError("框必须是包含四个数值的序列 [xmin, ymin, xmax, ymax]") from exc
    if len(values) != 4:
        raise ValueError("框必须恰好包含四个坐标")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
        raise ValueError("框坐标必须是实数，不能是字符串或布尔值")
    values = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("框坐标必须是有限数值，不能包含 NaN 或 Infinity")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("框坐标颠倒：必须满足 xmax >= xmin 且 ymax >= ymin")
    return values


def evaluate_boxes(ground_truth_box, prediction_box):
    """接收两个框，返回面积、IoU 和严格 IoU > 0.5 的定位判断。"""
    gx1, gy1, gx2, gy2 = validate_box(ground_truth_box)
    px1, py1, px2, py2 = validate_box(prediction_box)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    pred_area = (px2 - px1) * (py2 - py1)
    intersection_width = max(0.0, min(gx2, px2) - max(gx1, px1))
    intersection_height = max(0.0, min(gy2, py2) - max(gy1, py1))
    intersection = intersection_width * intersection_height
    union = gt_area + pred_area - intersection
    if not all(math.isfinite(v) for v in (gt_area, pred_area, intersection, union)):
        raise ValueError("坐标过大导致面积计算溢出，请缩小坐标范围")
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
    """独立的 IoU 接口，不依赖文件或预测来源。"""
    return evaluate_boxes(box_a, box_b)["iou"]


def run_checks():
    box = [0, 0, 10, 10]
    assert calculate_iou(box, box) == 1.0
    assert calculate_iou(box, [20, 20, 30, 30]) == 0.0
    # 手算：交集 5*10=50，并集 100+100-50=150，IoU=1/3。
    assert math.isclose(calculate_iou(box, [5, 0, 15, 10]), 1 / 3)
    assert calculate_iou(box, [10, 0, 20, 10]) == 0.0  # 仅接触边界
    assert calculate_iou([0, 0, 0, 10], box) == 0.0
    assert calculate_iou(box, [0, 0, 10, 0]) == 0.0
    assert calculate_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0
    assert not evaluate_boxes(box, [0, 0, 5, 10])["is_correct"]  # 恰好 0.5 不通过
    for invalid in ([10, 0, 0, 10], [0, 10, 10, 0],
                    [0, 0, float("nan"), 10], [0, 0, float("inf"), 10],
                    [0, 0, float("-inf"), 10], [0, 0, 10], [0, 0, "10", 10]):
        for a, b in ((invalid, box), (box, invalid)):
            try:
                calculate_iou(a, b)
            except ValueError:
                pass
            else:
                raise AssertionError(f"非法框未报错：{invalid}")
    print("全部评估断言通过。")


if __name__ == "__main__":
    run_checks()
