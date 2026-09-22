"""Evaluate one saved detection against one manual annotation (no model rerun)."""

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image
from src.evaluation import evaluate_boxes, validate_box
from src.visualization import draw_result

ROOT = Path(__file__).resolve().parent


def select_candidate(candidates, target):
    """Match complete label words; choose highest score, keeping first on ties."""
    words = target.lower().split()
    matches = []
    for candidate in candidates:
        validate_box(candidate["box"])
        score = candidate["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Candidate score must be a finite number")
        tokens = re.findall(r"\w+", candidate["label"].lower())
        if any(tokens[i:i + len(words)] == words for i in range(len(tokens))):
            matches.append(candidate)
    return max(matches, key=lambda item: item["score"], default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, default=ROOT / "data/annotations/desk_left_cup_001.json")
    parser.add_argument("--prediction", type=Path, default=ROOT / "outputs/detection/results.json")
    parser.add_argument("--target", default="cup")
    args = parser.parse_args()
    if not args.target.strip():
        parser.error("Target must not be empty")
    annotations = json.loads(args.annotation.read_text(encoding="utf-8-sig"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8-sig"))
    if len(annotations) != 1:
        raise ValueError("This entry point expects exactly one annotation")
    sample = annotations[0]
    image_path = (ROOT / sample["image_path"]).resolve()
    if image_path != (ROOT / prediction["image_path"]).resolve():
        raise ValueError("Annotation and prediction refer to different images")
    if prediction.get("prediction_source") != "model":
        raise ValueError("Expected real model predictions")
    if sample.get("annotation_source") != "manual":
        raise ValueError("Expected a manual annotation")
    gt = validate_box(sample["ground_truth_box"])
    with Image.open(image_path) as image:
        if image.size != (sample["image_width"], sample["image_height"]):
            raise ValueError("Annotation dimensions do not match the image")
        if not (0 <= gt[0] < gt[2] <= image.width and 0 <= gt[1] < gt[3] <= image.height):
            raise ValueError("Ground truth must have positive area inside the image")
    selected = select_candidate(prediction["candidates"], args.target)
    metrics = evaluate_boxes(gt, selected["box"]) if selected else {
        "iou": 0.0, "is_correct": False, "criterion": "IoU > 0.5"
    }
    output = ROOT / "outputs" / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", sample["sample_id"])
    result = {
        "annotation": sample, "prediction_run": prediction,
        "selection_rule": "highest score among labels containing target words; first on ties",
        "target": args.target, "selected_candidate": selected,
        "no_target_prediction": selected is None, "metrics": metrics,
        "visualization_path": None,
        "scope": "Single development example; not dataset accuracy or proof of spatial understanding.",
    }
    if selected:
        picture = output / f"{safe_id}_result.png"
        draw_result(image_path, gt, selected["box"], metrics, picture, "model")
        result["visualization_path"] = picture.relative_to(ROOT).as_posix()
    result_path = output / f"{safe_id}_evaluation.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Selected: {selected}")
    print(f"IoU: {metrics['iou']:.6f}; correct: {metrics['is_correct']}")
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
