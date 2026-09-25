"""Load example data, evaluate manually specified prediction boxes, and save results."""

import json
from pathlib import Path
from src.evaluation import evaluate_boxes
from src.visualization import draw_result

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "data/examples/samples.json").open(encoding="utf-8") as file:
        samples = json.load(file)
    results = []
    for sample in samples:
        gt, pred = sample["ground_truth_box"], sample["prediction_box"]
        metrics = evaluate_boxes(gt, pred)
        output = ROOT / "outputs" / f"{sample['sample_id']}_result.png"
        draw_result(ROOT / sample["image_path"], gt, pred, metrics,
                    output, sample["prediction_source"])
        result = {**sample, **metrics, "visualization_path": output.relative_to(ROOT).as_posix()}
        results.append(result)
        print(f"\nSample: {sample['sample_id']} | Source: {sample['prediction_source']}")
        print(f"Description: {sample['description']}")
        print(f"Ground truth: {gt}\nPrediction: {pred}")
        print(f"Ground truth area: {metrics['ground_truth_area']} | Prediction area: {metrics['prediction_area']}")
        print(f"Intersection: {metrics['intersection_area']} | Union: {metrics['union_area']}")
        print(f"IoU: {metrics['iou']:.9f} | Correct localization: {metrics['is_correct']} (IoU > 0.5)")
        print(f"Result image: {output}")
    (ROOT / "outputs").mkdir(exist_ok=True)
    result_path = ROOT / "outputs/evaluation_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nEvaluation JSON: {result_path}")


if __name__ == "__main__":
    main()
