"""读取主项目示例，评估手动预测框并保存结果。"""

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
        print(f"\n样本：{sample['sample_id']} | 来源：{sample['prediction_source']}")
        print(f"描述：{sample['description']}")
        print(f"真实框：{gt}\n预测框：{pred}")
        print(f"真实框面积：{metrics['ground_truth_area']} | 预测框面积：{metrics['prediction_area']}")
        print(f"交集：{metrics['intersection_area']} | 并集：{metrics['union_area']}")
        print(f"IoU：{metrics['iou']:.9f} | 定位正确：{metrics['is_correct']} (IoU > 0.5)")
        print(f"结果图片：{output}")
    (ROOT / "outputs").mkdir(exist_ok=True)
    result_path = ROOT / "outputs/evaluation_results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\n评估 JSON：{result_path}")


if __name__ == "__main__":
    main()
