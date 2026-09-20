"""Pillow 绘图；显示时将原图下移，为图例预留独立空间。"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .evaluation import validate_box


def draw_result(image_path, ground_truth_box, prediction_box, metrics,
                output_path, prediction_source):
    """保存带框图片；坐标始终相对于原图，不改变评估用坐标。"""
    gt = validate_box(ground_truth_box)
    pred = validate_box(prediction_box)
    if prediction_source not in ("simulated", "model"):
        raise ValueError("prediction_source 必须是 simulated 或 model")
    with Image.open(image_path) as original:
        scene = original.convert("RGB")
    header = 140
    canvas = Image.new("RGB", (max(640, scene.width), scene.height + header), "white")
    canvas.paste(scene, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    green, red = "#168044", "#d33232"
    label = "Simulated prediction" if prediction_source == "simulated" else "Model prediction"
    draw.text((18, 12), "Ground truth (green)", fill=green, font=font)
    draw.text((18, 40), f"{label} (red)", fill=red, font=font)
    verdict = "CORRECT" if metrics["is_correct"] else "INCORRECT"
    draw.text((18, 70), f"IoU = {metrics['iou']:.6f} | {verdict} | rule: IoU > 0.5", fill="black", font=font)
    note = "Manual boxes; no language understanding or object detection." if prediction_source == "simulated" else "Prediction from a model; see evaluation JSON for provenance."
    draw.text((18, 103), note, fill="#444444", font=ImageFont.load_default(size=15))
    for box, color in ((gt, green), (pred, red)):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1 + header, x2, y2 + header), outline=color, width=3)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
