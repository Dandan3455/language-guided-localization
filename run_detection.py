"""Run pretrained Grounding DINO on one image; this is not an accuracy benchmark."""

import argparse
import json
import os
from pathlib import Path
from src.run_paths import create_run_directory

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--text", default="a cat. a remote control.")
    parser.add_argument("--run-name", help="Experiment name, e.g. desk_left; omit for an automatic unique name")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if not args.text.strip():
        parser.error("Text must not be empty")
    try:
        destination = create_run_directory(ROOT / "outputs" / "detection", args.run_name)
    except (ValueError, FileExistsError) as exc:
        parser.error(f"{exc}. Choose a new --run-name; existing results are never overwritten.")
    print(f"Experiment: {destination.name}\nPrompt: {args.text}\nOutput: {destination}", flush=True)

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf-home"))
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    cache = ROOT / "models" / "huggingface"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}; first run downloads weights.", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, cache_dir=cache, use_safetensors=True, disable_custom_kernels=True
    ).to(device).eval()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.3,
        target_sizes=[(image.height, image.width)],
    )[0]
    labels = result.get("text_labels", result.get("labels"))
    candidates = [
        {"box": box.tolist(), "score": score.item(), "label": str(label)}
        for box, score, label in zip(result["boxes"].cpu(), result["scores"].cpu(), labels)
    ]
    record = {
        "run_name": destination.name,
        "image_path": str(args.image.resolve()), "prompt": args.text,
        "prediction_source": "model", "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "device": device, "box_format": "original image pixel xyxy, unclipped",
        "box_threshold": 0.4, "text_threshold": 0.3,
        "candidates": candidates, "no_detection": not candidates,
        "note": "All candidates; no target selection or ground-truth evaluation.",
    }
    (destination / "results.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    draw = ImageDraw.Draw(image)
    for candidate in candidates:
        box = candidate["box"]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], max(0, box[1] - 12)),
                  f"{candidate['label']} {candidate['score']:.2f}", fill="red")
    image.save(destination / "result.png")
    print(f"Detected {len(candidates)} candidates. Results: {destination}")


if __name__ == "__main__":
    main()
