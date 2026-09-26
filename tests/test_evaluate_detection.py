import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import evaluate_detection


class EvaluationWorkflowTests(unittest.TestCase):
    def test_annotation_is_required(self):
        with patch("sys.argv", ["evaluate_detection.py", "--prediction", "unused.json"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                evaluate_detection.main()
        self.assertEqual(error.exception.code, 2)

    def test_named_evaluation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20)).save(root / "image.png")
            annotation = root / "annotation.json"
            prediction = root / "prediction.json"
            annotation.write_text(json.dumps([{
                "sample_id": "right_cup", "image_path": "image.png",
                "annotation_source": "manual", "image_width": 20, "image_height": 20,
                "ground_truth_box": [10, 0, 20, 10],
            }]), encoding="utf-8")
            prediction.write_text(json.dumps({
                "run_name": "desk_right", "image_path": "image.png",
                "prediction_source": "model", "candidates": [
                    {"label": "cup", "score": 0.9, "box": [0, 0, 10, 10]},
                    {"label": "cup", "score": 0.8, "box": [10, 0, 20, 10]},
                ],
            }), encoding="utf-8")
            argv = ["evaluate_detection.py", "--annotation", str(annotation), "--prediction", str(prediction)]
            with patch.object(evaluate_detection, "ROOT", root), patch("sys.argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluate_detection.main()
                result_path = root / "outputs/evaluation/desk_right/right_cup_evaluation.json"
                original = result_path.read_bytes()
                result = json.loads(original)
                self.assertEqual(result["run_name"], "desk_right")
                self.assertEqual(result["metrics"]["iou"], 0)
                self.assertFalse(result["metrics"]["is_correct"])
                self.assertTrue((root / result["visualization_path"]).is_file())
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    evaluate_detection.main()
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(result_path.read_bytes(), original)
                with patch("sys.argv", argv + ["--run-name", "desk_right_02"]):
                    with contextlib.redirect_stdout(io.StringIO()):
                        evaluate_detection.main()
                self.assertTrue((root / "outputs/evaluation/desk_right_02/right_cup_evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()
