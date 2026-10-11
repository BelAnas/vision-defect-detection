# Model files

| File | What |
|---|---|
| `defects.onnx` | YOLO11n fine-tuned on NEU-DET, exported with `python -m defects.export` |
| `model_card.json` | Input and output format, class list, checksum of the ONNX file |
| `thresholds.json` | Confidence thresholds chosen on the validation images by `python -m defects.evaluate` |

## Licence of the model file

The code in this repository is MIT. **The model file is not.** It was trained with Ultralytics YOLO,
which is published under AGPL-3.0, and Ultralytics states that models trained with it carry the same
licence. `defects.onnx` is therefore provided under AGPL-3.0.

For a closed-source commercial product there are two clean routes: buy an Ultralytics Enterprise
licence, or retrain with a permissively licensed detector (for example YOLOX or RT-DETR, both
Apache-2.0). The serving code here reads any ONNX detector with the same output layout, and the
data preparation, metrics, evaluation and review screen do not depend on Ultralytics at all.
