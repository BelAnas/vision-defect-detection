# Evaluation report

Produced by `python -m defects.evaluate`. The model is the exported ONNX file, run with ONNX Runtime on CPU.
Thresholds were chosen on the 270 validation images. All numbers below are from the 270 test images (626 labelled defects), which were used for nothing else.

## Headline

| Metric | Value |
|---|---|
| mAP@0.5 | 0.747 (95% interval 0.720 to 0.782) |
| mAP@0.5:0.95 | 0.413 (95% interval 0.392 to 0.442) |
| Precision at the chosen thresholds | 0.645 |
| Recall at the chosen thresholds | 0.749 |
| Defects found / missed / false alarms | 469 / 157 / 258 |
| Latency per image | 20.7 ms median, 34.0 ms 95th percentile (Intel(R) Xeon(R) CPU E5-1620 v3 @ 3.50GHz) |

## Per defect type

| Defect type | Labelled | AP@0.5 | Threshold | Precision | Recall | Found | Missed | False alarms |
|---|---|---|---|---|---|---|---|---|
| crazing | 99 | 0.394 | 0.08 | 0.309 | 0.596 | 59 | 40 | 132 |
| inclusion | 150 | 0.847 | 0.26 | 0.728 | 0.787 | 118 | 32 | 44 |
| patches | 123 | 0.930 | 0.34 | 0.845 | 0.886 | 109 | 14 | 20 |
| pitted_surface | 67 | 0.837 | 0.12 | 0.781 | 0.746 | 50 | 17 | 14 |
| rolled-in_scale | 102 | 0.526 | 0.22 | 0.573 | 0.539 | 55 | 47 | 41 |
| scratches | 85 | 0.950 | 0.45 | 0.918 | 0.918 | 78 | 7 | 7 |

## Image-level outcome

Every test image contains at least one defect. Of 270 images: **268 rejected**, **0 sent to review**, **2 passed** (a pass here is a defective part let through).
The dataset has no defect-free images, so the false reject rate on good parts is not measured.

## Against the classical baseline

Finding a defect regardless of its type (class-agnostic AP@0.5):

| Method | AP@0.5 | 95% interval |
|---|---|---|
| OpenCV background subtraction, no training | 0.094 | 0.074 to 0.125 |
| Fine-tuned YOLO11n (ONNX) | 0.786 | 0.751 to 0.819 |

## Weak spots

The two weakest defect types are **crazing** (AP@0.5 0.394) and **rolled-in_scale** (AP@0.5 0.526). See `docs/confusion.png` for where the errors go.
