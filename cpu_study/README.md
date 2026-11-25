# CPU baselines and head-specific alarm thresholds

This experiment compares stock YOLO26n, YOLOv10n, YOLOv8n and original anchor-based YOLOv5s on the same grouped NEU-DET partition. The historical models each use three 200-epoch, 480-pixel runs. The four input sizes and the PConv-by-fusion experiment use the original A–E checkpoints.

Each 480-pixel path chooses its own confidence cutoff on the same 360 validation images. The cutoff maximizes the mean of the six class recalls at IoU 0.50, with at most 36 false boxes. Equal recall favors fewer false boxes, then the higher cutoff. Score ties stay together. The test evaluation applies the frozen cutoffs. False boxes per image refer to annotated steel images. NEU-DET supplies no clean-strip exposure.

## Reproduce the published result

From the repository root on Windows x64 with Python 3.12, unpack the signed evidence and install the reference environment. Use the existing virtual environment if you have already followed the faculty README.

```powershell
py -3.12 review.py unpack
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r cpu_study\requirements.txt
.\.venv\Scripts\python.exe -m cpu_study.review verify
```

The verifier recalculates per-class matches from the saved low-score boxes, checks all 48 validation thresholds, reconstructs every published pipeline percentile from individual calls, and rechecks the validation recommendation. An independent COCO evaluation checks all 12 historical model/split results and their 72 class-count rows. The measurements cover both splits, three sessions per path, 200 warm-ups and 1,000 observations per session. This verification uses the saved measurements and needs no GPU or retraining. [Independent COCO results](../artifacts/cpu-extension/diagnostics/historical-coco.json)

## Run a saved detector

The repository includes the held-out image `inclusion_11`. Each command runs one forward pass and checks its final boxes against the archived predictions.

```powershell
.\.venv\Scripts\python.exe -m cpu_study.review replay --family YOLOv8n
.\.venv\Scripts\python.exe -m cpu_study.review replay --family YOLOv5s
.\.venv\Scripts\python.exe -m cpu_study.review replay --family YOLOv5s --runtime pytorch
.\.venv\Scripts\python.exe -m cpu_study.review replay --family YOLO26n --head dense
.\.venv\Scripts\python.exe -m cpu_study.review replay --family YOLO26n --head one
```

Use `--family YOLOv10n`, `--seed 1` or `--image crazing_15` to change the configuration. Use `--save example.png` to export annotated ground truth and live predictions side by side. A native PyTorch replay uses the registered native threshold when available, and otherwise compares the native checkpoint with the export at the OpenVINO cutoff. The release includes all six historical checkpoints and their six static OpenVINO graphs, alongside the original fifteen checkpoints and seventy-five graphs. A graph directory identifies its checkpoint, precision, head and resolution. The 480-pixel one-to-one graph is reused. Original YOLOv5s multiplies objectness by the class score. YOLOv8n emits class scores directly. The same external code applies class-wise NMS to both.

## Read the evidence

| File | Question |
| --- | --- |
| [Validation family ranking](../artifacts/cpu-extension/stock/selection.json) | Which family had the smallest measured tail latency, and which met the quality limits? |
| [Test family table](../artifacts/cpu-extension/tables/test-families.csv) | What were the held-out accuracy, false alarms, latency and nine-session range? |
| [Test head timings](../artifacts/cpu-extension/tables/test-heads.csv) | What did the live postprocessor cost at each selected cutoff? |
| [Operating points](../artifacts/cpu-extension/heads/operating_points.csv) | How many true and false boxes did each independently fitted head produce? |
| [Paired effects](../artifacts/cpu-extension/heads/paired_effects.csv) | Which classes gained recall, with what group-bootstrap uncertainty? |
| [Alarm-budget sensitivity](../artifacts/cpu-extension/heads/budget_sensitivity.pdf) | How did realized test false alarms and recall change with the validation allowance? |
| [Registered design](protocol.json) | Which baseline sizes, heads, constraints and timing conditions were specified? |

The stock recommendation uses only validation. Its quality limits allow at most a one-percentage-point loss in mAP50–95 and a two-point loss in any class recall relative to YOLO26n, on seed zero and the paired three-seed mean. Latency is decoded grayscale image to final boxes on four physical performance cores, batch one, CPU FP32. A checkpoint's reported p95 is the median of three session percentiles. Each seed has equal weight in the family table. The original A–E test results were already public when the additional comparisons were designed.

## Collect fresh measurements

To time the saved checkpoints at their operating cutoffs again, run the following from the repository root with the laptop plugged in. The separate output directory keeps the new observations together. A subsequent invocation resumes completed sessions.

```powershell
.\.venv\Scripts\python.exe review.py images
.\.venv\Scripts\python.exe -m cpu_study.run --output artifacts/cpu-repeat --trained-from artifacts/cpu-extension
```

For a fresh training experiment, omit `--trained-from` and choose a new output directory. The historical detectors train for all three seeds. To repeat only the dual-head quality analysis, `python -m cpu_study.heads freeze` fits the original heads on validation and `python -m cpu_study.heads evaluate` recomputes the object-level and budget analyses. Upstream initialization and the original YOLOv5 source are checked by SHA-256 before use. [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0) · [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
