# CPU baselines and head-specific alarm thresholds

This experiment compares stock YOLO26n, YOLOv10n, YOLOv8n and original anchor-based YOLOv5s on the same grouped NEU-DET partition. The historical models each use three 200-epoch, 480-pixel runs. The four input sizes and the PConv-by-fusion experiment use the original A–E checkpoints.

Table IV of Lee Ting Sen and Khalifa Alshamsi's survey uses original YOLOv5s in three of its seven core comparisons and YOLOv8n in three more. Earlier work had already timed YOLOv5 on a CPU for steel inspection. This study measures which defects each head recovers at a common validation false-box allowance and includes filtering and NMS in the CPU timer. [Earlier CPU measurements, Fig. 10](https://digibuo.uniovi.es/dspace/bitstream/handle/10651/64510/ias2021-cnn.pdf?isAllowed=y&sequence=1)

Each 480-pixel path chooses its own confidence cutoff on the same 360 defective validation images. The cutoff maximizes the mean of the six class recalls at IoU 0.50, with at most 36 false boxes. Equal recall favors fewer false boxes, then the higher cutoff. Score ties stay together. False boxes per defective image count unmatched detections. NEU-DET contains zero clean exposures.

## Reproduce the published result

From the repository root on Windows x64 with Python 3.12, unpack the signed evidence and install the reference environment. Use the existing virtual environment if you have already followed the faculty README.

```powershell
py -3.12 review.py unpack
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r cpu_study\requirements.txt
.\.venv\Scripts\python.exe -m cpu_study.review verify
.\.venv\Scripts\python.exe -m steel report --tables-only --output .review\primary-tables
.\.venv\Scripts\python.exe -m cpu_study.verify_policies
.\.venv\Scripts\python.exe -m cpu_study.policy_timing verify
.\.venv\Scripts\python.exe -m cpu_study.feedback_audit --verify
.\.venv\Scripts\python.exe -m cpu_study.dense_sizes verify
.\.venv\Scripts\python.exe -m cpu_study.confirmation verify
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

### Joint head selection, class thresholds and input size

The follow-up compared 12 architecture/head options at 480. Each path used its own global validation threshold. The primary rule minimizes validation p95 among paths that meet the mAP, six-class recall and false-box limits. A recall-first ranking uses exactly the same eligible candidates. The original stock-family comparison, fixed-cutoff size sweep and factorial experiment have their own selection rules.

Six class thresholds share one allowance of 36 false boxes. One policy maximizes the mean class recall. The second first protects the lowest recall across classes. The code fits both policies for all 36 OpenVINO checkpoint/head combinations at 480 and fits a global threshold independently at every one of the 60 one-to-one graph sizes. The output counts matched objects, images flagged by any box, and images with a correct localization. The test set contains 360 annotated defective images and zero clean images.

Run these commands after unpacking the repository. The verifier rebuilds the validation decisions and 306 policy readouts in a temporary directory, checks both class objectives independently and compares the result with the distributed tables. The timing verifier reconstructs the extra 108,000 test calls from the archived observations.

```powershell
.\.venv\Scripts\python.exe -m cpu_study.verify_policies
.\.venv\Scripts\python.exe -m cpu_study.policy_timing verify
.\.venv\Scripts\python.exe -m cpu_study.review replay --family B --seed 0 --head dense --side 480 --policy global --image inclusion_11
.\.venv\Scripts\python.exe -m cpu_study.review replay --family B --head dense --image crazing_15 --policy class_minimum
```

The timing subset includes all four stock one-to-one sizes and three threshold policies at 480 for stock one-to-one, stock dense and the validation-selected PConv dense path. Every reported time includes live score filtering and any class-wise NMS. To repeat that rotation, restore the images with `review.py images` and run `python -m cpu_study.policy_timing collect --output artifacts/policy-repeat` on AC power.

[Validation policy and cutoffs](../artifacts/cpu-extension/policies/freeze.json) · [Eligibility at four tolerances](../artifacts/cpu-extension/policies/recall-tolerance.csv) · [Box and image counts](../artifacts/cpu-extension/policies/operating-results.csv) · [Re-timed thresholds](../artifacts/cpu-extension/policy-timing/timing.csv) · [Architecture diagram](../paper/generated/architecture.pdf)

### Interpret the demo and compare the two input sizes

The interactive viewer and original table rebuild use the **shared native one-to-one cutoff fitted at 480** for each checkpoint. For the prospective supervised comparison, use **B0, PConv YOLO26n, OpenVINO FP32, dense + NMS, 480, global cutoff 0.723431408405304**. The six-class minimum and mean objectives are alternative policies. On the bundled image `crazing_15` the prospective global policy finds zero of two objects. The class-minimum comparison finds one.

The dense-size experiment used the same three B checkpoints and a global cutoff fitted separately for each seed and size. At 320 and 480, mean test macro recall was **37.93% and 47.56%**. Test pipeline p95 in their own paired rotation was **7.45 and 15.24 ms**. The 480 graphs are reused. The 320 exports add three raw dense graphs, giving 84 graphs across the repository. The separate first comparison of B dense against stock one-to-one favored B at 12.99 versus 14.58 ms. The repeat favored stock at 13.41 versus 13.88 ms. [Size results](../artifacts/cpu-extension/dense-sizes/results.csv) · [Paired sessions and rotations](../artifacts/cpu-extension/feedback/timing-summary.csv)

```powershell
.\.venv\Scripts\python.exe -m cpu_study.review replay --family B --seed 0 --head dense --side 320 --policy global --image crazing_15
.\.venv\Scripts\python.exe -m cpu_study.review replay --family B --seed 0 --head dense --side 480 --policy global --image crazing_15
```

To collect a fresh two-size rotation with all images installed, run `python -m cpu_study.dense_sizes collect --output artifacts/dense-size-repeat` on AC power. Use `verify --output artifacts/dense-size-repeat` in the same module to reconstruct its 36 session percentiles. Each session has 200 warm-ups and 1,000 recorded calls. The later studies reuse an inspected test split. [Six-class results](../artifacts/cpu-extension/feedback/classes.csv) · [Frame counts](../artifacts/cpu-extension/feedback/frames.csv) · [Paired quality intervals](../artifacts/cpu-extension/feedback/paired-effects.csv) · [Validation changes by seed](../artifacts/cpu-extension/feedback/validation-tolerance.csv)

### Frozen policy for independent confirmation

The coursework proposal sets recall floors of 90% for both crazing and rolled-in scale, a ceiling of 1% rejected clean strips over one predetermined 100-frame window per strip, and a 20 ms capture-to-controller p95 deadline. These numerical limits are illustrative. The team has no independent steel cohort, plant-specified limits or controller timestamps. The repository supplies a frozen policy for a supervised comparison. [Policy and endpoint definitions](confirmation.json)

With newly acquired data, `python -m cpu_study.confirmation evaluate --manifest PATH` scores that policy without fitting. The JSON manifest records `policy_sha256` printed by the verifier, `acquisition_source`, `independently_acquired` set to true, and a `frames` array. Each frame supplies a unique `id`, `acquisition_group` (true coil or acquisition ID), `strip_id`, zero-based `frame_index`, `clean_strip` (whole-strip adjudication), relative `image` path, and `image_sha` (SHA-256). Supply `classes` as zero-based IDs in the frozen six-class order and `boxes` as original-image [left, top, right, bottom] pixel edges. A clean exposure has two empty arrays. Supply the first 100 consecutive nonoverlapping exposures for each adjudicated clean strip. Synchronized `capture_ns`, `controller_ack_ns` and boolean `controller_alarm` on every frame enable the line-latency endpoint. Missing endpoint evidence is reported as null.

### Original 480-pixel comparison

To time the saved checkpoints at their operating cutoffs again, run the following from the repository root with the laptop plugged in. The separate output directory keeps the new observations together. A subsequent invocation resumes completed sessions.

```powershell
.\.venv\Scripts\python.exe review.py images
.\.venv\Scripts\python.exe -m cpu_study.run --output artifacts/cpu-repeat --trained-from artifacts/cpu-extension
```

For a fresh training experiment, omit `--trained-from` and choose a new output directory. The historical detectors train for all three seeds. To repeat only the dual-head quality analysis, `python -m cpu_study.heads freeze` fits the original heads on validation and `python -m cpu_study.heads evaluate` recomputes the object-level and budget analyses. Upstream initialization and the original YOLOv5 source are checked by SHA-256 before use. [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/releases/tag/v7.0) · [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
