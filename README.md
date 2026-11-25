# Steel surface defect detection on a CPU

**Lee Ting Sen and Khalifa Alshamsi · MLR 512, Advanced Computer Vision**

This repository contains the code, trained detectors and measurements for the NEU-DET study. All **21 trained checkpoints** and **84 static OpenVINO graphs** are included. Five held-out images are bundled for immediate replay.

## Validate the submission

From the repository root in PowerShell, use Windows x64 and Python 3.12 (the reference version was 3.12.5).

~~~powershell
py -3.12 review.py unpack
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r cpu_study/requirements.txt
& $py review.py replay
& $py review.py verify
~~~

- **Unpack** checks SHA-256 and restores the graphs and saved measurements. It can resume an interrupted extraction.
- **Replay** runs fresh CPU inference on `inclusion_11`. Stock YOLO26n (A0) should find one inclusion and match the archived coordinates and score.
- **Verify** runs the tests, reproduces ten primary tables byte for byte, reopens the original exports and checks graph reuse, metrics, timings and threshold selection. The extended checks include the historical detectors, PConv dense size comparison and supervised comparison protocol. The saved checks run offline once dependencies are installed. [Verification counts](artifacts/study/report/verification.json) · [Independent metric checks](validation/research-metrics-check.json) · [Individual verification commands](cpu_study/README.md#reproduce-the-published-result)

## What the study found

| Comparison | Held-out result |
| --- | --- |
| PConv YOLO26n, dense head at 320 and 480 pixels | Mean macro recall rose from **37.93% to 47.56%**. In the same paired size experiment, pipeline p95 rose from **7.45 to 15.24 ms**. [By-seed results](artifacts/cpu-extension/dense-sizes/results.csv) |
| Four stock models at 480 pixels | YOLO26n had the lowest mean validation pipeline p95 (15.28 ms) and met the specified quality limits. Its test p95 was 14.58 ms. YOLOv8n had the highest test mAP50–95 (38.19%). [Validation selection](artifacts/cpu-extension/stock/selection.json) · [Test results](artifacts/cpu-extension/tables/test-families.csv) |
| Stock YOLO26n heads under the same validation false-box allowance | With each head’s own cutoff, dense achieved **46.38%** mean test macro recall and one-to-one achieved **41.63%**. [Head and policy analysis](cpu_study/README.md) |

The original stock A0 one-to-one result was **15.59 ms p95** at 480 pixels. The prospective PConv dense B0 candidate recovered **16/137 crazing** and **38/126 rolled-in-scale** objects. NEU-DET contains defect images. Clean-strip false rejection and continuous camera-to-controller performance require independent acquisitions. [Original results](artifacts/study/report/cells.csv) · [Per-class counts](artifacts/cpu-extension/feedback/classes.csv) · [Frozen supervised comparison](cpu_study/confirmation.json)

## Replay a specific model

A is stock YOLO26n, B adds partial convolution, C adds learned top-down fusion gates, D combines both changes and E is stock YOLOv10n. Each has seeds 0, 1 and 2. Their original OpenVINO graphs support one-to-one at 256, 320, 480 and 640 pixels and dense with NMS at 480. B also has a dense 320 graph for each seed. Original anchor-based YOLOv5s and YOLOv8n each have three trained checkpoints and three dense 480 graphs. Each graph is stored by checkpoint, precision, head and size. The same 480 one-to-one graph serves both original comparisons.

~~~powershell
& $py review.py replay --run E2 --head dense --side 480
& $py review.py replay --run D1 --runtime pytorch --head one --side 480
& $py -m cpu_study.review replay --family B --seed 0 --head dense --side 320 --policy global --image crazing_201
& $py -m cpu_study.review replay --family B --seed 0 --head dense --side 480 --policy global --image crazing_201
~~~

For the last two commands, the expected matches are **0/2 at 320** and **1/2 at 480**. Use `--save frame.png` to export the annotated comparison. The original all-model replay is `& $py review.py replay --all`. [Saved replay receipt](validation/inference-replay.json) · [Additional model and policy commands](cpu_study/README.md)

## Open the interactive viewer

Install the images to browse all 360 held-out examples and test the actual viewer controls. The loader checks each of the 1,800 source images against its split manifest. An existing official download also works with `& $py review.py images --zip "D:\datasets\NEU-DET.zip"`. [Dataset and citation](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) · [Split manifest](data/split.json)

~~~powershell
& $py review.py images
& $py -m steel demo --split test --seeds 0 --image-id inclusion_11 --side 480 --head one
& $py -m scripts.verify_demo
~~~

Change the image, detector, head or input size to trigger inference. The viewer uses each model’s original shared cutoff. The two B0 replay commands above use the size-specific dense cutoffs from the paper. To save a viewer image without opening a window, add `--output .review/demo.png` to the `steel demo` command.

## Locate the evidence

| Material | Location |
| --- | --- |
| Weights, training logs and OpenVINO bundles | [Original runs](artifacts/runs) · [Original graph archives](bundles/openvino) · [Historical runs](artifacts/cpu-extension/runs) · [Additional graph archives](bundles/cpu-extension/graphs) |
| Figures, per-class measurements and timing data | [Measured exhibits](paper/README.md) · [Figures](artifacts/study/report/figures) · [Class metrics](artifacts/study/report/class_metrics.csv) · [Session archives](bundles/measurements) · [Extended results](artifacts/cpu-extension/tables) |
| Model code, protocol and tests | [Primary implementation](steel) · [CPU comparisons](cpu_study) · [Independent evaluators](scripts) · [Protocol tests](tests) · [Reference environment](artifacts/audit/environment.json) |
| Frozen study decisions | [Original selection](artifacts/study/freeze.json) · [Test admission](artifacts/study/heldout-start.json) · [Registration and earlier test exposure](artifacts/audit/design-extension.json) · [Paired differences](artifacts/cpu-extension/feedback/paired-effects.csv) |

To train a new original model with the full dataset and a CUDA GPU, use a separate output directory. The reference training ran for 200 epochs.

~~~powershell
& $py -m steel train --data data --artifacts .review/new-study --models A --seeds 0 --epochs 200 --device cuda
~~~

The study uses the NEU-DET dataset, Ultralytics model definitions and weights, YOLOv10 and partial convolution from FasterNet. [Dataset authors](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) · [Included Ultralytics license](licenses/Ultralytics-AGPL-3.0.txt) · [YOLOv10 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract.html) · [FasterNet paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html)
