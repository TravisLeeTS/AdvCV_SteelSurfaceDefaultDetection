# Steel surface defect detection on a CPU

**Lee Ting Sen · MLR 512, Advanced Computer Vision**

<!-- cpu-study-results -->
At 480 pixels on four performance cores, **YOLO26n** recorded the lowest mean validation pipeline p95 across three seeds (15.28 ms). It also met every prespecified mAP and per-class recall limit. Its held-out mean was **14.58 ms**, with **37.59% mAP50–95** and **41.63% macro recall**. YOLOv8n had the highest held-out mAP (38.19%). The compared stock paths are YOLO26n, YOLOv10n, YOLOv8n and original anchor-based YOLOv5s.

When each YOLO26n head chose its own threshold at a validation allowance of 36 false boxes per 360 images, the dense head achieved **46.38%** mean test macro recall and one-to-one achieved **41.63%**. Their fresh test pipeline p95 values were 13.87 and 14.58 ms, respectively. On validation the corresponding values were 16.39 and 15.28 ms. [CPU comparison and replay commands](cpu_study/README.md) · [Validation ranking](artifacts/cpu-extension/stock/selection.json) · [Test results](artifacts/cpu-extension/tables/test-families.csv)
<!-- /cpu-study-results -->

Can a lighter detector inspect hot-rolled steel faster while retaining defects that are hard to see? The original architecture experiment compares five detectors on NEU-DET, with three paired training seeds per detector. Validation selected stock YOLO26n (A), one-to-one detection, at 480 pixels. On 360 held-out images, A0 achieved **38.29% mAP50–95** and a **15.59 ms pipeline p95**. It detected 20 of 137 crazing annotations and 43 of 126 rolled-in-scale annotations at the frozen cutoff, with 37 false-positive boxes overall. The latency is the median of three session p95s on an Intel Core i9-13900HX using four CPU cores, batch one and OpenVINO FP32. [Measured exhibits (PDF)](artifacts/study/report/MLR512-experimental-materials.pdf) · [Per-configuration results](artifacts/study/report/cells.csv)

## Run the faculty check

In PowerShell, open the root of this checkout. The reference environment is Windows x64 with Python 3.12.5. Use Git or download the repository ZIP and extract it first. All trained files are in this repository.

```powershell
py -3.12 review.py unpack
py -3.12 -m venv .venv
$py = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $py -m pip install -r cpu_study/requirements.txt
& $py review.py replay
& $py review.py verify
```

- **Unpack** uses the Python standard library. It checks SHA-256 for the published files and restores each graph and measurement to its original relative path. You can rerun it after an interrupted extraction.
- **Replay** performs fresh CPU inference on the bundled held-out image `inclusion_11`. The default A0 model returns one inclusion box. It checks the class, coordinates and score against the stored operating detections.
- **Verify** runs the unit tests, rebuilds ten primary tables in a fresh directory and compares every byte. It reopens all 75 graphs, checks the reuse of the same 480-pixel one-to-one graph in both comparisons, recalculates latency percentiles, and independently evaluates all 174 measured cells with the COCO evaluator. The grouped factorial and recall interval checks are included. New validation receipts are written under `.review/` and `artifacts/audit/factorial-check.json`. [Original verification counts](artifacts/study/report/verification.json) · [Independent evaluator results](validation/research-metrics-check.json)

The software installation needs network access or an existing wheel collection. Once installed, all three checks above work offline. CPU timing reported in the paper comes from the recorded benchmark sessions. The value in the live viewer measures one frame on the current machine.

## Run any checkpoint or graph

| ID | Detector |
| --- | --- |
| A | Stock YOLO26n |
| B | YOLO26n with partial convolution |
| C | YOLO26n with two learned top-down fusion gates |
| D | YOLO26n with both changes |
| E | Stock YOLOv10n |

Each ID has training seeds 0, 1 and 2. Each checkpoint has five complete static FP32 graphs. The one-to-one head supports 256, 320, 480 and 640 pixels. The raw one-to-many head (`dense`) runs at 480 pixels, followed by classwise NMS in the inference code. [Protocol](protocol.json) · [Export inventory](artifacts/study/export-inventory.json)

```powershell
& $py review.py replay --run E2 --head dense --side 480
& $py review.py replay --run D1 --runtime pytorch --head one --side 480
& $py review.py replay --run B0 --side 320 --image crazing_10
& $py review.py replay --all
```

The last command compares fresh detections on the same image across all 75 graphs and every native checkpoint. It includes the published A/E native head controls. Where the study measured only an OpenVINO cell, the native run is checked against that run's paired OpenVINO detections. The comparison uses the study's class, geometry and score tolerances. It writes the actual boxes and comparison results to `.review/replay-test-inclusion_11.json`. The five bundled image IDs are `inclusion_11`, `crazing_10`, `crazing_122`, `crazing_15` and `crazing_201`.

On a separate checkout with only the five bundled images and a fresh installation of the pinned environment, all 96 configurations matched the archived operating boxes on `inclusion_11`. The receipt identifies each checkpoint, comparison source and detection. [Fresh inference results](validation/inference-replay.json)

## Open the interactive demo or use all images

The viewer lets you change the image, detector, head and input size. Install the full dataset once to browse all 360 held-out images or select a validation example. The loader checks the publisher archive and each of the 1,800 images against the frozen manifests. The paper used 1,080 training, 360 validation and 360 test images. [NEU-DET download and citation](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/) · [Split and archive hashes](data/split.json)

```powershell
& $py review.py images
& $py -m steel demo --split test --seeds 0 --image-id inclusion_11 --side 480 --head one
```

An existing publisher download also works offline. Substitute its path for the example below.

```powershell
& $py review.py images --zip "D:\datasets\NEU-DET.zip"
```

To save a two-panel prediction without a window, append `--output .review/demo.png` to the viewer command. Use `--seeds 1` or `--seeds 2` to browse other checkpoints. With the full images installed, `& $py -m scripts.verify_demo` exercises the actual viewer controls and checks each displayed detection. The screenshot and check receipt are saved in `.review/demo/`.

## Find the evidence

| Question | Files |
| --- | --- |
| What did the study decide before test evaluation? | [Frozen selection and thresholds](artifacts/study/freeze.json) · [Held-out admission](artifacts/study/heldout-start.json) · [Registered C/D extension and prior test exposure](artifacts/audit/design-extension.json) |
| Where are the 15 trained weights and training logs? | [Per-run directories](artifacts/runs) contain the original `best.pt`, training configuration, 200 epoch rows and weight-transfer records. [Training verification](artifacts/audit/training-verification.json) |
| Where are the actual exported models? | [One archive per checkpoint](bundles/openvino). After unpacking, find `artifacts/runs/<run>/exports/<checkpoint SHA>/FP32/<head>-<side>/`. Each graph has its XML, binary weights, manifest and parity record. |
| How can I inspect individual measurements? | [Raw measurement archives by split and detector](bundles/measurements). Unpacking creates `artifacts/study/{val,test}/<cell>/` with predictions, individual timing samples, identified image order and session records. [Bundle checksums](bundles/index.json) |
| How did head, size and the learned gates affect results? | [Head pairs](artifacts/study/report/head_comparison.csv) · [Resolution sweep](artifacts/study/report/resolution_sweep.csv) · [Factorial effects](artifacts/study/report/fusion_contrasts.csv) · [Grouped intervals](artifacts/study/report/fusion_intervals.csv) |
| Where are the figures and error cases? | [Figures](artifacts/study/report/figures) · [Per-class metrics](artifacts/study/report/class_metrics.csv) · [Matched annotations](artifacts/study/report/matched_objects.csv) · [Saved demo frame](artifacts/study/report/demo.png) |
| What code produced these outputs? | [Frozen implementation](steel) · [Independent evaluators](scripts) · [Protocol tests](tests) · [Hardware and software](artifacts/audit/environment.json) |

The evidence covers images containing labeled defects. False-positive boxes per image on this set measure a different quantity from a false-alarm rate on clean production strips. The C/D extension was registered after earlier A/B/E results had been seen. The registration records that exposure explicitly.

## Training entry point

For a new training run, install the full images and use a separate artifact directory so the published checkpoints remain available for comparison. The original training used an NVIDIA GPU, 200 epochs and the recorded PyTorch/CUDA environment. The first run retrieves the versioned pretrained initialization and verifies its checksum.

```powershell
& $py -m steel train --data data --artifacts .review/new-study --models A --seeds 0 --epochs 200 --device cuda
```

The five architectures use Ultralytics model definitions and weights. The upstream license text is included. The detector and partial-convolution methods are credited to their original authors. [Ultralytics license](https://www.ultralytics.com/license) · [Included AGPL text](licenses/Ultralytics-AGPL-3.0.txt) · [YOLOv10 paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c34ddd05eb089991f06f3c5dc36836e0-Abstract.html) · [FasterNet paper](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html)

The NEU-DET images and annotations are credited to the dataset authors at Northeastern University. The bundled images illustrate the measured examples. The full archive comes from the dataset publisher. [Dataset and requested citations](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/)
