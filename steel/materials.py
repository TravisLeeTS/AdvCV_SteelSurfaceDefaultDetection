import csv
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from .common import CLASSES, read


def latex_audit(data, dest):
    from .data import load_rows
    lines=[r"\begin{tabular}{lrrr}",r"\hline",r"Split & Images & Groups & Boxes \\",r"\hline"]
    for s in ("train","val","test"):
        rows=load_rows(data,s)
        lines.append(f"{s.capitalize()} & {len(rows)} & {len(set(r['group'] for r in rows))} & {sum(len(r['boxes']) for r in rows)} "+r"\\")
    lines.extend([r"\hline",r"\end{tabular}"])
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"audit.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def selection_tables(sel):
    sections=["## Validation decision","",
              "Eligibility uses the paired-seed mean and seed 0. Timing columns show the seed-0 median p95 and its session range.",""]
    for title,records in [("Architecture at 480",sel["architecture"]),("Input size",sel["sizes"])]:
        if not records: continue
        sections += [f"### {title}","",
            "| Candidate | mAP change (pp), mean / seed 0 | Largest recall loss (pp), mean / seed 0 | FP/image, mean / seed 0 | Seed-0 p95 [range] (ms) | Eligibility |",
            "| --- | ---: | ---: | ---: | ---: | --- |"]
        for r in records:
            d=r["paired"]
            worst_mean=min(float(np.mean(d[c+"_recall"])) for c in CLASSES)
            worst_zero=min(d[c+"_recall"][0] for c in CLASSES)
            checks="eligible" if r["eligible"] else ", ".join(r["failed"]).replace("_"," ")
            sections.append(f"| {r['model']} {r['side']} | {100*np.mean(d['map5095']):+.2f} / {100*d['map5095'][0]:+.2f} | "
                            f"{100*max(-worst_mean,0):.2f} / {100*max(-worst_zero,0):.2f} | {np.mean(r['fp_per_image']):.3f} / {r['fp_per_image'][0]:.3f} | "
                            f"{r['p95']:.2f} [{r['low']:.2f}, {r['high']:.2f}] | {checks} |")
        sections.append("")
    return sections


def bundle_overleaf(paper, destination):
    members=[paper/name for name in ("main.tex","references.bib","IEEEtran.cls","IEEEtran.bst","README.md","source-notes.md")]
    members.extend(sorted((paper/"reference").glob("*.tex")))
    members.extend(sorted(path for path in (paper/"generated").iterdir() if path.suffix in (".tex",".pdf",".png",".jpg")))
    with ZipFile(destination,"w") as archive:
        for path in members:
            entry=ZipInfo(path.relative_to(paper).as_posix())
            entry.compress_type=ZIP_DEFLATED
            archive.writestr(entry,path.read_bytes())
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Overleaf source archive failed its integrity check")


def fusion_readout(dest):
    with (dest/"fusion_intervals.csv").open(newline="",encoding="utf-8") as stream:
        results={(r["contrast"],r["metric"],r["seed"]):r for r in csv.DictReader(stream) if r["split"]=="test"}
    with (dest/"fusion_contrasts.csv").open(newline="",encoding="utf-8") as stream:
        latency={r["contrast"]:float(r["estimate"]) for r in csv.DictReader(stream)
                 if r["split"]=="test" and r["metric"]=="pipeline_p95_ms" and r["seed"]=="paired mean"}
    lines=["## Does fusion help more with PConv?","",
           "At 480 on OpenVINO one-to-one, A is stock YOLO26n, B adds PConv, C adds two fusion gates and D combines the treatments. "
           "E is the separately trained YOLOv10n comparator. The registered measure is mAP50–95. "
           "Each seed uses the same split and six-class starting weights across the four YOLO26n treatments.","",
           "| Test contrast | Seed 0 (pp) | Seed 1 (pp) | Seed 2 (pp) | Mean (pp) | 95% group interval (pp) | Mean p95 change (ms) |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label in ("B-A","C-A","D-B","D-C","D-A","(D-B)-(C-A)"):
        m=results[label,"map5095","paired mean"]
        by_seed=" | ".join(f"{float(results[label,'map5095',str(k)]['estimate']):+.2f}" for k in (0,1,2))
        lines.append(f"| {label} | {by_seed} | {float(m['estimate']):+.2f} | "
                     f"[{float(m['low']):+.2f}, {float(m['high']):+.2f}] | {latency[label]:+.2f} |")
    inter=results["(D-B)-(C-A)","map5095","paired mean"]
    q=results["D-B","pitted_surface_recall","paired mean"]
    verdict=("The interaction interval includes zero. These images leave the direction of the interaction uncertain."
             if float(inter["low"])<=0<=float(inter["high"]) else
             "The interaction interval lies above zero." if float(inter["low"])>0 else
             "The interaction interval lies below zero.")
    lines+=["",f"{verdict} Adding fusion to the PConv backbone changed pitted-surface operating recall by "
            f"{float(q['estimate']):+.2f} pp (95% interval {float(q['low']):+.2f} to {float(q['high']):+.2f}). "
            "Each gate started with both input weights exactly one. Their learned weights were checked on 36 validation images.","",
            "The original A/B/E test results were available when the study was expanded. The primary interaction, gates and six additional runs "
            "were specified before C/D training. The expanded selection used validation, and all five architectures were timed in the same new sessions. "
            "[Registration](../../audit/design-extension.json) · [All factorial contrasts](fusion_contrasts.csv) · "
            "[Paired intervals](fusion_intervals.csv) · [Learned gate weights](../../audit/fusion-learned.json)",""]
    return lines


def package(data,out,dest):
    data,out,dest=Path(data),Path(out),Path(dest)
    paper=Path(__file__).resolve().parent.parent/"paper"
    generated=paper/"generated"
    latex_audit(data,generated)
    shutil.copyfile(out/"audit/training_geometry.jpg",generated/"training_geometry.jpg")
    for folder in ("figures","latex"):
        for p in (dest/folder).iterdir():
            if p.suffix in (".png",".pdf",".tex"):
                shutil.copyfile(p,generated/p.name)
    freeze=read(out/"study/freeze.json")
    sel=freeze["selection"]
    cases=read(dest/"figures/demo_cases.json")
    with (dest/"head_comparison.csv").open() as f: head=list(csv.DictReader(f))
    notes=["# Local demo notes","",f"The validation decision was **{sel['size_verdict']}**. "
           f"The displayed checkpoint is {cases['model']} seed 0 at {cases['side']} pixels. "
           f"The frozen confidence threshold is {cases['cutoff']:.4f}.",""]
    for tag,title in [("success","Show a matched image"),("miss","Show a missed object"),("false_positive","Show an extra prediction")]:
        if tag not in cases["cases"]: continue
        case=cases["cases"][tag]
        boxes="box" if case["predictions"]==1 else "boxes"
        notes+=[f"## {title}","",f"Open `{case['id']}` from the test image selector. "
                f"The detector matched {case['matched']}/{case['gt']} ground-truth annotations at IoU 0.50 "
                f"and returned {case['predictions']} {boxes}.","",
                f"`& $py -m steel demo --split test --image-id {case['id']} --side {cases['side']}`",""]
    changes_file=dest/"figures/size_change_objects.csv"
    if changes_file.exists():
        with changes_file.open(newline="",encoding="utf-8") as f:
            objects=list(csv.DictReader(f))
        notes+=["## Follow one annotation across sizes","",
                f"Use the one-to-one head and keep the cutoff at {cases['cutoff']:.4f}. A match requires the same class "
                "and IoU of at least 0.50. Select the image and switch the input size while watching the same annotation.","",
                "| Image | Defect | Side (px) | Score | IoU | Outcome |","| --- | --- | ---: | ---: | ---: | --- |"]
        for event in objects:
            score=f"{float(event['score']):.3f}" if event["score"] else "—"
            overlap=f"{float(event['iou']):.3f}" if event["iou"] else "—"
            notes.append(f"| {event['id']} | {event['defect'].replace('_',' ')} | {event['side']} | {score} | {overlap} | {event['outcome']} |")
        notes.append("")
    notes+=["## Explain the two timers","",
            "Switch between the two heads at 480. The live display times a single decoded image through final boxes. "
            "Use the three-session benchmark for the p95 comparison. The dense pipeline includes different head computation, candidate filtering and suppression. "
            "The isolated NMS timer starts with the same already-filtered candidates.","",
            "To repeat the comparison with the transferred YOLOv10n, start its seed-0 artifact and use the head selector.","",
            "`& $py -m steel demo --split test --models E --seeds 0 --side 480 --head dense`",""]
    for r in head:
        if r["split"]=="test" and r["seed"]=="0":
            notes.append(f"- {r['model_id']} on {r['runtime']} had a signed dense-minus-one p95 saving of "
                         f"{float(r['saving_ms']):+.2f} ms ({float(r['saving_percent']):+.1f}%). "
                         f"Suppression alone had p95 {float(r['dense_nms_p95_ms']):.3f} ms.")
    if "D" in freeze["models"] and "C" in freeze["models"]:
        notes+=["","## Show the fusion control","",
                "At 480 pixels, inspect the same annotated image with A, B, C and D at seed 0. "
                "C adds the two learned gates to stock A. D adds them to PConv B. E is the separate conference detector. "
                "Each model uses its own frozen validation cutoff. Use the paired differences across all three seeds to answer the interaction question.",
                "",f"`& $py -m steel demo --models D --seeds 0 --side 480 --image-id {cases['cases']['miss']['id']}`",
                "","[Test interaction and interval](fusion_intervals.csv) · [Architecture contrasts](fusion_contrasts.csv)"]
    (dest/"speaker_notes.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
    with (dest/"architecture_comparison.csv").open() as f: arch=list(csv.DictReader(f))
    b=[r for r in arch if r["model_id"]=="B" and r["split"]=="test"]
    facts=["# Held-out measurements","",
           f"Validation selected {sel['model'] or 'no eligible architecture'} and {sel['side'] or 'no eligible size'}. "
           f"The 480-pixel decision was **{sel['size_verdict']}**.","",
           *selection_tables(sel),*(fusion_readout(dest) if (dest/"fusion_intervals.csv").exists() else []),
           "## PConv at 480 on OpenVINO",""]
    with (dest/"paired_intervals.csv").open() as f: intervals=list(csv.DictReader(f))
    if b:
        changes=[float(r["p95_change_ms"]) for r in b]
        facts.append(f"Across the three matched seeds, the mean B-minus-A pipeline p95 change was {np.mean(changes):+.3f} ms "
                     f"(seed range {min(changes):+.3f} to {max(changes):+.3f} ms).")
        facts+=["","| Paired seed | A p95 (ms) | B p95 (ms) | B − A (ms) | B − A mAP50–95 (pp) |",
                "| ---: | ---: | ---: | ---: | ---: |"]
        for r in b:
            facts.append(f"| {r['seed']} | {float(r['reference_p95_ms']):.2f} | {float(r['model_p95_ms']):.2f} | "
                         f"{float(r['p95_change_ms']):+.3f} | {float(r['map5095_change_pp']):+.2f} |")
        facts+=["","| Defect | Mean paired recall change (pp) | 95% group interval (pp) |","| --- | ---: | ---: |"]
        for c in CLASSES:
            diff=np.mean([float(r[c+"_recall_change_pp"]) for r in b])
            ci=next(x for x in intervals if x["split"]=="test" and x["comparison"]=="B minus A at 480" and
                    x["seed"]=="paired mean" and x["metric"]==c+"_recall")
            facts.append(f"| {c.replace('_',' ')} | {diff:+.2f} | [{float(ci['low_pp']):+.2f}, {float(ci['high_pp']):+.2f}] |")
    with (dest/"cells.csv").open() as f: cells=list(csv.DictReader(f))
    facts+=["","## Input size on held-out images","",
            f"The table gives means across the three paired seeds for {cases['model']}. Each p95 is first summarized within seed.","",
            "| Side (px) | Crazing AP50 (%) | Rolled-in-scale AP50 (%) | mAP50–95 (%) | FP/image | Pipeline p95 (ms) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for side in (256,320,480,640):
        subset=[r for r in cells if r["split"]=="test" and r["runtime"]=="openvino" and r["head"]=="one"
                and r["model_id"]==cases["model"] and int(r["side"])==side]
        av=lambda c: float(np.mean([float(r[c]) for r in subset]))
        facts.append(f"| {side} | {100*av('crazing_ap50'):.2f} | {100*av('rolled_in_scale_ap50'):.2f} | "
                     f"{100*av('map5095'):.2f} | {av('fp_per_image'):.3f} | {av('pipeline_p95_ms'):.2f} |")
    facts+=["","## Head comparison at 480","",
            "Each seed summary uses the median of three signed, matched session differences. Ranges below span the three seeds.","",
            "| Detector | Runtime | Mean saving (ms) | Seed range (ms) | Mean saving (%) | Mean standalone NMS p95 (ms) |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for model in freeze["models"]:
        for rt in ("openvino","pytorch"):
            pairs=[r for r in head if r["split"]=="test" and r["model_id"]==model and r["runtime"]==rt]
            if not pairs: continue
            d=[float(p["saving_ms"]) for p in pairs]
            facts.append(f"| {model} | {rt} | {np.mean(d):+.3f} | {min(d):+.3f} to {max(d):+.3f} | "
                         f"{np.mean([float(p['saving_percent']) for p in pairs]):+.2f} | "
                         f"{np.mean([float(p['dense_nms_p95_ms']) for p in pairs]):.3f} |")
    facts+=["","The following held-out means use one-to-one minus dense for quality. AP changes are percentage points. "
            "FP/image is the observed count per annotated test image under each checkpoint's frozen cutoff.","",
            "| Model | Runtime | Dense / one p95 (ms) | Δ mAP50–95 (pp) | Δ crazing AP50 (pp) | Δ rolled-in-scale AP50 (pp) | Dense / one FP/image |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in freeze["models"]:
        for runtime in ("openvino","pytorch"):
            pairs=[r for r in head if r["split"]=="test" and r["model_id"]==model and r["runtime"]==runtime]
            if not pairs: continue
            def head_mean(field):
                return float(np.mean([float(r[field]) for r in pairs]))
            facts.append(f"| {model} | {runtime} | {head_mean('dense_p95_ms'):.2f} / {head_mean('one_p95_ms'):.2f} | "
                         f"{100*head_mean('one_minus_dense_map5095'):+.2f} | {100*head_mean('one_minus_dense_crazing_ap50'):+.2f} | "
                         f"{100*head_mean('one_minus_dense_rolled_in_scale_ap50'):+.2f} | "
                         f"{head_mean('dense_fp_per_image'):.3f} / {head_mean('one_fp_per_image'):.3f} |")
    facts+=["","For the OpenVINO seed-0 diagnostic, each head reads the same cached neck features. The one-to-one head includes "
            "decode and top-k. The dense head includes decode. The standalone NMS timer starts after score filtering and candidate preparation.","",
            "| Model | Dense head p95 (ms) | One-to-one head p95 (ms) | Standalone NMS p95 (ms) |",
            "| --- | ---: | ---: | ---: |"]
    for r in head:
        if r["split"]=="test" and r["model_id"] in ("A","E") and r["seed"]=="0" and r["runtime"]=="openvino":
            facts.append(f"| {r['model_id']} | {float(r['dense_head_p95_ms']):.3f} | "
                         f"{float(r['one_head_p95_ms']):.3f} | {float(r['dense_nms_p95_ms']):.3f} |")
    facts+=["","## Complexity and CPU runtime","",
            "The native active branch is counted at 480. Convolution and matrix products use two FLOPs per MAC. "
            "The count excludes normalization, activations, pooling, routing, box decoding, top-k, NMS and coordinate restoration. "
            "The p95 values below include the complete deployed pipeline. Each latency is averaged across the three seed summaries.","",
            "| Model | Head | Active parameters | GFLOPs | OpenVINO p95 (ms) | PyTorch p95 (ms) |",
            "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in freeze["models"]:
        for h in ("one","dense"):
            group=[c for c in cells if c["split"]=="test" and c["model_id"]==m and c["head"]==h and int(c["side"])==480]
            first=group[0]
            rt=[]
            for engine in ("openvino","pytorch"):
                times=[float(r["pipeline_p95_ms"]) for r in group if r["runtime"]==engine]
                rt.append(f"{np.mean(times):.2f}" if times else "outside design")
            facts.append(f"| {m} | {h} | {int(first['active_parameters']):,} | {int(first['flops_at_480'])/1e9:.3f} | {rt[0]} | {rt[1]} |")
    facts+=["","Within-checkpoint runtime differences and the A/E ordering for each seed and head are in the "
            "[runtime comparison](runtime_comparison.csv) and [runtime rankings](runtime_rankings.csv).","",
            "The six-class denominators and FP counts are in [class metrics](class_metrics.csv). "
            "The [session timings](timing_sessions.csv) give p50, p95, completed throughput and the raw observation file for each boundary. "
            "The [paired group intervals](paired_intervals.csv) resample images together within their inferred acquisition groups. "
            "Each interval describes its named metric. Validation eligibility uses the prespecified point estimates. "
            "Per-seed measurements, quality changes and raw timing references are in the [head comparison](head_comparison.csv) and "
            "[resolution sweep](resolution_sweep.csv).","",
            "The split contains annotated defect images. It supplies no clean-strip false-alarm rate. The size sweep reuses the same 200-pixel exposures. "
            "Timing describes the tested laptop, CPU affinity, FP32 runtimes and synchronous batch-one boundaries. "
            "The PConv comparison includes its pretrained-kernel transfer. The four-arm contrast estimates whether the change due to the two fusion gates differs between backbones."]
    facts+=["","## Relation to the seven-study survey","",
            "Ting and Alshamsi identify a three-size HEC experiment with one run per size and no measured latency (Section V-D). "
            "The four-size table above pairs each priority class's AP50 with measured CPU p95 on the same images. "
            "Section V-F identifies newer detector comparisons that leave the active NMS path unspecified. "
            "The head table records signed whole-path savings and the separate cost of suppression on matched candidates. "
            "The intervals condition on the one grouped split. Variation across acquisition partitions and production-line coverage "
            "remain questions in the survey's Table VII. See the [source notes](../../../paper/source-notes.md) for page references."]
    facts+=["","## Training runs","",
            "The best epoch uses native one-to-one validation mAP50–95 at 480. Times sum the recorded training and validation "
            "epochs. Peak memory is CUDA memory allocated by PyTorch.","",
            "| ID | Best epoch | Validation mAP50–95 (%) | Frozen cutoff | Epoch time (h) | Peak GPU memory (GiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    epoch_hours=0.0
    for run_id in freeze["checkpoints"]:
        with (out/"runs"/run_id/"epochs.csv").open(newline="",encoding="utf-8") as f:
            epochs=list(csv.DictReader(f))
        best=max(epochs,key=lambda row:float(row["map5095"]))
        hours=sum(float(row["seconds"]) for row in epochs)/3600
        epoch_hours+=hours
        peak=max(float(row["peak_vram_gib"]) for row in epochs)
        facts.append(f"| {run_id} | {best['epoch']} | {100*float(best['map5095']):.2f} | "
                     f"{freeze['cutoffs'][run_id]:.6f} | {hours:.3f} | {peak:.2f} |")
    cpu_seconds=0.0
    for cell in cells:
        path=out/"study"/cell["split"]/cell["source"]
        for session in range(3):
            cpu_seconds+=sum(stage["elapsed_s"] for stage in read(path/f"session-{session}.json")["summaries"])
            diagnostic=path/f"head-s{session}.json"
            if diagnostic.exists():
                cpu_seconds+=read(diagnostic)["summary"]["elapsed_s"]
    facts+=["",f"Recorded GPU training and validation epochs totaled {epoch_hours:.2f} hours. "
            f"On CPU, the recorded timed windows for the two primary boundaries, cached suppression and isolated heads "
            f"totaled {cpu_seconds/3600:.2f} hours. Warm-ups, image-level quality evaluation, compilation and the separate stage traces add wall time."]
    facts+=["","## Reproduction record","",
            f"The full study used {len(freeze['checkpoints'])} checkpoints and {len(cells)} validation and test cells. "
            f"The paired comparisons used {len(head)} matched head rows. See [coverage](coverage.csv) and the [frozen decision](../freeze.json).","",
            "The run IDs are "+", ".join(f"`{k}`" for k in freeze["checkpoints"])+". Each run used 200 epochs with batch 16 on the RTX 4080 Laptop GPU. "
            "The actual i9-13900HX system has 63.75 GiB of usable RAM. CPU measurements used four physical P-cores in Legion Balance Mode on AC.","",
            "The [runbook](../../../README.md) contains the exact PowerShell commands. "
            "The [training log for C and D](../../logs/fusion-training.log), [expanded benchmark log](../../logs/expanded-study.log), "
            "[hardware inventory](../../audit/environment.json) and [source manifest](../../audit/project-source.json) identify the experiment. "
            "The [export inventory](../export-inventory.json) records one directory per checkpoint, precision, head and size. "
            "Validation and test reuse those directories. The 480 one-to-one bundle serves both comparisons. "
            "Every dense cell applies per-class NMS to original decoded coordinates."]
    (dest/"results.md").write_text("\n".join(facts)+"\n",encoding="utf-8")
    bundle_overleaf(paper,dest/"overleaf-materials.zip")
