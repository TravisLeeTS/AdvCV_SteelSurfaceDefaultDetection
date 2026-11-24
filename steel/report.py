import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import CLASSES, MODELS, SIDES, digest, matrix, put, read, sha, source_hash, table
from .bench import name
from .factorial import factorial_rows

PRIORITY = ("crazing", "rolled_in_scale")


def values(path):
    meta = read(path/"cell.json")
    metric = read(path/"quality.json")["metrics"]
    records = [read(path/f"session-{i}.json") for i in range(meta["sessions"])]
    session_rows = [[dict(item,invocation_order=record["invocation_order"]) for item in record["summaries"]] for record in records]
    sessions = {row["boundary"]:[next(r for r in sr if r["boundary"]==row["boundary"]) for sr in session_rows]
                for row in session_rows[0]}
    p95 = [r["p95_ms"] for r in sessions["pipeline"]]
    prepared_p95=[r["p95_ms"] for r in sessions["prepared"]]
    result = {k:meta[k] for k in ("split","model_id","seed","runtime","head","side","checkpoint_sha","artifact_sha","precision","cutoff")}
    result.update(export_bundle=meta.get("export_bundle"),xml_sha=meta.get("xml_sha"),bin_sha=meta.get("bin_sha"))
    result.update(map50=metric["map50"], map5095=metric["map5095"], fp_per_image=metric["fp_per_image"],
                  fp_budget_met=metric["fp_per_image"]<=0.10,
                  images=metric["images"], pipeline_p50_ms=float(np.median([r["p50_ms"] for r in sessions["pipeline"]])),
                  pipeline_p95_ms=float(np.median(p95)), p95_min_ms=min(p95), p95_max_ms=max(p95),
                  model_p50_ms=float(np.median([r["p50_ms"] for r in sessions["prepared"]])),
                  model_p95_ms=float(np.median(prepared_p95)),model_p95_min_ms=min(prepared_p95),model_p95_max_ms=max(prepared_p95),
                  pipeline_fps=sum(r["timed"] for r in sessions["pipeline"])/sum(r["elapsed_s"] for r in sessions["pipeline"]),
                  nms_p95_ms=float(np.median([r["p95_ms"] for r in sessions["nms"]])) if "nms" in sessions else None,
                  nms_p50_ms=float(np.median([r["p50_ms"] for r in sessions["nms"]])) if "nms" in sessions else None,
                  head_p95_ms=None,head_p50_ms=None,active_parameters=None,flops_at_480=None,source=path.name,
                  cell_dir=f"{meta['split']}/{path.name}",score_floor=meta["floor"],nms_iou=meta["nms_iou"],
                  final_cap=meta["cap"],pre_nms_cap=meta["pre_nms_cap"],sessions=meta["sessions"],
                  warmups=meta["warm"],timed_per_session=meta["timed"],
                  stage_trace=f"{meta['split']}/{path.name}/stage-trace.csv" if (path/"stage-trace.json").exists() else None)
    if (path/"head-s0.json").exists():
        head = [read(path/f"head-s{i}.json")["summary"] for i in range(meta["sessions"])]
        sessions["head"]=[dict(item,invocation_order=records[i]["invocation_order"]) for i,item in enumerate(head)]
        result.update(head_p95_ms=float(np.median([s["p95_ms"] for s in head])),head_p50_ms=float(np.median([s["p50_ms"] for s in head])))
    if (path/"complexity.json").exists():
        cost=read(path/"complexity.json")
        result.update(active_parameters=cost["active_parameters"], flops_at_480=cost["flops"])
    for c, m in metric["classes"].items():
        result.update({f"{c}_{k}":v for k,v in m.items()})
    return result,sessions


def collect(study, splits, models, seeds):
    rows, times, coverage = {}, {}, []
    for split in splits:
        for cell in matrix(models,seeds):
            path = study/split/name(cell)
            key = (split,*cell)
            state = "pending"
            if (path/"cell.json").exists():
                n = read(path/"cell.json")["sessions"]
                if (path/"quality.json").exists() and all((path/f"session-{i}.json").exists() for i in range(n)):
                    model,seed,runtime,head,side=cell
                    needs_head=(model in ("A","E") and seed==0 and runtime=="openvino" and side==480)
                    if not needs_head or all((path/f"head-s{i}.json").exists() for i in range(n)):
                        rows[key],times[key]=values(path)
                        state="measured" if study.name=="study" else "smoke-test"
            coverage.append(dict(split=split,cell=name(cell),status=state,source=str(path.relative_to(study))))
    return rows,times,coverage


def matched_heads(rows,times):
    pairs=[]
    for (s,m,k,rt,h,side), dense in rows.items():
        if h!="dense" or (s,m,k,rt,"one",side) not in rows: continue
        dense_key=(s,m,k,rt,"dense",side)
        one_key=(s,m,k,rt,"one",side)
        one=rows[one_key]
        d=np.array([r["p95_ms"] for r in times[dense_key]["pipeline"]])
        o=np.array([r["p95_ms"] for r in times[one_key]["pipeline"]])
        dif=d-o
        result=dict(split=s,model_id=m,seed=k,runtime=rt,side=side,precision="FP32",
                    checkpoint_sha=one["checkpoint_sha"],cutoff=one["cutoff"],
                    dense_p95_ms=dense["pipeline_p95_ms"],one_p95_ms=one["pipeline_p95_ms"],
                    saving_ms=float(np.median(dif)),saving_percent=float(np.median(100*dif/d)),
                    paired_saving_min_ms=float(dif.min()),paired_saving_max_ms=float(dif.max()),
                    dense_nms_p50_ms=dense["nms_p50_ms"],dense_nms_p95_ms=dense["nms_p95_ms"],
                    dense_head_p95_ms=dense["head_p95_ms"],one_head_p95_ms=one["head_p95_ms"],
                    dense_source=f"{s}/{dense['source']}",one_source=f"{s}/{one['source']}")
        result.update(dense_stage_trace=dense.get("stage_trace"),one_stage_trace=one.get("stage_trace"),
                      dense_artifact_sha=dense.get("artifact_sha"),one_artifact_sha=one.get("artifact_sha"),
                      dense_export_bundle=dense.get("export_bundle"),one_export_bundle=one.get("export_bundle"),
                      dense_nms_iou=dense.get("nms_iou"),score_floor=dense.get("score_floor"),final_cap=dense.get("final_cap"),
                      nms_session_observations=f"{s}/{dense['source']}/timings-s*.csv.gz")
        for metric in ("map5095",*(c+"_ap50" for c in PRIORITY),"fp_per_image",*(c+"_recall" for c in CLASSES)):
            result["dense_"+metric]=dense[metric]
            result["one_"+metric]=one[metric]
            result["one_minus_dense_"+metric]=one[metric]-dense[metric]
        pairs.append(result)
    return pairs


def deltas(rows,keys,base_keys,metrics):
    return {metric:[rows[k][metric]-rows[b][metric] for k,b in zip(keys,base_keys,strict=True)] for metric in metrics}


def decision(rows,models,seeds):
    rules={"map5095":0.01, **{c+"_recall":0.02 for c in CLASSES}}
    def eligible(m,side,bm,bs,limits):
        keys=[("val",m,k,"openvino","one",side) for k in seeds]
        ref=[("val",bm,k,"openvino","one",bs) for k in seeds]
        paired=deltas(rows,keys,ref,limits)
        failures=[]
        demo=list(seeds).index(0)
        for metric,limit in limits.items():
            if np.mean(paired[metric]) < -limit-1e-12: failures.append("mean "+metric)
            if paired[metric][demo] < -limit-1e-12: failures.append("seed 0 "+metric)
        fp=[rows[k]["fp_per_image"] for k in keys]
        if np.mean(fp)>0.10+1e-12: failures.append("mean FP/image")
        if fp[demo]>0.10+1e-12: failures.append("seed 0 FP/image")
        row=rows[keys[demo]]
        return dict(model=m,side=side,eligible=not failures,failed=failures,paired=paired,fp_per_image=fp,
                    p95=row["pipeline_p95_ms"],low=row["p95_min_ms"],high=row["p95_max_ms"])
    def contenders(cases):
        valid=sorted((c for c in cases if c["eligible"]),key=lambda c:c["p95"])
        if not valid: return [],None
        fastest=valid[0]
        return [c for c in valid if c["low"]<=fastest["high"] and fastest["low"]<=c["high"]],fastest
    arch=[eligible(m,480,"A",480,rules) for m in models]
    ties,fast=contenders(arch)
    if fast is None:
        return dict(model=None,side=None,architecture=arch,sizes=[],size_verdict="No architecture met all validation limits.")
    chosen=next((c for c in ties if c["model"]=="A"),fast)["model"]
    limits={**rules, **{c+"_ap50":0.02 for c in PRIORITY}}
    sizes=[eligible(chosen,side,chosen,480,limits) for side in SIDES]
    ties,fast=contenders(sizes)
    selected=min((c["side"] for c in ties),default=None)
    tied_sides=[c["side"] for c in ties]
    if selected is None: verdict="No size met all validation limits."
    elif 480 in tied_sides and len(tied_sides)>1: verdict="480 in observed timing tie"
    elif selected==480: verdict="480 supported"
    else: verdict="480 superseded"
    return dict(model=chosen,side=selected,architecture=arch,sizes=sizes,size_verdict=verdict,
                architecture_tie=[c["model"] for c in contenders(arch)[0]],size_tie=tied_sides)


def analysis_revision():
    return source_hash("report.py","bench.py","metrics.py","export.py","post.py","geometry.py","models.py",
                       "uncertainty.py","figures.py","materials.py","checks.py","demo.py","hardware.py","common.py","data.py",
                       "stages.py","ir.py","smoke.py","export_compatibility.json","__main__.py","factorial.py")


def numerical_revision():
    """Files governing validation selection, inference, paired measurements and uncertainty."""
    return source_hash("report.py","bench.py","metrics.py","export.py","post.py","geometry.py","models.py",
                       "uncertainty.py","hardware.py","common.py","data.py","stages.py","ir.py","smoke.py",
                       "export_compatibility.json","__main__.py","factorial.py")


def freeze(data,out,models,seeds):
    data,out=Path(data),Path(out)
    study=out/"study"
    if (study/"freeze.json").exists():
        verify_freeze(data,out,models,seeds)
        return read(study/"freeze.json")
    rows,_,coverage=collect(study,["val"],models,seeds)
    missing=[r["cell"] for r in coverage if r["status"]!="measured"]
    if missing: raise ValueError(f"Validation cells pending: {missing}")
    decision_record=decision(rows,models,seeds)
    runs=[f"{m}{k}" for m in models for k in seeds]
    evidence={}
    for k in rows:
        p=study/"val"/name(k[1:])
        for filename in ["cell.json","quality.json","stage-trace.json",*(f"session-{s}.json" for s in range(3))]:
            evidence[(p/filename).relative_to(study).as_posix()]=sha(p/filename)
    rec=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),models=list(models),seeds=list(seeds),split_sha=read(data/"split.json")["split_hash"],
             checkpoints={r:sha(out/"runs"/r/"best.pt") for r in runs},
             thresholds={r:sha(study/"thresholds"/(r+".json")) for r in runs},
             cutoffs={r:read(study/"thresholds"/(r+".json"))["cutoff"] for r in runs},
             analysis_revision=analysis_revision(),numerical_revision=numerical_revision(),
             validation_evidence=evidence,selection=decision_record,
             export_inventory_sha=sha(study/"export-inventory.json"),reopened_before_timing_sha=sha(study/"reopened-before-timing.json"),
             protocol_sha=sha(Path(__file__).resolve().parent.parent/"protocol.json"))
    if any(m in models for m in ("C","D")):
        receipts=[out/"audit"/f"fusion-{label}.json" for label in ("initial","learned")]
        receipts.extend(study/f"reopened-before-timing-{m}0.json" for m in models if m in ("C","D") and 0 in seeds)
        registration=out/"audit/design-extension.json"
        if registration.exists(): receipts.append(registration)
        rec["fusion_evidence"]={p.relative_to(out).as_posix():sha(p) for p in receipts}
    put(study/"freeze.json",rec)
    return rec


def verify_freeze(data,out,models,seeds):
    study=Path(out)/"study"
    rec=read(study/"freeze.json")
    summary=read(Path(data)/"split.json")
    revision_ok=(rec["numerical_revision"]==numerical_revision() if "numerical_revision" in rec else rec["analysis_revision"]==analysis_revision())
    if (rec["models"]!=list(models) or rec["seeds"]!=list(seeds) or not revision_ok
            or rec["split_sha"]!=summary["split_hash"] or rec["export_inventory_sha"]!=sha(study/"export-inventory.json")
            or rec["reopened_before_timing_sha"]!=sha(study/"reopened-before-timing.json")
            or rec["protocol_sha"]!=sha(Path(__file__).resolve().parent.parent/"protocol.json")):
        raise ValueError("Frozen protocol, split, design or analysis revision mismatch")
    for r,h in rec["checkpoints"].items():
        if sha(Path(out)/"runs"/r/"best.pt")!=h or sha(study/"thresholds"/(r+".json"))!=rec["thresholds"][r]:
            raise ValueError(f"Changed frozen run or threshold: {r}")
    for f,h in rec["validation_evidence"].items():
        if sha(study/f)!=h: raise ValueError(f"Changed validation evidence: {f}")
    for f,h in rec.get("fusion_evidence",{}).items():
        if sha(Path(out)/f)!=h: raise ValueError(f"Changed fusion evidence: {f}")


def tables(study,dest,models=MODELS,seeds=(0,1,2),splits=("val","test")):
    rows,times,coverage=collect(Path(study),splits,models,seeds)
    dest=Path(dest)
    table(dest/"coverage.csv",coverage)
    if not rows: return rows
    table(dest/"cells.csv",rows.values())
    session_table=[]
    for key,row in rows.items():
        for boundary,measurements in times[key].items():
            for measurement in measurements:
                session=measurement["session"]
                raw=f"head-s{session}.csv" if boundary=="head" else f"timings-s{session}.csv.gz"
                identity={k:row[k] for k in ("split","model_id","seed","checkpoint_sha","runtime","precision","head","side","cutoff","nms_iou")}
                session_table.append(dict(**identity,**measurement,raw_observations=f"{row['cell_dir']}/{raw}"))
    table(dest/"timing_sessions.csv",session_table)
    sweeps=[v for k,v in rows.items() if k[3:5]==("openvino","one")]
    if sweeps: table(dest/"resolution_sweep.csv",sweeps)
    pairs=matched_heads(rows,times)
    if pairs: table(dest/"head_comparison.csv",pairs)
    byclass=[]
    runtime=[]
    for key,r in rows.items():
        for c in CLASSES:
            byclass.append(dict(split=key[0],model_id=key[1],seed=key[2],runtime=key[3],head=key[4],
                                side=key[5],defect=c,**{m:r[c+"_"+m] for m in ("ap50","recall","gt","tp","fp","fn")},
                                cutoff=r["cutoff"],precision=r["precision"],checkpoint_sha=r["checkpoint_sha"],
                                source=r["source"],cell_dir=r["cell_dir"]))
        s,m,k,rt,h,side=key
        peer=(s,m,k,"pytorch",h,side)
        if rt=="openvino" and peer in rows:
            other=rows[peer]
            runtime.append(dict(split=s,model_id=m,seed=k,head=h,side=side,openvino_p95_ms=r["pipeline_p95_ms"],
                                pytorch_p95_ms=other["pipeline_p95_ms"],openvino_map5095=r["map5095"],
                                pytorch_map5095=other["map5095"],checkpoint_sha=r["checkpoint_sha"],precision=r["precision"],
                                openvino_source=r["cell_dir"],pytorch_source=other["cell_dir"]))
    table(dest/"class_metrics.csv",byclass)
    if runtime: table(dest/"runtime_comparison.csv",runtime)
    architecture=[]
    rankings=[]
    contrasts=[(m,"A") for m in models if m!="A"]
    contrasts.extend(("D",m) for m in ("B","C") if "D" in models and m in models)
    for split in splits:
        for k in seeds:
            for m,reference in contrasts:
                base=(split,reference,k,"openvino","one",480)
                key=(split,m,k,"openvino","one",480)
                if base not in rows or key not in rows: continue
                a=rows[base]
                other=rows[key]
                entry=dict(split=split,model_id=m,reference=reference,seed=k,side=480,runtime="openvino",head="one",
                           p95_change_ms=other["pipeline_p95_ms"]-a["pipeline_p95_ms"],model_p95_ms=other["pipeline_p95_ms"],
                           reference_p95_ms=a["pipeline_p95_ms"],map5095_change_pp=100*(other["map5095"]-a["map5095"]),
                           fp_per_image=other["fp_per_image"],fp_change=other["fp_per_image"]-a["fp_per_image"],
                           source=other["source"],reference_source=a["source"],precision=other["precision"],
                           checkpoint_sha=other["checkpoint_sha"],reference_checkpoint_sha=a["checkpoint_sha"],
                           cell_dir=other["cell_dir"],reference_cell_dir=a["cell_dir"])
                for c in CLASSES: entry[c+"_recall_change_pp"]=100*(other[c+"_recall"]-a[c+"_recall"])
                architecture.append(entry)
            for rt in ("openvino","pytorch"):
                for head in ("one","dense"):
                    ka,ke=(split,"A",k,rt,head,480),(split,"E",k,rt,head,480)
                    if ke not in rows or ka not in rows: continue
                    ar,er=rows[ka],rows[ke]
                    rankings.append(dict(split=split,seed=k,runtime=rt,head=head,side=480,precision=ar["precision"],
                        A_checkpoint_sha=ar["checkpoint_sha"],E_checkpoint_sha=er["checkpoint_sha"],
                        A_source=ar["cell_dir"],E_source=er["cell_dir"],
                        A_p95_ms=ar["pipeline_p95_ms"],E_p95_ms=er["pipeline_p95_ms"],
                        faster="A" if ar["pipeline_p95_ms"]<=er["pipeline_p95_ms"] else "E",
                        observed_ranges_overlap=ar["p95_min_ms"]<=er["p95_max_ms"] and er["p95_min_ms"]<=ar["p95_max_ms"]))
    if architecture: table(dest/"architecture_comparison.csv",architecture)
    if rankings: table(dest/"runtime_rankings.csv",rankings)
    factorial=[r for split in splits for r in factorial_rows(rows,split,seeds)]
    if factorial: table(dest/"fusion_contrasts.csv",factorial)
    return rows
