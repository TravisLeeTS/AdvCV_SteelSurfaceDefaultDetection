"""Paired group bootstrap. Integer image weights duplicate whole images within each draw."""
from pathlib import Path
import numpy as np

from .bench import read_predictions
from .common import CLASSES, put, read, table
from .data import load_rows
from .metrics import RECALL_GRID, events
from .factorial import factorial_rows
from .report import PRIORITY


def bootstrap_metrics(items, cutoff, draws):
    scores=np.concatenate([x[0] for x in items])
    order=np.argsort(-scores,kind="stable")
    scores=scores[order]
    ids=np.repeat(np.arange(len(items)),[len(x[0]) for x in items])[order]
    labels=np.concatenate([x[1] for x in items])[order]
    flags=np.concatenate([x[2] for x in items])[order]
    gt=np.stack([x[3] for x in items])
    out={k:np.zeros(len(draws)) for k in ("map5095",*(c+"_ap50" for c in PRIORITY),*(c+"_recall" for c in CLASSES))}
    denominators=draws@gt
    if np.any(denominators==0):
        raise ValueError("A bootstrap draw lacks a defect class")
    for c,defect in enumerate(CLASSES):
        selected=labels==c
        img,hit=ids[selected],flags[selected]
        operating=img[(scores[selected]>=cutoff) & hit[:,0]]
        tp=np.bincount(operating,minlength=len(items))
        out[defect+"_recall"]=(draws@tp)/denominators[:,c]
        # Precision can attain its suffix maximum only after a TP event. Compute
        # repeated-image counts at those positions without materializing copied boxes.
        for k in range(10):
            positions=np.flatnonzero(hit[:,k])
            if not len(positions): continue
            aps=np.zeros(len(draws))
            for start in range(0,len(draws),32):
                sample=draws[start:start+32]
                all_at_tp=np.cumsum(sample[:,img],axis=1,dtype=np.int32)[:,positions]
                mult=sample[:,img[positions]]
                true_at_tp=np.cumsum(mult,axis=1,dtype=np.int32)
                prec=np.divide(true_at_tp,all_at_tp,out=np.zeros_like(all_at_tp,dtype=float),where=all_at_tp>0)
                prec[mult==0]=0
                prec=np.maximum.accumulate(prec[:,::-1],axis=1)[:,::-1]
                for j in range(len(sample)):
                    # For repeated identical true detections precision increases within
                    # the repeated block. Its last event supplies the precision envelope.
                    rec=true_at_tp[j]/denominators[start+j,c]
                    ix=np.searchsorted(rec,RECALL_GRID)
                    aps[start+j]=np.r_[prec[j],0.][ix].mean()
            out["map5095"]+=aps/60
            if k==0 and defect in PRIORITY: out[defect+"_ap50"]=aps
    return out


def intervals(data,study,rows,dest,draw_count=1000):
    study,dest=Path(study),Path(dest)
    decision=read(study/"freeze.json")["selection"]
    output=[]
    factorial=[]
    for split in sorted({k[0] for k in rows}):
        sample=load_rows(data,split)
        groups,inverse=np.unique([r["group"] for r in sample],return_inverse=True)
        rng=np.random.default_rng(5122026)
        group_weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=draw_count)
        draws=group_weights[:,inverse].astype(np.int16)
        cache={}
        def metrics(key):
            if key not in cache:
                source=study/split/rows[key]["source"]/"predictions.jsonl.gz"
                records=read_predictions(source)
                if [i for i,_ in records]!=[r["id"] for r in sample]: raise ValueError("Image-order mismatch in bootstrap")
                item=events(sample,[p for _,p in records])
                # Unit weights are an independent check on AP, recall, image matching and ordering.
                original=bootstrap_metrics(item,rows[key]["cutoff"],np.ones((1,len(sample)),np.int16))
                if any(abs(float(v[0])-rows[key][m])>1e-8 for m,v in original.items()):
                    raise ValueError(f"Bootstrap/point estimator mismatch for {key}")
                cache[key]=bootstrap_metrics(item,rows[key]["cutoff"],draws)
            return cache[key]
        comparisons=[]
        models=sorted({k[1] for k in rows})
        seeds=sorted({k[2] for k in rows})
        for model in (m for m in models if m!="A"):
            comparisons.append((f"{model} minus A at 480",(model,"openvino","one",480),("A","openvino","one",480)))
        for reference in ("B","C"):
            if "D" in models and reference in models:
                comparisons.append((f"D minus {reference} at 480",("D","openvino","one",480),(reference,"openvino","one",480)))
        for m in models:
            for rt in ("openvino","pytorch"):
                if (split,m,seeds[0],rt,"dense",480) in rows:
                    comparisons.append((f"{m} {rt} one minus dense",(m,rt,"one",480),(m,rt,"dense",480)))
        chosen=decision["model"]
        if chosen:
            for side in (256,320,640):
                comparisons.append((f"{chosen} {side} minus 480",(chosen,"openvino","one",side),(chosen,"openvino","one",480)))
        for label,(m,r,h,z),(other,rt,head,side) in comparisons:
            by_metric={}
            source=dict(left_model=m,left_runtime=r,left_head=h,left_side=z,right_model=other,right_runtime=rt,
                        right_head=head,right_side=side,precision="FP32")
            left_rows=[]
            right_rows=[]
            for seed in seeds:
                a,b=(split,m,seed,r,h,z),(split,other,seed,rt,head,side)
                left_rows.append(rows[a])
                right_rows.append(rows[b])
                aa,bb=metrics(a),metrics(b)
                for metric in aa:
                    v=(aa[metric]-bb[metric])*100
                    estimate=(rows[a][metric]-rows[b][metric])*100
                    by_metric.setdefault(metric,[]).append((estimate,v))
                    lo,hi=np.percentile(v,[2.5,97.5])
                    output.append(dict(split=split,comparison=label,metric=metric,seed=seed,delta_pp=estimate,
                                       low_pp=lo,high_pp=hi,draws=draw_count,groups=len(groups),**source,
                                       left_checkpoint_sha=rows[a]["checkpoint_sha"],right_checkpoint_sha=rows[b]["checkpoint_sha"],
                                       left_source=rows[a]["cell_dir"],right_source=rows[b]["cell_dir"]))
            for metric,sets in by_metric.items():
                lo,hi=np.percentile(np.mean([v for _,v in sets],axis=0),[2.5,97.5])
                output.append(dict(split=split,comparison=label,metric=metric,seed="paired mean",
                                   delta_pp=np.mean([e for e,_ in sets]),low_pp=lo,high_pp=hi,
                                   draws=draw_count,groups=len(groups),**source,
                                   left_checkpoint_sha="|".join(v["checkpoint_sha"] for v in left_rows),
                                   right_checkpoint_sha="|".join(v["checkpoint_sha"] for v in right_rows),
                                   left_source="|".join(v["cell_dir"] for v in left_rows),
                                   right_source="|".join(v["cell_dir"] for v in right_rows)))
        factorial.extend(factorial_rows(rows,split,seeds,bootstrap=metrics,group_count=len(groups)))
    table(dest/"paired_intervals.csv",output)
    if factorial: table(dest/"fusion_intervals.csv",factorial)
