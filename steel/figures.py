import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw

from .bench import read_predictions
from .common import CLASSES, SIDES, put, read, table
from .data import as_arrays, gray_image, load_rows
from .demo import font, pair_image, panel
from .factorial import CONTRASTS, Q
from .metrics import iou
from .report import PRIORITY

COLORS={"A":"#28698c","B":"#b74c27","C":"#835296","D":"#8a6821","E":"#277856"}
MODEL_NAMES={"A":"A · YOLO26n","B":"B · PConv","C":"C · Fusion","D":"D · Both","E":"E · YOLOv10n"}
SEED_LINES={0:"-",1:"--",2:":"}


def curves(out,figs,models,seeds):
    fig,axes=plt.subplots(1,2,figsize=(9.3,3.8),layout="constrained")
    for m in models:
        for k in seeds:
            with (out/"runs"/f"{m}{k}"/"epochs.csv").open() as f: log=list(csv.DictReader(f))
            epochs=[int(r["epoch"]) for r in log]
            losses=[sum(float(v) for key,v in r.items() if key.endswith("_loss")) for r in log]
            label=MODEL_NAMES[m] if k==0 else None
            axes[0].plot(epochs,losses,c=COLORS[m],ls=SEED_LINES[k],lw=1.1,alpha=0.75)
            axes[1].plot(epochs,[100*float(r["map5095"]) for r in log],c=COLORS[m],ls=SEED_LINES[k],lw=1.1,alpha=0.75,label=label)
    axes[0].set(ylabel="Reported loss components, sum",xlabel="Training epoch")
    axes[1].set(ylabel="Validation mAP50–95 (%)",xlabel="Training epoch")
    for ax in axes: ax.grid(axis="y",alpha=0.2)
    axes[0].legend(handles=[Line2D([0],[0],ls=SEED_LINES[k],color="#444",label=str(k)) for k in seeds],
                   title="Training seed",frameon=False)
    axes[1].legend(frameon=False,fontsize=8)
    fig.savefig(figs/"training_curves.pdf",bbox_inches="tight")
    plt.close(fig)


def size_plots(rows,figs,split,models,seeds):
    for cls in PRIORITY:
        fig,axes=plt.subplots(1,len(models),figsize=(2.15*len(models)+0.35,2.9),sharex=True,sharey=True,layout="constrained",squeeze=False)
        for m,ax in zip(models,axes.flat):
            means=[]
            for side in SIDES:
                p=[rows[(split,m,k,"openvino","one",side)] for k in seeds]
                x=[a["pipeline_p95_ms"] for a in p]
                y=[100*a[cls+"_ap50"] for a in p]
                ax.scatter(x,y,s=16,c=COLORS[m],alpha=0.5)
                means.append((float(np.mean(x)),float(np.mean(y))))
                offset=(2,-12) if side==320 else (2,7)
                ax.annotate(str(side),means[-1],xytext=offset,textcoords="offset points",fontsize=8)
            x,y=zip(*means)
            ax.plot(x,y,"o-",c=COLORS[m],lw=1,ms=4)
            ax.set_title(MODEL_NAMES[m],fontsize=11)
            ax.grid(alpha=0.2)
            ax.margins(x=0.15,y=0.22)
        axes[0,0].set_ylabel(cls.replace("_"," ").capitalize()+" AP50 (%)")
        fig.supxlabel("Decoded image to final boxes, p95 (ms)",fontsize=10)
        fig.savefig(figs/f"{split}_{cls}_size.pdf",bbox_inches="tight")
        plt.close(fig)


def fusion_plot(dest, figs):
    with (dest/"fusion_intervals.csv").open(newline="") as stream:
        results=[r for r in csv.DictReader(stream) if r["split"]=="test" and r["metric"]==Q]
    fig,ax=plt.subplots(figsize=(6.1,2.9),layout="constrained")
    ax.axvline(0,color="#888888",lw=0.8)
    for i,label in enumerate(CONTRASTS):
        group={r["seed"]:r for r in results if r["contrast"]==label}
        mean=group["paired mean"]
        ax.hlines(i,float(mean["low"]),float(mean["high"]),color="#253d51",lw=2.1)
        ax.scatter(float(mean["estimate"]),i,marker="D",color="#253d51",s=25,zorder=4)
        for seed,marker in zip(("0","1","2"),("o","s","^"),strict=True):
            ax.scatter(float(group[seed]["estimate"]),i+.18,marker=marker,s=20,
                       edgecolor="#92522a",facecolor="white",linewidth=.9,zorder=3)
    ax.set(yticks=range(len(CONTRASTS)),yticklabels=list(CONTRASTS),
           xlabel="Change in test mAP50–95 (percentage points)")
    ax.invert_yaxis()
    ax.grid(axis="x",alpha=.15)
    handles=[Line2D([],[],marker="D",color="#253d51",label="Paired mean, 95% group interval")]
    handles.extend(Line2D([],[],marker=marker,linestyle="none",mec="#92522a",mfc="white",label=f"Seed {i}")
                   for i,marker in enumerate(("o","s","^")))
    ax.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,1.02),ncol=4,
              fontsize=8,frameon=False,columnspacing=1)
    fig.savefig(figs/"fusion_contrasts.pdf",bbox_inches="tight")
    plt.close(fig)


def assignment(row,det):
    gt,labels=as_arrays(row)
    pairs={}
    for j in np.argsort(-det[:,4],kind="stable"):
        options=[i for i,c in enumerate(labels) if c==int(det[j,5]) and i not in pairs]
        if not options: continue
        overlap=iou(det[j:j+1,:4],gt[options])[0]
        k=int(np.argmax(overlap))
        if overlap[k]>=0.5: pairs[options[k]]=int(j)
    return pairs


def failure_gallery(data,study,rows,figs,selection):
    split="test"
    model=selection["model"] or "A"
    side=selection["side"] or 480
    chosen=rows[(split,model,0,"openvino","one",side)]
    sample=load_rows(data,split)
    files={s:study/split/rows[(split,model,0,"openvino","one",s)]["source"]/"predictions.jsonl.gz" for s in SIDES}
    operating={s:dict(read_predictions(f,"operating")) for s,f in files.items()}
    low={s:dict(read_predictions(f)) for s,f in files.items()}
    cases={}
    for row in sample:
        det=operating[side][row["id"]]
        matched=assignment(row,det)
        conditions={"success":len(matched)==len(row["boxes"])==len(det),"miss":len(matched)<len(row["boxes"]),
                    "false_positive":len(matched)<len(det)}
        for category,valid in conditions.items():
            if not valid or category in cases: continue
            gray=gray_image(data,row)
            footer=f"{len(matched)} matched   {len(row['boxes'])-len(matched)} missed   {len(det)-len(matched)} false positives     Cutoff {chosen['cutoff']:.4f}"
            pic=pair_image(row,gray,det,f"{row['id']}   |   {model} seed 0, {side} px, one-to-one",footer)
            pic.save(figs/f"{category}.png")
            cases[category]=dict(id=row["id"],gt=len(row["boxes"]),matched=len(matched),predictions=len(det),path=f"{category}.png")
    explanations=[]
    for c in PRIORITY:
        found=False
        for row in sample:
            rid=row["id"]
            gt,labels=as_arrays(row)
            if CLASSES.index(c) not in labels: continue
            hits={s:assignment(row,operating[s][rid]) for s in SIDES}
            for g in np.flatnonzero(labels==CLASSES.index(c)):
                if len({int(g) in hits[s] for s in SIDES})<2: continue
                gray=gray_image(data,row)
                canvas=Image.new("RGB",(840,932),(249,249,246))
                draw=ImageDraw.Draw(canvas)
                draw.text((12,5),f"{rid}    {c.replace('_',' ')}    cutoff {chosen['cutoff']:.3f}",font=font(21),fill="#15202b")
                for pos,s in enumerate(SIDES):
                    op=operating[s][rid]
                    det=low[s][rid]
                    match=hits[s].get(int(g))
                    same=det[det[:,5]==labels[g]]
                    overlaps=iou(same[:,:4],gt[g:g+1])[:,0]
                    qualifying=np.flatnonzero(overlaps>=0.50)
                    if match is not None:
                        candidate=op[match]
                        why="matched"
                    elif len(qualifying):
                        candidate=same[qualifying[np.argmax(same[qualifying,4])]]
                        why="below cutoff" if candidate[4]<chosen["cutoff"] else "matched elsewhere"
                    elif len(same):
                        candidate=same[int(np.argmax(overlaps))]
                        why="IoU below 0.50"
                    else:
                        candidate=None
                        why="no class candidate"
                    score=float(candidate[4]) if candidate is not None else None
                    overlap=float(iou(candidate[None,:4],gt[g:g+1])[0,0]) if candidate is not None else None
                    tile=panel(gray,gt[g:g+1],labels[g:g+1],size=390)
                    d=ImageDraw.Draw(tile)
                    if candidate is not None:
                        d.rectangle(tuple(candidate[:4]*390/200),outline=(42,224,222),width=2)
                    x,y=12+(pos%2)*414,42+(pos//2)*441
                    detail=f"score {score:.3f}    IoU {overlap:.3f}" if score is not None else "No same-class box above AP floor"
                    draw.text((x,y),f"{s} px   {why}",font=font(17),fill="#15202b")
                    canvas.paste(tile,(x,y+28))
                    draw.text((x,y+419),detail,font=font(14),fill="#15202b")
                    explanations.append(dict(defect=c,id=rid,gt_index=int(g),side=s,score=score,iou=overlap,
                                             outcome=why,threshold=chosen["cutoff"]))
                canvas.save(figs/f"size_change_{c}.png")
                found=True
                break
            if found: break
    put(figs/"demo_cases.json",dict(model=model,side=side,cutoff=chosen["cutoff"],split=split,cases=cases,
                                    rule="First image in frozen ID order satisfying each condition"))
    if explanations: table(figs/"size_change_objects.csv",explanations)


def latex_tables(dest,rows,models,seeds):
    out=dest/"latex"
    out.mkdir(parents=True,exist_ok=True)
    def save(name,columns,body):
        content="\\begin{tabular}{l"+"r"*(len(columns)-1)+"}\n\\hline\n"
        content+=" & ".join(columns)+" \\\\\n\\hline\n"
        for line in body: content+=" & ".join(line)+" \\\\\n"
        (out/name).write_text(content+"\\hline\n\\end{tabular}\n",encoding="utf-8")
    lines=[]
    for m in models:
        for s in SIDES:
            p=[rows[("test",m,k,"openvino","one",s)] for k in seeds]
            lines.append([m,str(s),*(f"{np.mean([v[c+'_ap50'] for v in p])*100:.1f}" for c in PRIORITY),
                          f"{np.mean([v['map5095'] for v in p])*100:.1f}",f"{np.mean([v['pipeline_p95_ms'] for v in p]):.2f}"])
    save("resolution.tex",["Model","Side","Cr AP","RS AP","mAP","p95 (ms)"],lines)
    with (dest/"head_comparison.csv").open() as f:pairs=list(csv.DictReader(f))
    lines=[]
    for m in models:
        for rt in ("openvino","pytorch"):
            p=[v for v in pairs if v["split"]=="test" and v["model_id"]==m and v["runtime"]==rt]
            if p:
                lines.append([m,"OV" if rt=="openvino" else "PT",*(f"{np.mean([float(v[col]) for v in p]):.2f}"
                              for col in ("dense_p95_ms","one_p95_ms","saving_ms","saving_percent","dense_nms_p95_ms"))])
    save("heads.tex",["Model","CPU","Dense","One","Saved","\\%","NMS"],lines)
    if all(m in models for m in "ABCD"):
        with (dest/"fusion_intervals.csv").open(newline="") as stream:
            quality={(r["contrast"],r["seed"]):r for r in csv.DictReader(stream)
                     if r["split"]=="test" and r["metric"]==Q}
        with (dest/"fusion_contrasts.csv").open(newline="") as stream:
            latency={r["contrast"]:float(r["estimate"]) for r in csv.DictReader(stream)
                     if r["split"]=="test" and r["metric"]=="pipeline_p95_ms" and r["seed"]=="paired mean"}
        lines=[]
        for contrast in CONTRASTS:
            m=quality[contrast,"paired mean"]
            values=[f"{float(quality[contrast,str(seed)]['estimate']):+.2f}" for seed in seeds]
            lines.append(["$"+contrast+"$",*values,f"{float(m['estimate']):+.2f}",
                          f"[{float(m['low']):+.2f}, {float(m['high']):+.2f}]",f"{latency[contrast]:+.2f}"])
        save("fusion.tex",["Contrast",*(f"Seed {k}" for k in seeds),"Mean","95\\% interval","$\\Delta p_{95}$ (ms)"],lines)
        lines=[]
        for model in models:
            p=[rows[("test",model,k,"openvino","one",480)] for k in seeds]
            lines.append([model,f"{p[0]['active_parameters']:,}",f"{p[0]['flops_at_480']/1e9:.3f}",
                          f"{100*np.mean([r['map5095'] for r in p]):.2f}",
                          f"{np.mean([r['pipeline_p95_ms'] for r in p]):.2f}"])
        save("architecture_cost.tex",["Model","Parameters","GFLOPs","mAP (\\%)","p95 (ms)"],lines)
    if "B" in models:
        with (dest/"paired_intervals.csv").open() as f:
            recall_intervals={r["metric"]:r for r in csv.DictReader(f)
                              if r["split"]=="test" and r["comparison"]=="B minus A at 480" and r["seed"]=="paired mean"}
        lines=[]
        for cls in CLASSES:
            diff=recall_intervals[cls+"_recall"]
            a=[rows[("test","A",seed,"openvino","one",480)][cls+"_recall"] for seed in seeds]
            b=[rows[("test","B",seed,"openvino","one",480)][cls+"_recall"] for seed in seeds]
            lines.append([cls.replace("_"," "),f"{100*np.mean(a):.1f}",f"{100*np.mean(b):.1f}",
                          f"{float(diff['delta_pp']):+.1f}",
                          f"[{float(diff['low_pp']):+.1f}, {float(diff['high_pp']):+.1f}]"])
        save("pconv_recall.tex",["Defect","A (\\%)","B (\\%)","Change","95\\% interval"],lines)


def research_figures(data,out,rows,dest):
    data,out,dest=Path(data),Path(out),Path(dest)
    study=out/"study"
    freeze=read(study/"freeze.json")
    models,seeds=freeze["models"],freeze["seeds"]
    figs=dest/"figures"
    figs.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    curves(out,figs,models,seeds)
    for s in ("val","test"): size_plots(rows,figs,s,models,seeds)
    if (dest/"fusion_intervals.csv").exists(): fusion_plot(dest,figs)
    failure_gallery(data,study,rows,figs,freeze["selection"])
    latex_tables(dest,rows,models,seeds)
