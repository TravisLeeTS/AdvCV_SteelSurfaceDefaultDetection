"""Plot the speed and operating recall of each held-out stock checkpoint."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot(out):
    labels={"A":"YOLO26n","E":"YOLOv10n","yolov8n":"YOLOv8n","yolov5s":"YOLOv5s"}
    styles={"A":("#1f537b","o"),"E":("#a44c16","s"),"yolov8n":("#167264","^"),"yolov5s":("#704783","D")}
    with (out/"tables/test-paths.csv").open(encoding="utf-8",newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if r["runtime"]=="openvino" and r["model"] in labels
              and r["head"]==("one" if r["model"] in ("A","E") else "dense")]
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                         "axes.spines.top":False,"axes.spines.right":False}):
        fig,ax=plt.subplots(figsize=(3.58,2.52),layout="constrained")
        for model,label in labels.items():
            color,marker=styles[model]
            group=[r for r in rows if r["model"]==model]
            if len(group)!=3:
                raise ValueError(f"Expected three test checkpoints for {label}")
            x=np.array([float(r["p95_ms"]) for r in group])
            y=np.array([100*float(r["macro_recall"]) for r in group])
            ax.scatter(x,y,marker=marker,s=21,facecolors="white",edgecolors=color,linewidth=.9,zorder=2)
            ax.scatter([x.mean()],[y.mean()],marker=marker,s=37,color=color,label=label,zorder=3)
        ax.set_xlabel("Decoded image to final boxes, test $p_{95}$ (ms)")
        ax.set_ylabel("Test macro recall (%)")
        ax.grid(axis="both",color="#d8dfe4",linewidth=.5,zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.legend(loc="upper left",bbox_to_anchor=(0,1.28),ncol=2,frameon=False,
                  columnspacing=1.4,handlelength=1.1,fontsize=7.5)
        target=out/"tables/stock-speed-recall.pdf"
        fig.savefig(target,metadata={"Creator":"MLR 512 CPU benchmark","CreationDate":None,"ModDate":None})
        plt.close(fig)
    return target


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("artifacts/cpu-extension"))
    print(plot(parser.parse_args().out))
