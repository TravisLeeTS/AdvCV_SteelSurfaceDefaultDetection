"""Vector figure of held-out recall at four validation-selected alarm allowances."""
from collections import defaultdict
from statistics import mean

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import rc_context

from .head_calibration import BUDGETS, PRIMARY_BUDGET


def plot_head_sensitivity(rows, destination):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "test" and row["runtime"] == "openvino" and row["model"] in ("A", "E"):
            groups[row["model"], row["head"], row["val_budget_fp_per_image"]].append(row)
    for sample in groups.values():
        if sorted(row["seed"] for row in sample) != [0, 1, 2]:
            raise ValueError("Expected the same three training seeds for each plotted operating point.")
    series = (("dense", "Dense + NMS", "#0072B2", "s", "-"),
              ("one", "One-to-one", "#A94A00", "o", "--"))
    scores = (("macro_recall", "Six-class macro"), ("crazing_recall", "Crazing"),
              ("rolled_in_scale_recall", "Rolled-in scale"))
    with rc_context({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.65,
                     "pdf.fonttype": 42}):
        fig = Figure(figsize=(7.1, 3.75))
        FigureCanvasAgg(fig)
        axs = fig.subplots(2, 3, sharex=True, sharey="col")
        for i, (model, title) in enumerate((("A", "YOLO26n"), ("E", "YOLOv10n"))):
            for j, (metric, label) in enumerate(scores):
                ax = axs[i, j]
                for head, _, color, marker, line in series:
                    points = [groups[model, head, budget / 100] for budget in BUDGETS]
                    x = [mean(row["fp_per_image"] for row in p) for p in points]
                    y = [100 * mean(row[metric] for row in p) for p in points]
                    ax.plot(x, y, color=color, marker=marker, markersize=3.5,
                            linewidth=1.25, linestyle=line)
                    primary = BUDGETS.index(PRIMARY_BUDGET)
                    ax.plot(x[primary], y[primary], color="black", marker="o",
                            markersize=8.4, fillstyle="none", linestyle="none", markeredgewidth=0.9)
                if i == 0:
                    ax.set_title(label, loc="left", fontweight="bold", fontsize=8.5, pad=6)
                if j == 0:
                    ax.set_ylabel(f"{title}\nRecall (%)")
                ax.set_xlim(0, 0.37)
                ax.set_ylim(0, (70, 30, 50)[j])
                ax.set_xticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])
                ax.tick_params(length=3, width=0.55, pad=2.5)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", linewidth=0.47, color="#DDDDDD")
                ax.set_axisbelow(True)
        handles = [Line2D([], [], color=c, marker=m, linestyle=ls, lw=1.3, ms=4, label=label)
                   for _, label, c, m, ls in series]
        handles.append(Line2D([], [], color="black", marker="o", markersize=7,
                              markerfacecolor="none", linestyle="none", label="0.10 validation FP/image"))
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.supxlabel("False-positive boxes per annotated test image", x=0.52, y=0.035, fontsize=8.5)
        fig.subplots_adjust(left=0.092, right=0.99, bottom=0.16, top=0.815, hspace=0.22, wspace=0.34)
        fig.savefig(destination, format="pdf", metadata={"Creator": "MLR512 steel defect study",
                    "Title": "Test recall at validation-selected false-positive budgets",
                    "CreationDate": None, "ModDate": None})
