"""Draw the four modified bottlenecks and the two learned top-down joins."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw(target):
    with plt.rc_context({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 9}):
        fig, ax = plt.subplots(figsize=(7.1, 2.58))
        ax.set(xlim=(0, 10), ylim=(0, 3.75))
        ax.axis("off")
        ink, line, teal = "#20323c", "#526574", "#176858"

        def box(x, y, width, label, filled=False):
            patch = FancyBboxPatch((x, y-.24), width, .48,
                boxstyle="round,pad=0.012,rounding_size=.05", facecolor="#e6f2ee" if filled else "#f5f7f9",
                edgecolor=teal if filled else line, linewidth=.85)
            ax.add_patch(patch)
            ax.text(x+width/2, y, label, ha="center", va="center", fontsize=8, color=ink)

        def arrow(points):
            for (x, y), (u, v) in zip(points[:-2], points[1:-1]):
                ax.plot((x, u), (y, v), color=line, linewidth=.9)
            ax.annotate("", xy=points[-1], xytext=points[-2],
                        arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, lw=.9, color=line,
                                        mutation_scale=8))

        ax.text(.03, 3.52, "(a)  Backbone bottleneck", fontsize=10, weight="bold", color=ink)
        ax.text(.03, 3.20, "P4 and P5  ·  two replacements per stage", fontsize=8.5, color=line)
        ax.text(.05, 2.40, "Input\nchannels", va="center", ha="left", fontsize=8, color=ink)
        box(1.52, 2.84, 2.18, "First ¼  ·  3 × 3 conv", True)
        box(1.52, 1.98, 2.18, "Remaining ¾  ·  copy")
        arrow([(.80, 2.40), (1.18, 2.40), (1.18, 2.84), (1.50, 2.84)])
        arrow([(1.18, 2.40), (1.18, 1.98), (1.50, 1.98)])
        arrow([(3.71, 2.84), (4.00, 2.84), (4.00, 1.36), (3.60, 1.36)])
        arrow([(2.62, 1.73), (2.62, 1.62)])
        box(1.57, 1.36, 2.02, "Concatenate")
        arrow([(2.58, 1.10), (2.58, .88)])
        box(.99, .62, 3.18, "Full-width norm + activation")
        ax.plot([4.63, 4.63], [.30, 3.54], color="#c8d2d4", lw=.8)

        ax.text(4.91, 3.52, "(b)  Learned feature fusion", fontsize=10, weight="bold", color=ink)
        ax.text(4.91, 3.20, "Two existing joins  ·  P5 → P4 and P4 → P3", fontsize=8.5, color=line)
        box(4.98, 2.64, 1.88, "Top-down tensor")
        box(7.75, 2.64, 1.93, "Lateral tensor")
        box(5.98, 1.68, 2.69, "Pool both → MLP → 2 weights", True)
        arrow([(5.93, 2.39), (5.93, 2.14), (6.69, 2.14), (6.69, 1.94)])
        arrow([(8.72, 2.39), (8.72, 2.14), (7.96, 2.14), (7.96, 1.94)])
        box(5.82, .62, 3.01, "Scale branches and concatenate")
        arrow([(7.33, 1.43), (7.33, .88)])
        arrow([(5.15, 2.39), (5.15, .62), (5.80, .62)])
        arrow([(9.50, 2.39), (9.50, .62), (8.85, .62)])
        ax.text(7.33, .11, "Twice-softmax starts both weights at 1.", ha="center", fontsize=8, color=line)
        fig.subplots_adjust(left=.012, right=.991, top=.988, bottom=.035)
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
