"""Build the F5 results grid and chart from the prediction files in preds/f5/.

    uv run python -m bench.report_f5

Prints the grid (model × clean/OCR: page, field and amount accuracy) as markdown and writes
results/f5_chart_{light,dark}.png. Same palette and layout rules as bench/report.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.extract import load_preds, pages, score_page, summarise
from bench.report import THEMES

SIZES = ["0.5B", "1.5B", "3B"]
METHODS = {"Zero-shot": "qwen-{s}", "Fine-tuned (SFT)": "qwen-{s}-sft", "SFT + GRPO": "qwen-{s}-grpo"}


def metrics(stem: str, variant: str, page_lists: dict) -> dict | None:
    path = Path("preds/f5") / f"{stem}__{variant}.jsonl"
    if not path.exists():
        return None
    preds = load_preds(path)
    ps = page_lists[variant]
    return summarise([score_page(p, preds.get((p["seed"], p["doc_type"]), "")) for p in ps], ps)


def main():
    page_lists = {v: pages("test", v) for v in ("clean", "ocr")}
    rows = [(f"Qwen {s}", m, stem.format(s=s)) for s in SIZES for m, stem in METHODS.items()]
    rows += [("Gemini 3.5 Flash", "Zero-shot", "gemini"), ("Gemini 3.1 Pro", "Zero-shot", "gemini-pro")]

    fmt = lambda x: f"{x:.1%}" if x is not None else "–"
    print("| Model | Method | Clean: pages | OCR: pages | OCR: fields | OCR: amounts |")
    print("|---|---|---|---|---|---|")
    grid = {}
    for label, method, stem in rows:
        c, o = metrics(stem, "clean", page_lists), metrics(stem, "ocr", page_lists)
        if c is None and o is None:
            continue
        grid[(label, method)] = (c, o)
        bold = "**" if "SFT" in method else ""
        print(f"| {label} | {method} | {bold}{fmt(c and c['page_accuracy'])}{bold} | {bold}{fmt(o and o['page_accuracy'])}{bold} "
              f"| {fmt(o and o['field_accuracy'])} | {fmt(o and o['amount_accuracy'])} |")

    gemini = {v: grid[("Gemini 3.1 Pro", "Zero-shot")][i]["page_accuracy"] * 100 for i, v in enumerate(("clean", "ocr"))}
    for theme, t in THEMES.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=t["surface"])
        for ax, (vi, variant, title) in zip(axes, [(0, "clean", "Clean text"), (1, "ocr", "OCR text (realistic)")]):
            ax.set_facecolor(t["surface"])
            ax.axhline(gemini[variant], color=t["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.text(-0.28, gemini[variant] + 1.5, f"Gemini 3.1 Pro ({gemini[variant]:.0f}%)", color=t["muted"], fontsize=9, va="bottom")
            ends = {}
            for (m, _), color in zip(METHODS.items(), t["series"]):
                pts = [(i, grid[(f"Qwen {s}", m)][vi]["page_accuracy"] * 100) for i, s in enumerate(SIZES)
                       if (f"Qwen {s}", m) in grid and grid[(f"Qwen {s}", m)][vi]]
                if not pts:
                    continue
                style = {"lw": 2} if len(pts) > 1 else {"lw": 0}
                ax.plot(*zip(*pts), color=color, marker="o", ms=8, mec=t["surface"], mew=2, zorder=3, label=m, **style)
                if len(pts) == 1:  # a lone point sits on another line's path: label it underneath
                    (i, y), = pts
                    ax.annotate(f"{m} {y:.0f}%", (i, y), xytext=(0, -16), textcoords="offset points", color=t["text"],
                                fontsize=9, ha="center", va="top", zorder=4,
                                bbox={"boxstyle": "square,pad=0.15", "fc": t["surface"], "ec": "none"})
                    continue
                ends.setdefault((pts[-1][0], round(pts[-1][1])), []).append((m, pts[-1][1]))
            for (i, _), items in ends.items():
                y = items[0][1]
                label = " & ".join([items[0][0]] + [n.lower() for n, _ in items[1:]])
                ax.annotate(f"{label} {y:.0f}%", (i, y), xytext=(10, 0), textcoords="offset points", color=t["text"],
                            fontsize=9, va="center", zorder=4, bbox={"boxstyle": "square,pad=0.15", "fc": t["surface"], "ec": "none"})
            ax.set_title(title, color=t["text"], fontsize=11, loc="left", pad=10)
            ax.set_xticks(range(len(SIZES)), [f"Qwen {s}" for s in SIZES], color=t["muted"])
            ax.set_xlim(-0.3, len(SIZES) - 0.3 + 0.9)
            ax.set_ylim(0, 110)
            ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], color=t["muted"])
            ax.grid(axis="y", color=t["grid"], lw=0.8)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
        axes[0].set_ylabel("Pages with every field right (98 test pages)", color=t["muted"])
        fig.legend(*axes[1].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.02), labelcolor=t["text"])
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = Path("results") / f"f5_chart_{theme}.png"
        fig.savefig(out, dpi=160, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
