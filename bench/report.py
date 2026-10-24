"""Build the F2 results grid and chart from the prediction files.

    uv run python -m bench.report

Prints the full grid (size × method × text variant) as a markdown table and writes the chart
in light and dark versions to results/f2_chart_{light,dark}.png, for a <picture> element
that follows the reader's GitHub theme.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.score import load_predictions, score, test_pages

SIZES = ["0.5B", "1.5B", "3B"]
VARIANTS = {"clean": "Clean", "masked": "Titles hidden", "ocr": "OCR", "ocr-masked": "OCR + titles hidden"}
METHODS = {  # label → prediction-file stem for a given size
    "Zero-shot": lambda s: "qwen-zeroshot" if s == "0.5B" else f"qwen-zeroshot-{s}",
    "Few-shot": lambda s: "qwen-fewshot" if s == "0.5B" else f"qwen-fewshot-{s}",
    "Fine-tuned": lambda s: f"qwen-finetuned-{s}",
}
# Categorical slots 1–3 of the reference palette (validated light and dark, all pairs).
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#3a3936",
             "series": ["#3987e5", "#d95926", "#199e70"]},
}


def accuracy(stem: str, variant: str, pages) -> float | None:
    path = Path("preds") / f"{stem}{'' if variant == 'clean' else '__' + variant}.jsonl"
    return score(load_predictions(path), pages)["accuracy"] * 100 if path.exists() else None


def grid(pages) -> dict:
    return {(m, s, v): accuracy(stem(s), v, pages) for m, stem in METHODS.items() for s in SIZES for v in VARIANTS}


def table(g, pages) -> str:
    fmt = lambda x: "–" if x is None else f"{x:.1f}%"
    lines = ["| Model | Method | " + " | ".join(VARIANTS.values()) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for s in SIZES:
        for m in METHODS:
            cells = [fmt(g[(m, s, v)]) for v in VARIANTS]
            if m == "Fine-tuned":
                cells = [f"**{c}**" for c in cells]
            lines.append(f"| Qwen {s} | {m} | " + " | ".join(cells) + " |")
    gem = [fmt(accuracy("gemini-pro", v, pages)) for v in VARIANTS]
    lines.append("| Gemini 3.1 Pro | Zero-shot | " + " | ".join(gem) + " |")
    return "\n".join(lines)


def chart(g, theme: str, out: Path) -> None:
    t = THEMES[theme]
    panels = [("clean", "Clean text"), ("ocr-masked", "OCR + titles hidden (realistic)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, facecolor=t["surface"])
    x = range(len(SIZES))
    for ax, (variant, title) in zip(axes, panels):
        ax.set_facecolor(t["surface"])
        ax.axhline(100, color=t["muted"], lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.text(-0.28, 101.5, "Gemini 3.1 Pro (100%)", color=t["muted"], fontsize=9, va="bottom", ha="left")
        ends = {}  # end value → methods, so tied lines share one label instead of overprinting
        for (m, _), color in zip(METHODS.items(), t["series"]):
            ys = [g[(m, s, variant)] for s in SIZES]
            pts = [(i, y) for i, y in zip(x, ys) if y is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), color=color, lw=2, marker="o", ms=8, mec=t["surface"], mew=2, zorder=3, label=m)
            ends.setdefault(pts[-1], []).append(m)
        for (i, y), names in ends.items():
            label = " & ".join([names[0]] + [n.lower() for n in names[1:]])
            ax.annotate(f"{label} {y:.0f}%", (i, y), xytext=(10, 0), textcoords="offset points",
                        color=t["text"], fontsize=9, va="center", zorder=4,
                        bbox={"boxstyle": "square,pad=0.15", "fc": t["surface"], "ec": "none"})
        ax.set_title(title, color=t["text"], fontsize=11, loc="left", pad=10)
        ax.set_xticks(list(x), [f"Qwen {s}" for s in SIZES], color=t["muted"])
        ax.set_xlim(-0.3, len(SIZES) - 0.3 + 0.9)
        ax.set_ylim(0, 110)
        ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)], color=t["muted"])
        ax.grid(axis="y", color=t["grid"], lw=0.8)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Accuracy on 124 test pages", color=t["muted"])
    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False,
                        bbox_to_anchor=(0.5, 1.02), labelcolor=t["text"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=t["surface"])
    plt.close(fig)


def main():
    pages = test_pages()
    g = grid(pages)
    print(table(g, pages))
    for theme in THEMES:
        out = Path("results") / f"f2_chart_{theme}.png"
        chart(g, theme, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
