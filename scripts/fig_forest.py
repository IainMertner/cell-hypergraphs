"""The results figure: every difference with its corrected interval, one axis.

The point of putting the abundance contrasts on the same axis as the
architectural ones is that a reader sees the design resolving an effect of one
size and not another. Two tables cannot show that; a shared axis does it at a
glance, and it is the whole basis for reporting the nulls as bounds rather than
as absences.

Differences and intervals come from the same corrected test the text quotes, so
the figure cannot disagree with the tables.

    python scripts/fig_forest.py --out figs \
        --row "LumA: spatial|~/results/lumaClr|abundance-only|pw-radius@gin" \
        --row "LumA: hypergraph|~/results/lumaClr|pw-radius@gin|hg-radius@deepsets2"

Each --row is "label|results dir|baseline arm|variant arm", and the plotted
difference is variant minus baseline. Rows appear in the order given; a row
whose label is empty leaves a gap, for separating groups.
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from combine_results import load_parts, run_order, stacked      # noqa: E402
from corrected_test import corrected_t_test                     # noqa: E402


def one_row(spec, field):
    fields = [t.strip() for t in spec.split("|")]
    if not fields[0]:                       # a gap row, for separating groups
        return None
    if len(fields) != 4:
        raise SystemExit(f'--row needs "label|dir|baseline|variant", got {spec!r}')
    label, d, base, arm = fields
    files = glob.glob(os.path.join(os.path.expanduser(d), "*.json"))
    if not files:
        raise SystemExit(f"no result JSON in {d}")
    parts = load_parts(files)
    order = run_order(parts)
    for a in (base, arm):
        if a not in parts[0]["scores"]:
            raise SystemExit(f"{d}: no arm {a!r}; have "
                             f"{sorted(parts[0]['scores'])}")
    diffs = stacked(parts, order, arm, field) - stacked(parts, order, base, field)
    n_te = float(np.mean([r["n_test"] for p in parts for r in p["runs"]]))
    n_tr = float(np.mean([p["n_train_mean"] for p in parts]))
    mean, _t, p, half = corrected_t_test(diffs, n_te, n_tr)
    return dict(label=label, mean=mean, half=half, p=p, n=len(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", default=[],
                    help='"label|dir|baseline arm|variant arm"')
    ap.add_argument("--field", default="f1",
                    help="f1 for macro-F1 or C-index, acc for accuracy")
    ap.add_argument("--xlabel", default="difference (macro-F1)")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args()
    if not args.row:
        raise SystemExit("give at least one --row")

    rows = [one_row(r, args.field) for r in args.row]

    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.2))
    ax.axvline(0.0, color="#7f8c8d", linewidth=1.0, zorder=1)

    ys, labels = [], []
    for i, r in enumerate(rows):
        y = len(rows) - i
        ys.append(y)
        labels.append("" if r is None else r["label"])
        if r is None:
            continue
        # an interval clear of zero is the only thing this figure asserts, so
        # it is the only thing given emphasis
        clear = abs(r["mean"]) > r["half"]
        c = "#c0392b" if clear else "#34495e"
        ax.plot([r["mean"] - r["half"], r["mean"] + r["half"]], [y, y],
                color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
        ax.plot([r["mean"]], [y], "o", color=c, markersize=5.5, zorder=3)
        ax.annotate(f"{r['mean']:+.3f}", (r["mean"], y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=c)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(axis="y", length=0)      # a gap row must leave no tick behind
    ax.set_ylim(0.4, len(rows) + 0.9)
    ax.set_xlabel(args.xlabel)
    ax.grid(axis="x", color="#ecf0f1", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/fig_forest.{args.format}"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    for r in rows:
        if r:
            print(f"  {r['label']:<34} {r['mean']:+.3f} "
                  f"[{r['mean'] - r['half']:+.3f}, {r['mean'] + r['half']:+.3f}] "
                  f"p={r['p']:.3f}  n={r['n']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
