"""The results figure: two contrasts, one scale.

Each task is one row, and the row is read left to right as two successive
steps: what the graph buys over cell-type composition, then what the hyperedge
buys over the graph. Both panels share an x-axis, which is the entire point --
the left intervals sit clear of zero and the right ones straddle it at a
fraction of the width, so a reader sees the same pipeline, on the same folds,
resolving an effect of one size and not another. That is what licenses
reporting the nulls as bounds rather than as absences, and no pair of tables
shows it.

Differences and intervals come from the same corrected test the text quotes, so
the figure cannot disagree with the tables.

    python scripts/fig_results.py --out figs \
        --task "TIL arrangement|~/results/tilClr|abundance-only|pw-radius@gin|hg-radius@deepsets2" \
        --task "PAM50 luminal A|~/results/lumaClr|abundance-only|pw-radius@gin|hg-radius@deepsets2"

Each --task is "label|results dir|composition arm|pairwise arm|hypergraph arm".
Rows appear in the order given.

Survival does not belong here: a C-index is not a macro-F1 and sharing an axis
with one would be a category error. Report it in the table alone.
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

CLEAR = "#c0392b"       # interval excludes zero
NULL = "#34495e"        # interval contains zero
ZERO = "#95a5a6"


def contrast(parts, order, base, arm, field):
    """variant minus baseline, with the corrected interval."""
    diffs = stacked(parts, order, arm, field) - stacked(parts, order, base, field)
    n_te = float(np.mean([r["n_test"] for p in parts for r in p["runs"]]))
    n_tr = float(np.mean([p["n_train_mean"] for p in parts]))
    mean, _t, p, half = corrected_t_test(diffs, n_te, n_tr)
    return dict(mean=mean, half=half, p=p, n=len(diffs))


def one_task(spec, field):
    fields = [t.strip() for t in spec.split("|")]
    if len(fields) != 5:
        raise SystemExit('--task needs "label|dir|composition|pairwise|hypergraph"'
                         f", got {spec!r}")
    label, d, comp, pw, hg = fields
    files = glob.glob(os.path.join(os.path.expanduser(d), "*.json"))
    if not files:
        raise SystemExit(f"no result JSON in {d}")
    parts = load_parts(files)
    order = run_order(parts)
    have = parts[0]["scores"]
    for a in (comp, pw, hg):
        if a not in have:
            raise SystemExit(f"{d}: no arm {a!r}; have {sorted(have)}")
    # Both contrasts come from the same runs on the same folds, so the pairing
    # the corrected test assumes holds for each panel separately.
    return dict(label=label,
                left=contrast(parts, order, comp, pw, field),
                right=contrast(parts, order, pw, hg, field))


def draw(ax, rows, key, title, xlim, show_labels):
    ax.axvline(0.0, color=ZERO, linewidth=1.0, zorder=1)
    ys = []
    for i, r in enumerate(rows):
        y = len(rows) - i
        ys.append(y)
        c = r[key]
        # emphasis marks the only thing the figure asserts: whether the
        # interval excludes zero
        col = CLEAR if abs(c["mean"]) > c["half"] else NULL
        ax.plot([c["mean"] - c["half"], c["mean"] + c["half"]], [y, y],
                color=col, linewidth=1.7, solid_capstyle="round", zorder=2)
        ax.plot([c["mean"]], [y], "o", color=col, markersize=5.5, zorder=3)
        ax.annotate(f"{c['mean']:+.3f}", (c["mean"], y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=col)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows] if show_labels else [],
                       fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(0.4, len(rows) + 0.95)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=9, pad=8)
    ax.grid(axis="x", color="#ecf0f1", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=[],
                    help='"label|dir|composition arm|pairwise arm|hypergraph arm"')
    ap.add_argument("--field", default="f1")
    ap.add_argument("--xlabel", default="difference in macro-F1")
    ap.add_argument("--unshared-x", action="store_true",
                    help="scale each panel to its own range. Off by default and "
                         "worth leaving off: the shared scale is what shows the "
                         "right-hand intervals are narrow rather than merely "
                         "uninformative")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--format", default="pdf")
    args = ap.parse_args()
    if not args.task:
        raise SystemExit("give at least one --task")

    rows = [one_task(t, args.field) for t in args.task]

    def span(keys):
        lo = min(r[k]["mean"] - r[k]["half"] for r in rows for k in keys)
        hi = max(r[k]["mean"] + r[k]["half"] for r in rows for k in keys)
        pad = 0.12 * (hi - lo)
        return min(lo - pad, -0.01), hi + pad

    if args.unshared_x:
        xlim_l, xlim_r = span(["left"]), span(["right"])
    else:
        xlim_l = xlim_r = span(["left", "right"])

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(8.2, 0.52 * len(rows) + 1.7),
        gridspec_kw=dict(width_ratios=[1, 1], wspace=0.12))
    draw(axl, rows, "left", "graph over composition\n(pairwise − composition)",
         xlim_l, True)
    draw(axr, rows, "right", "hyperedge over graph\n(hypergraph − pairwise)",
         xlim_r, False)
    for ax in (axl, axr):
        ax.set_xlabel(args.xlabel, fontsize=9)

    os.makedirs(args.out, exist_ok=True)
    path = f"{args.out}/fig_results.{args.format}"
    fig.savefig(path, bbox_inches="tight")

    for r in rows:
        for k, name in (("left", "graph-comp"), ("right", "hyper-pair")):
            c = r[k]
            print(f"  {r['label']:<22} {name:<11} {c['mean']:+.3f} "
                  f"[{c['mean'] - c['half']:+.3f}, {c['mean'] + c['half']:+.3f}] "
                  f"p={c['p']:.3f}  n={c['n']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
