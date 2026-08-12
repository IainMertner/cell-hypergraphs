"""Merge the per-seed JSONs from `train_patterns.py --save-results` into the
report a single unsplit run would have produced.

Splitting is safe because folds are deterministic in (cohort, n_folds, seed) and
arms are evaluated over the same ordered run list, so concatenating parts keeps
scores paired index-for-index. What would break that is checked here: a changed
cohort (segmentation landing mid-sweep), a changed task/folds/epochs/arms, or a
repeated seed, which would double-count runs and shrink the variance.

    python combine_results.py ~/Scratch/results/
"""

import argparse
import glob
import json
import os

import numpy as np

from train_patterns import corrected_t_test


def load_parts(paths):
    parts = []
    for p in sorted(paths):
        with open(p) as fh:
            d = json.load(fh)
        d["_path"] = p
        parts.append(d)
    return parts


def check_compatible(parts):
    """Refuse to merge parts that are not the same experiment."""
    ref = parts[0]
    keys = ("cohort", "task", "classes", "folds", "epochs", "arms")
    for d in parts[1:]:
        for k in keys:
            if d.get(k) != ref.get(k):
                raise SystemExit(
                    f"REFUSING TO MERGE: {os.path.basename(d['_path'])} differs "
                    f"from {os.path.basename(ref['_path'])} on {k!r}\n"
                    f"  {os.path.basename(ref['_path'])}: {ref.get(k)!r}\n"
                    f"  {os.path.basename(d['_path'])}: {d.get(k)!r}\n"
                    + ("  Cohort differs -- the slide set changed between tasks "
                       "(new segmentation?). These are separate experiments and "
                       "cannot be pooled." if k == "cohort" else ""))
    # A part is identified by the (seed, fold) pairs it actually ran, not by its
    # seeds: --fold splits one seed across several parts, and keying on the seed
    # alone would reject that as a duplicate.
    seen = {}
    for d in parts:
        for r in d["runs"]:
            key = (r["seed"], r.get("fold"))
            if key in seen:
                where = "seed {}{}".format(
                    key[0], "" if key[1] is None else f" fold {key[1]}")
                raise SystemExit(
                    f"REFUSING TO MERGE: {where} appears in both "
                    f"{os.path.basename(seen[key])} and "
                    f"{os.path.basename(d['_path'])}. Double-counting runs would "
                    "understate the variance.")
            seen[key] = d["_path"]
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--baseline", default="abundance-only",
                    help="arm every other arm is tested against. Default is "
                         "the triviality control; name a pairwise arm for the "
                         "hypergraph-vs-pairwise comparison. Run once per "
                         "question rather than printing every pair -- each "
                         "test is another comparison you have to report")
    ap.add_argument("--save", default=None,
                    help="write the merged numbers to this JSON. The per-seed "
                         "files remain the record; this is the aggregate a "
                         "table or a slide is built from, so it does not have "
                         "to be re-derived by re-running this command")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.results_dir, args.glob))
    if not paths:
        raise SystemExit(f"no files matching {args.glob} in {args.results_dir}")
    parts = load_parts(paths)
    ref = check_compatible(parts)

    # dedupe: with --fold, each of a seed's folds is its own part and reports
    # that seed, so the raw list holds every seed k times
    seeds = sorted({s for d in parts for s in d["seeds"]})
    n_runs = sum(len(d["runs"]) for d in parts)
    print(f"merged {len(parts)} part(s) | seeds {seeds} | {n_runs} runs/arm")
    print(f"cohort {ref['cohort']} | {ref['n_slides']} slides | "
          f"{ref['n_patients']} patients")
    print(f"task={ref['task']} | {len(ref['classes'])} classes: {ref['classes']}")
    print(f"class counts {ref['class_counts']}")
    print(f"folds={ref['folds']} epochs={ref['epochs']}\n")

    # Concatenate in (seed, fold) order so every arm keeps the same run ordering
    # and the paired test compares like with like. Ordering by seed alone would
    # leave fold-split parts of one seed in filesystem order, which differs
    # between arms only by accident -- and mispaired scores fail silently.
    def part_key(d):
        return min((r["seed"], r.get("fold") if r.get("fold") is not None else 0)
                   for r in d["runs"])
    order = sorted(range(len(parts)), key=lambda i: part_key(parts[i]))
    def stack(arm, field):
        return np.concatenate([np.array(parts[i]["scores"][arm][field])
                               for i in order])

    n_te = float(np.mean([r["n_test"] for d in parts for r in d["runs"]]))
    n_tr = float(np.mean([d["n_train_mean"] for d in parts]))

    base = args.baseline
    known = ["abundance-only"] + list(ref["arms"])
    if base not in known:
        raise SystemExit(f"--baseline {base!r} not in this run: {known}")
    ab, ab_f1 = stack(base, "acc"), stack(base, "f1")
    summary = {k: ref.get(k) for k in
               ("cohort", "n_slides", "n_patients", "task", "classes",
                "class_counts", "folds", "epochs", "majority_baseline")}
    summary.update(baseline=base, seeds=seeds, n_runs=n_runs,
                   n_test_mean=n_te, n_train_mean=n_tr, arms={}, ablations={})
    summary["arms"][base] = dict(
        acc=float(ab.mean()), acc_sd=float(ab.std()),
        f1=float(ab_f1.mean()), f1_sd=float(ab_f1.std()))
    print(f"=== test scores over {len(seeds)} seeds x {ref['folds']} folds "
          f"({n_runs} runs/arm) ===")
    print(f"  {'majority baseline':<18} acc {ref['majority_baseline']:.3f}")
    print(f"  {base:<18} acc {ab.mean():.3f} +- {ab.std():.3f}"
          f" | macroF1 {ab_f1.mean():.3f}")
    print(f"  {'':<18} {'':<4}(the bar every arm below must clear)\n")

    for arm in ref["arms"]:
        if arm == base:
            continue
        sc, f1 = stack(arm, "acc"), stack(arm, "f1")
        beat = float((sc > ab).mean())
        mean_d, _t, p = corrected_t_test(sc - ab, n_te, n_tr)
        sig = "n/a" if np.isnan(p) else f"p={p:.3f}"
        print(f"  {arm:<18} acc {sc.mean():.3f} +- {sc.std():.3f}"
              f" | macroF1 {f1.mean():.3f}")
        print(f"  {'':<18} vs {base} {mean_d:+.3f} ({sig}, corrected) "
              f"| wins {beat:.0%} of paired runs")
        summary["arms"][arm] = dict(
            acc=float(sc.mean()), acc_sd=float(sc.std()),
            f1=float(f1.mean()), f1_sd=float(f1.std()),
            delta_vs_baseline=float(mean_d), win_rate=float(beat),
            p_corrected=None if np.isnan(p) else float(p))

    # Ablations are per-part means, so average across parts. Reported because the
    # headline score cannot distinguish "structure adds nothing" from "the model
    # never used the graph".
    abl = {}
    for d in parts:
        for arm, m in (d.get("ablations") or {}).items():
            for k, v in m.items():
                abl.setdefault(arm, {}).setdefault(k, []).append(v)
    if abl:
        print("\npermutation ablations -- the drop from full is how much the "
              "trained\nmodel relied on that input:")
        for arm, m in abl.items():
            full = float(np.mean(m.get("f1_full", [float("nan")])))
            bits = []
            for key, lbl in (("f1_graph_permuted", "graph"),
                             ("f1_abundance_permuted", "abundance")):
                if key in m:
                    v = float(np.mean(m[key]))
                    bits.append(f"{lbl} permuted {v:.3f} ({v - full:+.3f})")
            print(f"  {arm:<28} full {full:.3f} | " + " | ".join(bits))
            summary["ablations"][arm] = dict(
                full=full,
                **{lbl: float(np.mean(m[key]))
                   for key, lbl in (("f1_graph_permuted", "graph_permuted"),
                                    ("f1_abundance_permuted",
                                     "abundance_permuted")) if key in m})
        print("  a drop near zero means that input was not being used")

    if args.save:
        # A combined file from a different cohort is a different experiment.
        # Silently replacing one with the other is how a slide ends up quoting
        # numbers from a slide set that no longer exists.
        if os.path.exists(args.save):
            with open(args.save) as fh:
                old = json.load(fh)
            if old.get("cohort") != summary["cohort"]:
                raise SystemExit(
                    f"REFUSING TO OVERWRITE {args.save}: it holds cohort "
                    f"{old.get('cohort')}, this run is {summary['cohort']}. "
                    "Write to a different path.")
        with open(args.save, "w") as fh:
            json.dump(summary, fh, indent=2)
        print("")
        print(f"wrote {args.save} (baseline {base})")
    print(f"\n{base} is the bar; the majority baseline is the floor.")
    print("  - macroF1 near floor with respectable acc = collapsed to majority")
    print("  - the +- is a spread, not a standard error: repeated-CV runs share")
    print("    training data, which is what the corrected p accounts for")
    print("  - at this n, read p together with the win rate")


if __name__ == "__main__":
    main()
