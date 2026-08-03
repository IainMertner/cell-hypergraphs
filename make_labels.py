"""Build a SlideID,label CSV from a patient-level clinical/molecular table.

Every task after the first needs the same join: clinical data is keyed by PATIENT
(TCGA-XX-YYYY) while the pipeline is keyed by SLIDE (TCGA-XX-YYYY-01Z-00-DX1...).

    python make_labels.py --clinical data_clinical_patient.txt \\
        --column ER_STATUS_BY_IHC --graph-cache graph_cache --out er.csv \\
        --keep Positive Negative

Then:
    qsub -v TASK=auto,LABEL_COL=label,LABELS=er.csv ... scripts/run_patterns_array.sh

Reports how many slides matched and the class balance, because a task that looks
viable in the clinical table can still be too thin once restricted to the slides
actually segmented.
"""

import argparse
import csv
import glob
import os


def read_table(path):
    """cBioPortal and GDC exports are TSV with optional leading # comment rows."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in fh if not r.startswith("#")]
    return list(csv.DictReader(rows, delimiter="\t"))


def patient_of(tcga_id):
    """Any TCGA barcode -> patient. Both sides of the join are normalised.

    Slides are TCGA-A2-A0CQ-01Z-00-DX1, patient files key on TCGA-A2-A0CQ and
    sample files on TCGA-A2-A0CQ-01, so truncating everything to the first three
    fields lets either clinical file be used without a flag.
    """
    return "-".join(tcga_id.split("-")[:3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True,
                    help="patient-level TSV (cBioPortal data_clinical_patient.txt)")
    ap.add_argument("--column", required=True, help="column holding the label")
    ap.add_argument("--patient-column", default=None,
                    help="id column; default: first of PATIENT_ID, SAMPLE_ID, "
                         "bcr_patient_barcode, Patient ID that exists. Sample "
                         "ids work too -- both sides are truncated to patient")
    ap.add_argument("--graph-cache", default="graph_cache",
                    help="restrict to slides that actually have graphs")
    ap.add_argument("--slide-dir", default=None,
                    help="alternative source of slide ids: a cellvit_out tree")
    ap.add_argument("--keep", nargs="*", default=None,
                    help="keep only these label values (drops Indeterminate, "
                         "'Not Performed', blanks and similar)")
    ap.add_argument("--positive", nargs="*", default=None,
                    help="collapse to binary: these values become the positive "
                         "class, everything surviving --keep becomes 'other'. "
                         "For a 5-class target where only one class is "
                         "morphologically distinct, or where the rest are too "
                         "thin to split")
    ap.add_argument("--positive-name", default=None,
                    help="name for the positive class (default: joined values)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_table(args.clinical)
    if not rows:
        raise SystemExit(f"{args.clinical} has no data rows")
    cols = rows[0].keys()
    pcol = args.patient_column
    if pcol is None:
        for c in ("PATIENT_ID", "SAMPLE_ID", "bcr_patient_barcode",
                  "Patient ID", "Sample ID", "case_submitter_id"):
            if c in cols:
                pcol = c
                break
    if pcol is None or pcol not in cols:
        raise SystemExit(f"no patient id column found; pass --patient-column. "
                         f"Available: {sorted(cols)}")
    if args.column not in cols:
        raise SystemExit(f"no column {args.column!r}; available: {sorted(cols)}")

    label_of, conflicts = {}, 0
    for r in rows:
        pid, val = (r.get(pcol) or "").strip(), (r.get(args.column) or "").strip()
        if not pid or not val:
            continue
        pid = patient_of(pid)
        # sample-keyed files can carry several samples per patient; a patient
        # whose samples disagree is dropped rather than silently taking one
        if pid in label_of and label_of[pid] != val:
            label_of[pid] = None
            conflicts += 1
        elif pid not in label_of:
            label_of[pid] = val

    if args.slide_dir:
        slides = sorted(os.path.basename(os.path.dirname(p)) for p in
                        glob.glob(os.path.join(args.slide_dir, "*", "cells_cache.npz")))
    else:
        slides = sorted(os.path.basename(f)[:-3] for f in
                        glob.glob(os.path.join(args.graph_cache, "*.pt"))
                        if not f.endswith("_params.pt"))
    if not slides:
        raise SystemExit("no slides found; check --graph-cache / --slide-dir")

    out, missing, dropped = [], 0, {}
    for sid in slides:
        val = label_of.get(patient_of(sid))
        if val is None:
            missing += 1
            continue
        if args.keep is not None and val not in args.keep:
            dropped[val] = dropped.get(val, 0) + 1
            continue
        if args.positive is not None:
            val = (args.positive_name or "+".join(args.positive)
                   ) if val in args.positive else "other"
        out.append((sid, val))

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SlideID", "label"])
        w.writerows(out)

    counts = {}
    for _, v in out:
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(slides)} slides | {len(out)} labelled | {missing} with no "
          f"clinical record")
    if conflicts:
        print(f"{conflicts} patient(s) dropped: samples disagreed on "
              f"{args.column}")
    if dropped:
        print("dropped by --keep: " + ", ".join(f"{k} ({n})"
                                                for k, n in sorted(dropped.items())))
    print(f"\n{args.column} over the segmented cohort:")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24} {n:>5}  ({n / max(len(out), 1):.1%})")
    if counts:
        smallest = min(counts.values())
        print(f"\nsmallest class {smallest} slides -> ~{smallest / 5:.1f} per fold "
              f"at 5-fold CV")
        if smallest < 25:
            print("  WARNING: thin. Patient-grouped folds may leave a class absent "
                  "from a test fold,\n  which depresses macro-F1 for reasons "
                  "unrelated to the model.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
