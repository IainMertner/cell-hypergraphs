"""Build a SlideID,time,event CSV for the survival task.

Survival needs two columns where every other task needs one: a follow-up time
and a flag saying whether that time is a death or the last date the patient was
known to be alive. The second case is censoring, and it is why survival cannot
be recast as classification -- a patient censored at 3 years is not "survived 3
years", it is "survived AT LEAST 3 years", and dropping such patients keeps only
those with long follow-up, who are not a random subset.

    python make_survival_labels.py \\
        --clinical brca_tcga_pan_can_atlas_2018/data_clinical_patient.txt \\
        --graph-cache ~/Scratch/gc_final --out ~/Scratch/os_labels.csv

Then:
    TASK=survival LABELS=os_labels.csv ... scripts/run_local.sh

--status/--time select which endpoint: OS_* (overall survival, the default),
DSS_* (disease-specific, deaths from other causes are censored), PFS_* or DFS_*.
Overall survival is the standard reported endpoint; disease-specific is the
better-matched target for a tumour-morphology model but has more censoring.
"""

import argparse
import csv
import glob
import os

from make_labels import patient_of, read_table


def parse_status(raw):
    """cBioPortal writes '1:DECEASED' / '0:LIVING'. GDC writes bare words.

    Returns 1 for the event, 0 for censored, None if unparseable -- an
    unparseable status is dropped rather than guessed, because guessing it
    censored would silently remove deaths from every risk set.
    """
    s = (raw or "").strip()
    if not s:
        return None
    head = s.split(":")[0].strip()
    if head in ("0", "1"):
        return int(head)
    u = s.upper()
    if any(w in u for w in ("DECEASED", "DEAD", "PROGRESSION", "RECURRED")):
        return 1
    if any(w in u for w in ("LIVING", "ALIVE", "CENSORED", "DISEASEFREE",
                            "DISEASE FREE", "PROGRESSIONFREE")):
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical", required=True)
    ap.add_argument("--status", default="OS_STATUS")
    ap.add_argument("--time", default="OS_MONTHS")
    ap.add_argument("--patient-column", default=None)
    ap.add_argument("--graph-cache", default="graph_cache")
    ap.add_argument("--min-time", type=float, default=0.0,
                    help="drop patients whose follow-up is <= this. A time of "
                         "zero carries no information and breaks the risk set")
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
    for need in (pcol, args.status, args.time):
        if need is None or need not in cols:
            raise SystemExit(f"missing column {need!r}; available: {sorted(cols)}")

    surv, bad_status, bad_time = {}, 0, 0
    for r in rows:
        pid = (r.get(pcol) or "").strip()
        if not pid:
            continue
        ev = parse_status(r.get(args.status))
        if ev is None:
            bad_status += 1
            continue
        try:
            t = float((r.get(args.time) or "").strip())
        except ValueError:
            bad_time += 1
            continue
        if t <= args.min_time:
            bad_time += 1
            continue
        surv[patient_of(pid)] = (t, ev)

    slides = sorted(os.path.basename(f)[:-3] for f in
                    glob.glob(os.path.join(args.graph_cache, "*.pt"))
                    if not f.endswith("_params.pt"))
    if not slides:
        raise SystemExit("no slides found; check --graph-cache")

    out, missing = [], 0
    for sid in slides:
        rec = surv.get(patient_of(sid))
        if rec is None:
            missing += 1
            continue
        out.append((sid, f"{rec[0]:.4f}", rec[1]))

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["SlideID", "time", "event"])
        w.writerows(out)

    events = sum(e for _, _, e in out)
    times = sorted(float(t) for _, t, _ in out)
    ev_times = sorted(float(t) for _, t, e in out if e)
    med = times[len(times) // 2] if times else 0.0
    print(f"{len(slides)} slides | {len(out)} with survival | {missing} with no "
          f"clinical record")
    if bad_status:
        print(f"{bad_status} patient(s) dropped: unparseable {args.status}")
    if bad_time:
        print(f"{bad_time} patient(s) dropped: missing or non-positive {args.time}")
    print(f"\nendpoint {args.status} / {args.time}")
    print(f"  events           {events} ({events / max(len(out), 1):.1%})")
    print(f"  censored         {len(out) - events}")
    print(f"  median follow-up {med:.1f} months")
    if ev_times:
        print(f"  median time to event {ev_times[len(ev_times) // 2]:.1f} months")
    if events < 100:
        print("\n  WARNING: the number of EVENTS bounds what is learnable here, "
              "not the number\n  of slides. Under ~100 events a C-index is very "
              "wide however many slides there are.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
