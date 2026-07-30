# Dissertation write-up template

Structure with per-section notes on what this project already has, what is still
missing, and where the honest caveats belong. Indicative lengths assume a
~12,000 word MSc dissertation — scale to your actual limit.

---

## Abstract (~250 words)

One paragraph each: the problem, what you did, what you found, what it means.
Write it last.

If the result is null, say so in the abstract. "We find no evidence that X" is a
finding; burying it reads as evasion.

---

## 1. Introduction (~1,000 words)

- **Motivation.** TILs predict prognosis and treatment response in breast cancer.
  Their spatial *arrangement* — not just abundance — is clinically graded, but
  computational methods mostly count.
- **The gap.** Cell graphs represent tissue as cells + pairwise edges. A
  neighbourhood is a *set*, and a clique expansion of a set is not injective:
  `{a,b,c}` and the three pairs `{a,b},{b,c},{a,c}` expand identically. Pairwise
  graphs therefore cannot distinguish groupings that a hypergraph can.
- **Research question.** State it in one sentence. Something like: *does
  representing cell neighbourhoods as hyperedges with sum aggregation improve
  prediction of TIL spatial pattern over the field-standard pairwise cell graph,
  once abundance and model capacity are controlled?*
- **Contributions.** Be honest and specific. Candidates:
  1. A controlled comparison of pairwise vs hypergraph cell representations on a
     clinically graded spatial task.
  2. A triviality control (abundance-only) establishing whether the task is
     spatial at all.
  3. Structural characterisation of the constructions (clique-expansion ratio).
  4. Methodological: identification and correction of confounds that would have
     invalidated the comparison (see §7).

---

## 2. Background (~2,000 words)

- **Cell graphs in computational pathology.** CGC-Net, cell-graph GNNs, what node
  features are conventional (type, morphology).
- **Hypergraph neural networks.** HGNN, HyperGCN, and specifically why mean
  aggregation (`HypergraphConv`) is near-equivalent to a GCN on the clique
  expansion — its Laplacian *is* a clique-expansion Laplacian. Deep Sets and
  permutation-invariant set functions.
- **Multiple-instance learning for WSIs.** ABMIL (Ilse et al. 2018), CLAM.
  Why slide-level labels force a bag formulation.
- **TILs and the Saltz taxonomy.** Define Brisk/Non-Brisk and
  Diffuse/Band-like/Focal/Multifocal. **Verify these definitions against the
  source** — the arrangement axis used in this work depends on them.

---

## 3. Data (~1,200 words)

- **Cohort.** TCGA-BRCA diagnostic slides. State: total available (1133 in the
  GDC manifest), how many downloaded, segmented, graphed, and labelled. Report
  the numbers actually used, not the aspiration.
- **Segmentation.** CellViT++ (SAM backbone, PanNuke taxonomy), 5 nuclear classes.
  Per-cell centroid, class, and 5 morphology descriptors.
- **Region extraction.** 4000px tiles, ≥2000 cells, ≥50 inflammatory cells.
  Justify the thresholds and report how many slides/regions survive them.
- **Labels.** Saltz TIL PatternLabels. Report the class distribution — yours is
  imbalanced (31/13/50/19 at n=113), which matters for every metric.
- **Reproducibility.** Each run records a cohort fingerprint (hash of the exact
  slide set). Quote it alongside every result table.

---

## 4. Methods (~2,500 words)

### 4.1 Graph constructions
`pw-knn` (k-NN, k=5, 35µm cap) and `hg-knn` (one hyperedge per cell containing
{cell + k nearest}). Emphasise these are matched: same k, same neighbour set,
same node features. The *only* difference is set vs pairs.

### 4.2 Structural characterisation
Clique-expansion ratio and hyperedge cardinality per construction. Report
cohort-wide figures from `stats_table.py`, not a single slide.

### 4.3 Models
Node features (10-d: 5 one-hot type + 5 z-scored morphology). Encoder →
readout (mean ⊕ sum) → fixed-width projection → attention pool → linear head.
Give the `DeepSetsHyperConv` equations; the sum-pool is the mechanism.

### 4.4 Capacity matching
State the parameter counts. Explain that the region encoders emit a fixed width
so the MIL pool and head are identical across arms *by construction*, and only
the encoder differs. Report the final spread (~1.008×).

### 4.5 Controls
- `abundance-only`: MLP on per-slide cell-type fractions. No spatial structure.
- Majority baseline.
- For arrangement tasks: the briskness-only rule.
State explicitly that an arm must clear the *highest* of these, not the lowest.

### 4.6 Evaluation
Patient-grouped stratified k-fold, folds resampled per seed. Early stopping on
validation only. Metrics: accuracy and macro-F1 (macro-F1 exposes majority
collapse, which accuracy cannot). Significance: Nadeau–Bengio corrected
resampled t-test, because repeated-CV runs share training data and an
uncorrected test overstates badly — quantify by how much.

---

## 5. Experimental design (~800 words)

- Tasks: 4-class `pattern4`, and the 2-class arrangement collapse.
- **The collinearity problem.** In this label set Brisk pairs only with
  Diffuse/Band-like and Non-Brisk only with Focal/Multifocal, so a naive 2-class
  collapse *is* the abundance axis. Describe the crossed mapping and report the
  residual briskness-only accuracy — on this cohort abundance alone still reaches
  0.717 vs a 0.558 majority baseline, so the task is only partly spatial.
- Sweep size, epochs, and why (early stopping must terminate runs, not the cap —
  a low cap can penalise whichever arm converges slower and manufacture a null).
- State what was frozen before the confirmatory run, and what was chosen from
  pilot runs. Be explicit; it is the difference between confirmatory and
  exploratory.

---

## 6. Results (~1,500 words)

Table per task: majority baseline, abundance-only, each arm, with accuracy,
macro-F1, mean difference vs control, corrected p, and win rate. Include the
cohort fingerprint and n.

Report macro-F1 prominently. On this cohort a majority predictor scores
**0.153** — quote that number so a collapsed model is visible to the reader.

If null: say so plainly, then use §7 and §8 to say what that does and does not
rule out.

---

## 7. Methodological findings (~1,000 words)

This is a legitimate contribution and worth its own section, especially if the
headline result is null. Each item: what was wrong, how it was detected, what it
would have done to the result.

- Morphology features unnormalised on the cluster path — raw pixel areas
  (hundreds) concatenated with 0/1 one-hot columns, swamping the type signal.
- Class weighting silently inert — `F.cross_entropy(reduction='mean')` divides by
  the *sum of weights*, so on a single sample the weight cancels exactly. The
  imbalance correction never reached the optimiser.
- Capacity confound — matching encoders widened the downstream pool and head,
  leaving the baseline with 1.77× the parameters of the test arm. Direction of
  bias unsigned at this sample size, which is worse than a known direction.
- Arrangement task collinear with abundance — an earlier mapping was *exactly*
  the briskness axis.
- Cache invalidation — precomputed graphs store finished feature tensors, so a
  feature-encoding change cannot be repaired at load time; versioning added.

---

## 8. Discussion (~1,500 words)

- Interpret against the controls, not in isolation.
- **If null**, distinguish the hypotheses it cannot separate:
  - higher-order structure genuinely adds nothing here;
  - sum-vs-mean is the wrong lever (needs the mean-aggregation control);
  - the task is not spatial enough (abundance control already speaks to this);
  - the architecture cannot express the relevant structure (see below);
  - insufficient power at this n.
- **Architectural limitation worth raising.** Attention pooling is a
  softmax-weighted *mean* over regions, which discards how *many* regions are
  TIL-rich — and that count is precisely the focal-vs-multifocal distinction.
  There is also no positional encoding between regions, so inter-region spatial
  arrangement is not representable at all. A whole-slide arrangement label may
  therefore be partly outside the model's hypothesis space.

---

## 9. Limitations (~600 words)

n and slide-level labels; class imbalance; single cohort; single segmentation
model; region thresholds; the residual abundance confound in the arrangement
task; the clique-expansibility spectrum being narrower than assumed; results at
different cohort sizes being separate experiments rather than refinements.

---

## 10. Conclusion and future work (~600 words)

The mean-aggregation control; region-level labels if they become available;
count-preserving pooling (sum rather than softmax-mean over regions); positional
encoding between regions; the remaining constructions along the expansibility
spectrum if any arm clears the control.

---

## Appendices

Full parameter counts; per-fold results; construction parameters; cohort
fingerprints; the structural statistics table; software versions.

---

### Notes on honest framing

- A well-characterised null is a result; an uncharacterised one is not. Most of
  §7 and §8 exists to make yours the former.
- Report the corrected p, and state that repeated-CV runs are not independent.
- Quote the majority baseline and macro-F1 floor next to every result so a
  reader can see collapse for themselves.
- Say which decisions were made before seeing results and which after.
