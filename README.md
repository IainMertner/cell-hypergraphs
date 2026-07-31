# cell-hypergraphs

Cell graph / hypergraph methods for TCGA-BRCA histopathology.

```
precompute_graphs.py        stage 1 CLI - cells -> graph cache
train_patterns.py           stage 2 CLI - graph cache -> results
stats_table.py              structural diagnostics
models.py                   Deep Sets layer, MIL classifier, capacity matching

graphs/
  __init__.py               build() dispatch, arm list, construction params
  cells.py                  .npz loading, morphology scaling, region tiling
  common.py                 node features, shared assembly, structural stats
  constructions/
    pw_knn.py               baseline: k-NN cell graph
    hg_knn.py               primary: {cell + k nearest} as one hyperedge

segmentation/               slides -> cells
  next_batch.sh             derive the next download batch from the manifest
  download_slides.sh        GDC download
  make_slide_list.sh        build the array-job slide list
  cellvit_chunked.sh        CellViT inference
  cache_cells.py            cells.json -> compact .npz

scripts/                    cluster job submission
  precompute.sh             stage 1
  run_patterns.sh           stage 2

env.sh                      cluster environment, sourced by every job script
gdc_manifest_brca_dx.txt    full 1133-slide cohort
```