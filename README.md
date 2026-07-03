# cell-hypergraphs

## Pipeline

* H&E slides from TCGA-BRCA
* Cell segmentation and classification with CellVit++
* Five graph construction arms:
    - Two field baselines (kNN and Delaunay)
    - Two controls (hypergraph and its flattened pairwise counterpart)
    - Main hypergraph method (control hypergraph with DeepSets aggregation layer)
* Downstream tasks:
    - Cell-level type prediction using 30% masking