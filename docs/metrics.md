# Metric definitions

[Overview](../README.md) · [Evaluation commands](reproduction.md#evaluation) ·
[Reference results](results.md)

## Gene-expression prediction

For each cell type with at least 10 test cells, expression is averaged into a pseudobulk
profile, normalized to CPM over the full gene panel and transformed to `log10(CPM + 1)`.
Metrics compare predicted and measured expression across seen or unseen genes; reported
scores are unweighted means over cell types.

`main_table.csv` reports PCC (Pearson correlation), CCC (concordance correlation coefficient)
and R² (coefficient of determination) for the four combinations of seen/unseen sequences and
cell types. `metrics_summary.csv` and `metrics_per_celltype.csv` also include Spearman
correlation, MAE and RMSE in linear and log space. The evaluation defaults to the test split.

## Cell clustering

Encoder outputs are averaged over each cell's peaks, reduced to 50 principal components,
and clustered with k-means using the number of annotated cell types as `k`.
`clustering_summary.csv` reports the mean and standard deviation over 10 fixed seeds for
ARI (adjusted Rand index), FMI (Fowlkes–Mallows index), NMI (normalized mutual information)
and AMI (adjusted mutual information).
