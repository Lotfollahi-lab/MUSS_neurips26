# Reference results

[Overview](../README.md) · [Evaluation commands](reproduction.md#evaluation) ·
[Metric definitions](metrics.md)

## Gene-expression prediction

Reference scores for the released fine-tuned checkpoints on the original processed test
data: **PCC / CCC / R²**, measured on `log10(CPM + 1)`. Locally recomputed embeddings may
yield small numerical differences.

| Setting | PBMC10K | BMMC | Open Problems | HCA bone |
|---|---|---|---|---|
| Seen sequence · seen cell types | 0.903 / 0.889 / 0.799 | 0.971 / 0.967 / 0.935 | 0.995 / 0.994 / 0.988 | 0.958 / 0.945 / 0.897 |
| Seen sequence · unseen cell types | 0.874 / 0.862 / 0.748 | 0.934 / 0.919 / 0.837 | 0.976 / 0.975 / 0.951 | 0.940 / 0.928 / 0.863 |
| Unseen sequence · seen cell types | 0.679 / 0.655 / 0.403 | 0.741 / 0.728 / 0.510 | 0.754 / 0.734 / 0.551 | 0.683 / 0.611 / 0.427 |
| Unseen sequence · unseen cell types | 0.678 / 0.659 / 0.401 | 0.734 / 0.712 / 0.458 | 0.760 / 0.738 / 0.559 | 0.683 / 0.609 / 0.419 |

## Cell clustering

Reference scores on the held-out PBMC169K batches, reported as mean ± standard deviation:

| Held-out PBMC169K batch | ARI | FMI | NMI | AMI |
|---|---|---|---|---|
| VIB_10xv1_1 | 0.673 ± 0.001 | 0.742 ± 0.001 | 0.745 ± 0.001 | 0.744 ± 0.001 |
| BIO_ddseq_1 | 0.519 ± 0.001 | 0.684 ± 0.001 | 0.594 ± 0.001 | 0.593 ± 0.001 |
