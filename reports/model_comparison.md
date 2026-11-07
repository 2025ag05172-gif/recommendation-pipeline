# Model comparison

Dataset `movietweetings`, time-based split, a rating >= 8 counts as relevant. Evaluated on 1892 test users. Items a user already rated in train are never recommended.

## K = 10

| model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0253 | 0.1236 | 0.0635 | 0.0131 | 0.6672 | 1.5897 | 1.1970 |
| svd_cf | 0.0233 | 0.1097 | 0.0650 | 0.0367 | 0.6995 | 1.4588 | 1.0834 |
| item_knn_cf | 0.0261 | 0.1219 | 0.0701 | 0.2554 | 0.6841 | n/a | n/a |
| content_based | 0.0036 | 0.0177 | 0.0108 | 0.5866 | 0.1282 | n/a | n/a |
| hybrid | 0.0264 | 0.1263 | 0.0724 | 0.0986 | 0.6733 | n/a | n/a |

## K = 5

| model | Precision@K | Recall@K | NDCG@K | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0234 | 0.0541 | 0.0388 | 0.0084 | 0.4263 | 1.5897 | 1.1970 |
| svd_cf | 0.0257 | 0.0612 | 0.0473 | 0.0240 | 0.6695 | 1.4588 | 1.0834 |
| item_knn_cf | 0.0272 | 0.0653 | 0.0493 | 0.1429 | 0.6093 | n/a | n/a |
| content_based | 0.0044 | 0.0097 | 0.0081 | 0.4319 | 0.1092 | n/a | n/a |
| hybrid | 0.0284 | 0.0695 | 0.0516 | 0.0396 | 0.6069 | n/a | n/a |

Coverage = share of the catalog that appears in at least one list. Diversity = 1 - mean pairwise genre cosine similarity inside a list. RMSE and MAE are computed on all test ratings by models that predict ratings.
