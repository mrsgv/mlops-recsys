# Two-Tower V2.1 Experiment

## Objective

Evaluate whether replacing in-batch negatives with explicit
user-history-aware negative sampling improves the metadata-aware
Two-Tower recommender.

## Configuration

- Embedding dimension: 64
- Hidden dimension: 128
- Categorical embedding dimension: 16
- Batch size: 1024
- Epochs: 5
- Learning rate: 0.001
- Explicit negatives per positive: 5
- Negative sampling seed: 42
- Top-K: 10

## Evaluation Protocol

- Video Games dataset
- Chronological train/test split
- Latest interaction held out per eligible user
- Previously seen training items filtered from recommendations
- Common Precision@K, Recall@K, Hit Rate@K and NDCG@K evaluator

## Results

| Epoch | Loss | Precision@10 | Recall@10 | Hit Rate@10 | NDCG@10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.395462 | 0.001645 | 0.016452 | 0.016452 | 0.007263 |
| 2 | 1.308989 | 0.001234 | 0.012336 | 0.012336 | 0.005532 |
| 3 | 1.291284 | 0.001305 | 0.013054 | 0.013054 | 0.005966 |
| 4 | 1.283980 | 0.000777 | 0.007767 | 0.007767 | 0.003493 |
| 5 | 1.278332 | 0.000453 | 0.004527 | 0.004527 | 0.001985 |

## Best Checkpoint

- Best epoch: 1
- Recall@10: 0.016452
- NDCG@10: 0.007263

## Interpretation

Explicit user-history-aware negative sampling substantially improved
neural retrieval compared with Two-Tower V2.

However, the best neural result remains below the iALS benchmark.

Training loss decreased throughout training while ranking metrics
deteriorated after epoch 1, demonstrating that training loss alone
is not an adequate model-selection criterion for this recommendation
task.

## Decision

Freeze further neural-model experimentation after V2.1.
Proceed to model comparison and MLOps integration.