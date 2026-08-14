# SVD Baseline — Video Games 5-Core

## Dataset

- Dataset: Amazon Video Games 5-Core
- Interactions: 814,586
- Users: 94,762
- Products: 25,612
- Minimum user history: 5
- Minimum product history: 5
- Duplicate user-product pairs: 0

## Preprocessing

- Null values: 0
- Invalid ratings: 0
- User IDs encoded as integer indices
- Product IDs encoded as integer indices
- Rating retained as the interaction signal
- Timestamp retained for temporal evaluation

## Train/Test Split

- Training interactions: 719,824
- Test interactions: 94,762
- Test users: 94,762
- One held-out interaction per user

## Model

- Model: SVD
- Latent factors: 50
- Recommendation type: Top-K
- K: 10
- Previously seen training items masked from recommendations

## Results

| Metric | @10 |
|---|---:|
| Precision | 0.003745 |
| Recall | 0.037452 |
| Hit Rate | 0.037452 |
| NDCG | 0.020084 |

## Interpretation

The SVD model successfully produces personalized Top-10 recommendations for every evaluated user.

The baseline achieves a Hit Rate@10 of 3.7452%, meaning approximately 3.75% of users had their held-out interaction appear in their Top-10 recommendations.

This SVD implementation serves as the initial collaborative-filtering baseline against which subsequent recommendation approaches can be compared.
