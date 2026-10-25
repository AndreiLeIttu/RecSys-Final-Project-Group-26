# Task 2.5

Training-only groups separate users by median activity and median head-item preference. Head items are the top 20% of the catalog by training count. All models use the same full-catalog test protocol. The table contrasts accuracy, novelty and popularity exposure. Higher activity can raise MRR through more relevant opportunities while reducing recall through a larger denominator. Head/tail recall must therefore be interpreted alongside exposure, rather than treating popularity concentration as retrieval success. Random provides a control for novelty without personalization. These are observational groups, not causal or demographic fairness estimates. Coverage depends on group size; cold-item and small-group conclusions require caution. Full model/group metrics, support counts and paired uncertainty intervals accompany this table.
For WeightedHybrid, low/high activity MRR is 0.354/0.714, but recall is 0.350/0.214. Head exposure is 98.9%; head/tail micro recall is 0.340/0.010. The ensemble retains popularity concentration despite combining components.

| Model | Activity | Users | MRR | Recall | Novelty (bits) | Diversity |
|---|---|---:|---:|---:|---:|---:|
| Random | low | 477 | 0.006 | 0.006 | 11.92 | 0.750 |
| Random | high | 466 | 0.025 | 0.005 | 12.21 | 0.752 |
| Pop | low | 477 | 0.065 | 0.088 | 7.98 | 0.709 |
| Pop | high | 466 | 0.317 | 0.079 | 8.10 | 0.714 |
| WeightedHybrid | low | 477 | 0.354 | 0.350 | 8.49 | 0.719 |
| WeightedHybrid | high | 466 | 0.714 | 0.214 | 8.54 | 0.733 |
| FeatureCombination | low | 477 | 0.282 | 0.244 | 8.43 | 0.629 |
| FeatureCombination | high | 466 | 0.588 | 0.161 | 8.58 | 0.668 |
| GroupAwareHybrid | low | 477 | 0.354 | 0.350 | 8.49 | 0.719 |
| GroupAwareHybrid | high | 466 | 0.691 | 0.204 | 8.58 | 0.737 |

| Model | Item group | Relevant interactions | Recall (micro) | Exposure share |
|---|---|---:|---:|---:|
| Pop | head | 6100 | 0.127 | 1.000 |
| Pop | tail | 3496 | 0.000 | 0.000 |
| WeightedHybrid | head | 6100 | 0.340 | 0.989 |
| WeightedHybrid | tail | 3496 | 0.010 | 0.011 |
| GroupAwareHybrid | head | 6100 | 0.333 | 0.989 |
| GroupAwareHybrid | tail | 3496 | 0.010 | 0.011 |

# Task 2.6

Hypothesis: activity and popularity preference identify complementary model strengths. A four-cell switch selects full rankings by validation MRR; cells with fewer than 30 validation users or gains no greater than 0.005 retain the global reference. The reference is WeightedHybrid; final cell choices are {0: 'WeightedHybrid', 1: 'WeightedHybrid', 2: 'WeightedHybrid', 3: 'ItemKNN'}. The gate is cross-fitted on reserved validation users before fitting its final rule. Test labels enter neither grouping nor selection. The switch does not improve test MRR over its validation-selected reference. Novelty, diversity and coverage changes qualify any accuracy gain. If multiple cells choose the same model, these signals provide limited evidence for specialization. Base hyperparameters and the pre-existing hybrid structure reused full validation, so gate cross-fitting is not fully nested evaluation. One seed and exploratory group comparisons limit generalization. Freeze the rule and replicate on fresh splits before claiming a superior hybrid.

| Metric | Reference | Group-aware hybrid | Change |
|---|---:|---:|---:|
| MRR@10 | 0.5317 | 0.5201 | -0.0116 |
| NDCG@10 | 0.3334 | 0.3268 | -0.0066 |
| Novelty@10 | 8.5105 | 8.5303 | +0.0197 |
| IntraListDiversity@10 | 0.7256 | 0.7276 | +0.0020 |
| Coverage@10 | 0.2247 | 0.2265 | +0.0018 |
