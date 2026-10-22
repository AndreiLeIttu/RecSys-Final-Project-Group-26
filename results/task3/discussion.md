# Task 3.2: What changed after reranking?

We evaluated EASE and the refitted weighted hybrid, with Random and Pop as
reference baselines. Each method reranks the same 100 candidates into a top 10.
Settings were selected on validation data before running the test comparison.

The table below shows the weighted hybrid on the test set. Higher nDCG and
diversity are better; lower values are better for the other three measures.

| Method | Strength | nDCG@10 | Diversity | Genre mismatch | UPD | Exposure gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reranking | 0 | 0.3335 | 0.7260 | 0.6878 | 0.2768 | 0.7889 |
| Diversity | 0.25 | 0.3280 | 0.7986 | 0.5504 | 0.2779 | 0.7902 |
| Calibration | 0.10 | 0.3315 | 0.7594 | 0.3836 | 0.2750 | 0.7872 |
| User fairness | 0.25 | 0.3292 | 0.7164 | 0.6952 | 0.1516 | 0.6898 |
| Item fairness | 0.25 | 0.3335 | 0.7183 | 0.6906 | 0.2264 | 0.7405 |

Diversification increased genre variety by about 10%, with a 1.7% relative
nDCG loss. It selects less-similar movies, but does not directly address exposure.
Calibration reduced genre mismatch by about 44%, with a 0.6% nDCG loss. It
preserves the user's genre mix rather than maximizing variety for everyone.

User fairness reduced popularity deviation by about 45%, with a 1.3% nDCG loss.
Item fairness reduced the exposure gap by about 6%, with almost unchanged nDCG.
However, its remaining gap is large: relevance and the available candidates
limit how closely it can reach the catalog-proportional target. EASE showed
similar directions of change; see `test.csv` for all measurements and deltas.

These fairness objectives can conflict. Matching an individual user's preference
for popular movies can concentrate exposure. For Random, user fairness improved
UPD but increased exposure inequality. Also, the hybrid's activity-group nDCG
gap increased from 0.1359 to 0.1389 with user fairness and 0.1427 with item fairness.
Better popularity matching or exposure does not imply equal recommendation quality.

## Limits

- This is one split, without confidence intervals or significance tests.
- The hybrid uses a separate half of validation users for coefficient fitting,
  but earlier base-model selection used the full validation set.
- A 5% validation accuracy budget does not guarantee the same test loss. Pop's
  item-fairness setting lost about 6% test nDCG.
- A reranker cannot retrieve movies outside its candidate list. Stronger
  interventions and different candidate sizes remain useful follow-up checks.
- The exposure target is our explicit fairness choice, not a universal standard.

`tradeoffs.svg` shows the validation trade-offs, with selected settings circled.
`protocol.json` records the checkpoints, split, weights, metrics, and settings.
