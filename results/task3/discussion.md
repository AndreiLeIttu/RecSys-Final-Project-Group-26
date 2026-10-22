# Task 3.2: What changed after reranking?

We evaluated EASE, FeatureCombination, the original and tuned weighted hybrids,
and SwitchingHybrid, with Random and Pop as reference baselines. Each method reranks the same 100 candidates into a top 10.
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

## After merging the hybrid experiments

The rerun includes all three new variants. The original four models' results
are unchanged. Both regression hybrids are refitted on the same 471 validation
users; reranker strengths are chosen on the other 472 users. The tuned hybrid
keeps Task 1's compact component set and negative ratio 10.

| TunedWeightedHybrid method | Strength | nDCG@10 | Diversity | Genre mismatch | UPD | Exposure gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No reranking | 0 | 0.3328 | 0.7287 | 0.6730 | 0.2788 | 0.7911 |
| Diversity | 0.25 | 0.3279 | 0.7955 | 0.5455 | 0.2797 | 0.7922 |
| Calibration | 0.10 | 0.3323 | 0.7606 | 0.3859 | 0.2769 | 0.7896 |
| User fairness | 0.25 | 0.3285 | 0.7197 | 0.6765 | 0.1588 | 0.6986 |
| Item fairness | 0.25 | 0.3323 | 0.7221 | 0.6790 | 0.2357 | 0.7521 |

For this hybrid, diversity rises by 9.2% with a 1.5% relative nDCG loss.
Calibration reduces genre mismatch by 42.7% with a 0.1% nDCG loss. User fairness
reduces UPD by 43.1% with a 1.3% nDCG loss; item fairness reduces the exposure gap
by 4.9% with a 0.1% nDCG loss. The compact hybrid is not better on every measure:
its refitted baseline nDCG is slightly below the original hybrid's 0.3335.

FeatureCombination starts at nDCG 0.2444. Diversity rises from 0.6595 to 0.7689
with a 1.2% nDCG loss, and calibration reduces genre mismatch from 0.7774 to
0.4199 with a 0.9% loss. User fairness reduces UPD from 0.2683 to 0.1395, but
costs 4.1% test nDCG. Its selected checkpoint was unavailable locally, so this
run retrained the selected configuration under `artifacts/task3/checkpoints/`.
The recovered baseline differs from Task 2's archived 0.2482; compare rerankers
against the baseline from this run. The exact cause of that difference has not
been established, and the archived Task 1/2 results were preserved.

Task 1 selected the SLIMElastic endpoint for SwitchingHybrid. Therefore this
variant uses SLIMElastic for every user. Its baseline nDCG of 0.3269 matches the
archived SLIMElastic result; it provides no evidence that activity switching
improves recommendations. Reranking it still improves the targeted measures.

The expanded run has 168 validation trials and 35 test comparisons across seven
models. All selected settings meet the validation accuracy budget. Every one of
the 330,050 saved test recommendations was checked for uniqueness, candidate
membership, and exclusion of padding and observed items.

## Limits

- This is one split, without confidence intervals or significance tests.
- Both regression hybrids use a separate half of validation users for coefficient
  fitting, but earlier base-model and hybrid structure/ratio/switch selection
  used the full validation set.
- A 5% validation accuracy budget does not guarantee the same test loss. Pop's
  item-fairness setting lost about 6% test nDCG.
- A reranker cannot retrieve movies outside its candidate list. Stronger
  interventions and different candidate sizes remain useful follow-up checks.
- The exposure target is our explicit fairness choice, not a universal standard.

`tradeoffs.svg` shows the validation trade-offs, with selected settings circled.
`protocol.json` records the checkpoints, split, weights, metrics, and settings.
