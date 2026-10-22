# Tasks 3.1 and 3.2

We keep the recommender fixed and change which movies appear in its final list.
Every method starts with the same top 100 unseen candidates and returns 10 movies.
The input scores are normalized to [0, 1] within each user's candidate list.
Ties use the original candidate order: score first, then internal movie ID.

## Rerankers

Each method builds the list greedily, choosing the next movie with the best
`(1 - strength) * mean_relevance + strength * benefit`.

| Method | Benefit | What it tries to do |
| --- | --- | --- |
| Diversity | Mean pairwise genre cosine distance | Avoid a list of very similar movies |
| Calibration | Negative genre miscalibration | Match the user's mix of genres |
| User fairness | Negative User Popularity Deviation (UPD) | Respect the user's taste for popular or less-popular movies |
| Item fairness | Negative head/tail exposure gap | Give less-popular movies more opportunity to appear |

For calibration, each movie splits one unit of weight equally among its genres.
User genre profiles average training-history movies. Miscalibration is
`KL(p || 0.99*q + 0.01*p)` using natural logs, as in the calibration lecture.
Recommendation genre distributions use equal weights per movie.

The popularity head is the top 20% of catalog items by training interaction count;
the tail contains the rest. Count ties use movie ID. UPD is total variation
between the user's historical and recommended head/tail proportions.
Item fairness targets exposure proportional to catalog group size (about 20/80),
with attention discounted by `1/log2(rank + 1)`. The greedy objective uses this
target per list; evaluation measures the aggregate exposure gap.

These are specific fairness choices. Matching popularity preferences does not
guarantee equal accuracy across users. Catalog-proportional exposure is not a
claim about movie quality, demographics, or equal exposure for every movie.

## Evaluation

- Reuse the seed-2020 Task 1 training/validation/test split and selected settings.
- Refresh missing or outdated checkpoints in `artifacts/task3/checkpoints/`.
  Task 1's checkpoints and results are left intact.
- Compare EASE, FeatureCombination, WeightedHybrid, TunedWeightedHybrid,
  SwitchingHybrid, Random, and Pop.
- Refit both regression hybrids on validation interactions from 471 users.
  The original uses all 11 individual models and negative ratio 5; the tuned
  hybrid reuses Task 1's selected components and ratio (SLIMElastic, EASE, NeuMF,
  ItemKNN; ratio 10). Tune rerankers on the other 472 users, chosen with a seeded shuffle.
- Reuse Task 1's switching choice. Its current winner is the SLIMElastic endpoint,
  so no activity switch is applied. If a quantile switch is selected in a future
  Task 1 run, its threshold is recomputed from the fit users' training activity.
- Earlier base-model selection, early stopping, and hybrid structure/ratio/switch
  selection used the full validation set.
  This is still development data, not a fully independent nested-validation study.
- Try strengths `0, 0.1, 0.25, 0.5, 0.75, 1`. Select the best method-specific
  objective within a **5% relative validation nDCG loss** from its own baseline.
  Ties prefer lower strength. Zero is always available as a fallback.
- Freeze every setting before evaluating on the 943 test users. Test labels are
  never used for selection. The 5% validation budget is not a guarantee on test.
- Profiles and popularity always use training interactions only. Validation
  masks training items; test masks training and all validation items.

Task 2 supplies accuracy, coverage, diversity, novelty, and average popularity.
We add genre miscalibration, UPD, discounted exposure gap and exposure Gini
(including items never recommended). Activity GRU is the absolute nDCG gap
between users at/below the training-history median and those above it.

The Task 3 regression hybrids are refitted, so their baselines can differ from
Task 1's saved hybrids. Stable tie handling and fixed scoring batches can also change Random/Pop
results slightly. Compare each reranker with its own baseline in `test.csv`.

## Files and commands

```bash
python -m task3.evaluate
python -m task3.plot_results
python -m unittest discover -s tests
```

- `rerankers.py`: candidate selection, profiles, and the four rerankers.
- `prepare.py`: checkpoint checks, cached scores, and Task 3 hybrid fitting.
- `metrics.py` and `evaluate.py`: metrics, validation selection, and test evaluation.
- `results/task3/`: validation trials, selected settings, test comparisons, group
  summaries, protocol, plots, and a short discussion.
- `artifacts/task3/`: local scores, candidates, final rankings, and per-user metrics.

The methods follow the course's Diversity (slides 9–11), Calibration (12–15),
and Fairness (38–43, 48) lectures. We use the lecture's greedy list-diversity
objective with mean relevance so its scale stays stable as the list grows.
Tasks 3.3 and 3.4 are separate follow-up experiments.
