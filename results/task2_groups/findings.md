# Task 2.5: Group performance

Groups use training interactions only. All models rank the same unseen full catalog; the CSVs include all five accuracy measures, novelty, diversity, popularity and coverage. Paired bootstrap intervals compare aligned users against Pop and the validation-selected reference. Item metrics retain original top-K positions and report relevance and exposure separately.

- activity_low (477 users): EASE has the highest observed MRR (0.3599); Random has the highest novelty (11.92 bits). Test winners are descriptive, not selection rules.
- activity_high (466 users): TunedWeightedHybrid has the highest observed MRR (0.7169); Random has the highest novelty (12.21 bits). Test winners are descriptive, not selection rules.
- head_affinity_low (473 users): TunedWeightedHybrid has the highest observed MRR (0.6258); Random has the highest novelty (12.13 bits). Test winners are descriptive, not selection rules.
- head_affinity_high (470 users): EASE has the highest observed MRR (0.4438); Random has the highest novelty (11.99 bits). Test winners are descriptive, not selection rules.

For WeightedHybrid, low/high-activity MRR is 0.3536/0.7140, but recall is 0.3505/0.2136. The reversal is consistent with more relevant opportunities but larger recall denominators for active users; MRR alone would conceal their lower fraction of recovered relevant items.

WeightedHybrid assigns 98.91% of discounted exposure to the head (20.0% of items). Head/tail micro recall is 0.3397/0.0097. The collaborative ensemble therefore retains a strong popularity concentration: combining several interaction models does not by itself recover the tail. Tail slot precision can still be high because its denominator contains very few recommendations; inspect counts before interpreting that ratio as broad tail success.

# Task 2.6: Group-aware hybrid

Hypothesis: activity and preference for head items identify users who benefit from different components. A four-cell switch selects complete model rankings using validation MRR, with a global-model fallback for cells with fewer than 30 users or gains at most 0.005. It includes single-model endpoints so a lack of complementarity need not force a hybrid.

Validation-selected global reference: **WeightedHybrid**. Cell choices: `{0: 'WeightedHybrid', 1: 'WeightedHybrid', 2: 'WeightedHybrid', 3: 'ItemKNN'}`.

Cross-fitted gate development MRR: 0.4234; reference development MRR: 0.4379. The global reference uses all gate development users, so this is not a fully nested comparison.

| Test metric | Reference | Group-aware hybrid | Change |
|---|---:|---:|---:|
| MRR@10 | 0.5317 | 0.5201 | -0.0116 |
| NDCG@10 | 0.3334 | 0.3268 | -0.0066 |
| Novelty@10 | 8.5105 | 8.5303 | +0.0197 |
| IntraListDiversity@10 | 0.7256 | 0.7276 | +0.0020 |
| Coverage@10 | 0.2247 | 0.2265 | +0.0018 |

Superiority requires consistent accuracy gains without unacceptable losses in other dimensions; a positive point estimate alone does not establish it. If the gate collapses to one endpoint, these groups do not support the proposed specialization. Inspect paired_deltas.csv for uncertainty.

Mechanism checks: compare activity-group recall with MRR because users with larger held-out histories have more chances for a hit but a larger recall denominator. Compare head/tail recall with exposure and catalog share to distinguish accurate retrieval from simply recommending popular items. Higher novelty is not evidence of relevance; Random is the control for that distinction. Observed group differences motivate hypotheses, not causal explanations.

Limits: one seed, small or empty cold-item strata, reused base-model validation, unequal group sizes, and implicit relevance from observed ratings. No demographic fairness claim is supported. The gate cross-fit addresses gate-fitting optimism only. Repeat frozen choices on fresh splits before claiming a novel superior hybrid.

In this run the gate loses 0.0116 test MRR while slightly changing beyond-accuracy values. Its cross-fit result also fails to support adopting the rule. Retain the validation-selected reference rather than promoting this candidate. The next testable hypothesis is a tail-focused mixture with genuinely different content evidence and a validation accuracy constraint; activity/preference switching among largely collaborative models did not resolve the observed tail failure. This is future work, not a demonstrated superior model.
