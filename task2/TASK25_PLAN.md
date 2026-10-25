# Tasks 2.5 and 2.6: plan and repository audit

Audited main at aa3c915 after fetching and fast-forwarding on 24 October 2026.

## Existing work

- Tasks 1.1–1.2: selected individual RecBole models, systematic searches, saved
  configurations and aggregate/seed results.
- Tasks 1.3–1.5: weighted, feature-combination and switching hybrids;
  validation selection, coefficient tables and ablations.
- Tasks 2.1–2.2: independently implemented ranking/genre metrics, baseline
  comparison, cached rankings, aggregate results for 15 variants.
- Tasks 2.3–2.4: coefficient, component/feature ablation and overlap analysis;
  cross-fit hybrid selection and three-seed comparisons.
- Tasks 3.1–3.4: rerankers, placement comparisons, validation selections,
  test tables and reranker group impact figures.

These are implementation/result artifacts, not proof that a final report or
all rubric explanations are complete. The README links to missing Task 3
method/discussion files. Checkpoints are ignored and most were absent locally.
The final report and submission bundle are absent from this checkout.
The checked-in RecBole gitlink is `439c5a8` (the previous README claimed
`7aa0ec9`); the README now reflects the actual pin. This experiment retains
the checked-in dependency commit rather than changing model implementations.
The saved Task 3 manifest records `7aa0ec9`. Comparing `439c5a8..7aa0ec9`
shows only Finder metadata and eight added lines in `save_split.py`; model,
training, configuration and evaluation Python sources are identical. The
version mismatch therefore does not change the model algorithms used here.

## Gap and implementation plan

1. Compare original, unreranked models across training-defined activity and
   popularity-preference user groups, plus head/tail and cold item groups.
2. Reuse independent accuracy and beyond-accuracy metrics; report group sizes,
   exposure, coverage, appropriate relevance denominators, and paired intervals.
3. Test the hypothesis that activity and popularity preference predict useful
   component specialization with a four-cell validation-selected switch.
   Include global endpoints, minimum support and minimum gain safeguards.
4. Cross-fit gate fitting as a development diagnostic; freeze the final rule
   before test evaluation. Preserve base-validation reuse limitations explicitly.
5. Save runnable code, raw/per-group tables, provenance, a figure and concise
   report material; run metric/gate tests and inspect the generated outputs.

## Protocol choice

Reuse `task3.prepare` to recover selected checkpoints and build consistent score
matrices. Its weighted hybrids fit on one reserved half of validation users;
the other half fits the new gate. This yields an internally consistent
comparison but is a fresh refit, so numbers need not reproduce Task 2.2.
All original model settings remain frozen. No reranker is applied. Test labels
are unavailable to group construction or gate selection. Training-only groups
are observational; they do not measure demographic fairness or causal effects.

## Deliverables

`python -m task2.group_analysis` writes `results/task2_groups/` with user/item
group tables, per-user values, paired differences, gate selection, protocol,
findings, concise report sections and a comparison figure. Use `--models` for
a smaller explicit set; the default evaluates all 15 existing variants and
the new gate.
