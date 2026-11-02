# Gold snapshot

Exported 2026-10-30 for revision 2 from the run09 record (run09_20261030T090522Z_3eced8fb), as of 2026-06-30 23:00 local, after the prod job run (registered model pdm.ml.nbm_point_models version 8, champion). The files are the local reference's current Delta versions, checked against prod table by table: equal row counts on every table and every numeric column's sum within 1e-9 relative, except the selection tables' chance shares (about 1e-4, tie order in the budget interpolation) and the shared-shape Weibull (about 1e-7, optimizer tolerance).

- gold/, monitoring/: one Parquet file per table written by the job or the pipeline, including the checkpoint (b) tables (cpb_*).
- silver/: points, confirmed conditions, healthy windows.
- bronze/: the asset register, CMMS work orders, the contribution margin and the burst index.

Not included: the hourly score and exceedance tables (nbm_scores, nbm_exceedances, ml_train_scores, ml_fold_scores, scoring_log, watch_hours), the rule-floor exceedances, rolling features and hourly indicators, which stay in the workspace and the local reference. The page build adds what the pages need (bursts, trend references, MLflow metadata) when the pages are rebuilt.

The run06 snapshot behind the published pages is kept as data/gold_snapshot_run06.
