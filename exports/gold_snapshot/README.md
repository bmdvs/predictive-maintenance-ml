# Gold snapshot

Exported 2026-10-26 from the prod workspace (catalog pdm) after the final checkpoint (b) run, as of 2026-06-30 23:00 local. The pages build from these files without a workspace.

- gold/, monitoring/: one Parquet file per table written by the job or the pipeline.
- silver/: points, confirmed conditions, healthy windows; hourly_indicators from 2026-04-01 05:00 UTC (the quarter and the 90-day trends); hourly_indicators_trend_reference, the four trend indicators from 2026-01-30 to that cut for the queue's baseline bands, added 2026-10-27 from the local reference after a check that it equals this export on every overlapping row and column; the prognosis instances' velocity series; the HMC rows of controller_hourly (coolant pressure, time-weighted and pump-on).
- bronze/: the asset register, CMMS work orders and the contribution margin.
- bursts/: for each open case's vibration points, the baseline reference burst, the first burst after the first alarm and the latest before the cut (float32), with panel_bursts.parquet giving role, time, offset and the point's demodulation band.
- mlflow/: experiments and runs under /Shared/pdm (parameters, metrics, tags) and the registry (versions, tags, aliases) of pdm.ml.nbm_point_models.
- parity_check.parquet: each exported table against the local reference (1e-9 relative with a 1e-12 floor; autoencoder-derived values and the penalized Cox concordance judged at outcome level).

Not included: the hourly per-point model score tables (nbm_scores, nbm_train_scores, scoring_log, scoring_log_baseline), which stay in the workspace and the local reference.
