# Ground-truth QC summary

Rows: 32,916
Columns: 53
Report date range: 2002-08-31 to 2025-11-30
Unique managers: 171
Unique funds: 554
All required columns are present.

## Main cautions
- Treat `outcome_label` as historical association, not causal proof.
- Use `data_quality_flags == ok` as the clean first-pass subset for ML.
- Rows with missing future outcomes are expected near the end of the sample because there is no complete future 12-month window.
- Allocation/exposure proxies can exceed 100% when raw CRSP holdings contain leverage, shorts, derivatives, duplicated exposure, or proxy-completed values; review range plots before modeling.

## Figures generated
- 01_outcome_label_distribution.png
- 02_action_type_counts.png
- 03_action_type_avg_future_excess.png
- 04_action_type_positive_rate.png
- 05_style_conditioned_action_heatmap.png
- 06_market_regime_execution_heatmap.png
- 07_style_deviation_bucket_future_excess.png
- 08_yearly_action_event_count.png
- 09_yearly_avg_future_excess.png
- 10_data_quality_flags.png