# manager_action_ground_truth data dictionary

Each row is a manager-action event at a holdings report date. The v2 builder expands each base event into 3Y and 5Y training-window views, both predicting future 12-month outcomes.

## Key additions

- `current_*`: current report-month fund characteristics.
- `fund_trailing_*`: generic alias for the chosen training window.
- `rolling_style_deviation_score`: deviation from the manager's own past style before report_date.
- `label_positive_excess_12m`: binary target for Part6 ML.
- `leakage_check_passed`: confirms the style window ends before report_date.