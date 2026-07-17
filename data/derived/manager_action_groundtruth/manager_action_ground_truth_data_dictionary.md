# manager_action_ground_truth data dictionary

Each row is a manager-action event. Features end before the event and outcomes cover 3M, 6M, 9M and 12M after it.

## Key additions

- `current_*`: current report-month fund characteristics.
- `fund_trailing_*`: generic alias for the chosen training window.
- `rolling_style_deviation_score`: deviation from the manager's own past style before report_date.
- `rolling_sector_deviation_score`: 11-sector exposure deviation from the strict prior-36M manager baseline.
- `rolling_action_deviation_score`: action-delta deviation from the strict prior-36M manager baseline.
- `style_window_type`: strict event-time trailing 36M manager history, excluding every current-date event.
- `direction_label_{h}m`: -1/0/+1 direction with a +/-0.5% neutral band.
- `outcome_5class_{h}m`: large loss, small loss, neutral, small win, or large win.
- `leakage_check_passed`: confirms the style window ends before report_date.