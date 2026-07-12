# 3Y Backend ML Runnable Patch

## Run

```bash
pip install -r requirements.txt
python main.py
```

Open: http://127.0.0.1:8000

## Check paths

Open these two URLs first:

- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/api/files

`/api/files` should show `exists: true` for the three default Part1 files:

- balanced_before2010.csv
- balanced_after2010.csv
- sp500_monthly_returns_1871_2026.csv

Part5 additionally needs the holdings files, FRB_H15.csv, Part5 beta, and Part5B CSVs.
Part6 additionally needs:

- data/derived/prediction/part6_prediction_dataset_trailing3y_future12m.csv
- models/action_effectiveness/v001/lightgbm_action_model_trailing3y.pkl
- models/action_effectiveness/v001/feature_columns.json

## Important

This version fixes these common failures:

1. `backend/` folder missing: included in this patch.
2. CSVs placed flat under the project root: `api_server.py` now accepts both nested `data/...` paths and flat root files.
3. Part6 mode reset to nonexistent `frontend`: now reset defaults to `backend`.
4. Frontend still tries to load from CDN for Plotly and PapaParse. If the page says Plotly/PapaParse is not loaded, check internet/CDN access or download local vendor JS files.
