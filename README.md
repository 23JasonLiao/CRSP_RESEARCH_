# 專案架構 (Project Structure)

```text
your_project/
│
├── main.py
├── api_server.py
├── requirements.txt
│
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── data/
│   ├── crsp/
│   │   ├── fund_level/
│   │   │   ├── balanced_before2010.csv
│   │   │   └── balanced_after2010.csv
│   │   │
│   │   └── holdings_raw/
│   │       ├── stock berfore 2010_new___.csv
│   │       ├── stock between 2010_2014_new___.csv
│   │       ├── stock between 2015_2019_new___.csv
│   │       └── stock between 2020_2026_new___.csv
│   │
│   ├── market/
│   │   ├── sp500_monthly_returns_1871_2026.csv
│   │   └── FRB_H15.csv
│   │
│   ├── part5_equity_beta/
│   │   ├── part5_unique_company_stocks_for_yahoo_beta.csv
│   │   ├── part5_yearly_trailing_stock_beta.csv
│   │   ├── part5_yfinance_failed_tickers.csv
│   │   ├── part5_yfinance_monthly_close_cache.csv
│   │   └── part5_yfinance_sector_cache.csv
│   │
│   ├── part5_non_individual_holdings/
│   │   ├── part5_excluded_non_company_holdings_audit.csv
│   │   ├── part5_excluded_two_group_enriched.csv
│   │   ├── part5_excluded_two_group_summary.csv
│   │   ├── part5_excluded_two_group_top_items.csv
│   │   ├── part5_excluded_two_group_active_year_panel.csv
│   │   └── part5_excluded_individual_stock_like_removed_audit.csv
│   │
│   └── derived/
│       ├── manager_action_groundtruth/
│       │   ├── manager_action_ground_truth.csv
│       │   ├── manager_action_ground_truth_trailing3y_future12m.csv
│       │   ├── manager_action_ground_truth_trailing5y_future12m.csv
│       │   ├── manager_action_ground_truth_audit.json
│       │   ├── manager_action_ground_truth_schema.json
│       │   └── manager_action_ground_truth_data_dictionary.md
│       │
│       └── prediction/
│           ├── part6_prediction_dataset.csv
│           ├── part6_prediction_dataset_trailing3y_future12m.csv
│           └── part6_prediction_dataset_trailing5y_future12m.csv
│
├── scripts/
│   ├── preprocessing/
│   │   ├── extract_company_stocks_for_beta.py
│   │   ├── calculate_yearly_trailing_beta_for_part5_with_sector.py
│   │   └── preprocess_part5_excluded_two_groups.py
│   │
│   └── modeling/
│       ├── build_manager_action_groundtruth_complete.py
│       └── train_action_effectiveness_model.py
│
├── backend/
│   ├── __init__.py
│   ├── feature_builder.py
│   ├── prediction_service.py
│   ├── shap_service.py
│   └── shap.py
│
├── models/
│   └── action_effectiveness/
│       └── v001/
│           ├── lightgbm_action_model.pkl
│           ├── xgboost_action_model.pkl
│           ├── lightgbm_action_model_trailing3y.pkl
│           ├── xgboost_action_model_trailing3y.pkl
│           ├── shap_background_sample_trailing3y.csv
│           ├── lightgbm_action_model_trailing5y.pkl
│           ├── xgboost_action_model_trailing5y.pkl
│           ├── shap_background_sample_trailing5y.csv
│           ├── feature_columns.json
│           ├── preprocessing_config.json
│           └── model_metadata.json
│
└── outputs/
    └── backend_payloads/
        ├── visual_state_latest.json
        ├── part1_latest.json
        ├── part2_latest.json
        ├── part3_latest.json
        ├── part4_latest.json
        ├── part5_latest.json
        └── backend_ml_latest.json