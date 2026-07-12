#**系統框架**
'''
系統框架
your_project/
│
├── main.py
├── api_server.py
├── requirements.txt
├── README.md
│
├── static/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── data/
│   │
│   ├── crsp/
│   │   ├── fund_level/
│   │   │   ├── balanced_before2010.csv
│   │   │   └── balanced_after2010.csv
│   │   │
│   │   ├── holdings_raw/
│   │   │   ├── stock berfore 2010_new___.csv
│   │   │   ├── stock between 2010_2014_new___.csv
│   │   │   ├── stock between 2015_2019_new___.csv
│   │   │   └── stock between 2020_2026_new___.csv
│   │   │
│   │   └── sql/
│   │       ├── crsp_before2010.sql
│   │       ├── crsp_after2010.sql
│   │       ├── stock before 2010_new__.sql
│   │       ├── stock between 2010_2014_new__.sql
│   │       ├── stock between 2015_2019_new__.sql
│   │       └── stock between 2020_2026_new__.sql
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
│   ├── prediction(*)/
│   │   ├── manager_action_ground_truth_table.csv
│   │   ├── part6_prediction_dataset.csv
│   │   ├── part6_prediction_results.csv
│   │   ├── part6_feature_importance.csv
│   │   └── part6_decile_backtest.csv
│   │
│   ├── external_docs(*)/
│   │   ├── fund_reports/
│   │   ├── market_news/
│   │   ├── fomc/
│   │   └── manager_commentary/
│   │
│   └── rag_chunks(*)/
│       ├── chunks.jsonl
│       ├── embeddings.parquet
│       └── metadata.csv
│
├── scripts/
│   │
│   ├── preprocessing/
│   │   ├── extract_company_stocks_for_beta copy.py
│   │   ├── calculate_yearly_trailing_beta_for_part5_with_sector.py
│   │   └── preprocess_part5_excluded_two_groups.py
│   │
│   ├── modeling(*)/
│   │   ├── build_manager_action_ground_truth.py
│   │   ├── train_action_effectiveness_model.py
│   │   ├── predict_selected_visual_state.py
│   │   ├── compute_shap_explanations.py
│   │   └── build_decile_backtest.py
│   │
│   └── rag(*)/
│       ├── build_rag_chunks.py
│       ├── embed_rag_chunks.py
│       └── retrieve_context.py
│
├── backend(*)/
│   ├── __init__.py
│   ├── feature_builder.py
│   ├── prediction_service.py
│   ├── shap_service.py
│   ├── rag_service.py
│   └── llm_service.py
│
├── models(*)/
│   ├── lightgbm_action_model.pkl
│   ├── feature_columns.json
│   └── model_metadata.json
│
├── outputs(*)/
│   ├── backend_payloads/
│   ├── prediction_runs/
│   ├── shap_outputs/
│   ├── llm_explanations/
│   └── figures/
│
└── notebooks(*)/
    ├── data_check_part5a.ipynb
    ├── data_check_part5b.ipynb
    └── model_experiments.ipynb
'''