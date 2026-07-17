# 人機協作平衡型基金研究系統：研究定位、Style 定義與 Part1–Part7 完整執行手冊

## 1. 文件目的

本文件說明本專案從原始資料、前端視覺篩選、經理人行為資料、Part6 機器學習與 TreeSHAP，到 Part7 evidence-grounded LLM critic 的完整流程。它同時區分三個不同概念：

1. `manager_style_group`：經理人的長期投資風格類別。
2. `rolling_style_deviation_score`：某次配置相對該經理人過往行為的偏離程度。
3. `SHAP cluster`：模型在該批事件上所使用的相似決策邏輯，不是經理人風格。

目前研究範圍是**平衡型基金**，核心問題不是讓 AI 取代基金經理人，而是研究專家行為資料是否能改善 AI 決策、並讓人類可檢查 AI 在不同市場狀態下的判斷依據與失效風險。

---

## 2. Introduction：研究問題與系統定位

傳統量化模型通常只把基金報酬、總體市場與基金屬性當成預測變數，卻未充分利用基金經理人已實際揭露的股、債、現金配置與持股調整。另一方面，單純使用專家平均配置，也會忽略經理人能力、風格差異、市場環境與資料揭露延遲。

本研究因此建構一套以平衡型基金為主體的人機協作決策架構：先由 Part1–Part5 讓研究者以視覺方式建立可追蹤的基金、經理人與持股事件集合，再由 Part6 使用事件前資訊建立多天期 AI 診斷、TreeSHAP 解釋及專家加權股債配置，最後由 Part7 的 LLM 擔任 evidence-grounded critic，主動尋找支持證據與反證，而不是直接預測報酬。

研究單位是：

> 某位基金經理人在某一平衡型基金、某個 `report_date` 所揭露的配置與持股行為事件。

研究的時間規則是：

- 特徵原則上截止於事件日以前或事件當期可觀察資料。
- Style baseline 不使用當期事件建立自身基準。
- 結果變數由事件後 3M、6M、9M、12M 的基金相對 S&P 500 超額報酬建立。
- 持股揭露不必然等同實際交易日，因此論文與口試必須說明 disclosure lag。

---

## 3. 這套系統的使用者是誰

### 3.1 主要使用者

| 使用者 | 使用目的 | 系統提供的價值 |
|---|---|---|
| 資產配置研究員 | 分析經理人股債配置是否具前瞻資訊 | Part5 真實配置、Part6 多天期診斷與專家加權配置 |
| 平衡型基金經理人 | 比較自身配置與歷史風格、其他同風格專家及 AI | Style 分組、style deviation、SHAP、反證提示 |
| 投資委員會／風險委員會 | 審查某項 AI 建議是否有足夠依據 | Part6 可解釋結果、Part7 evidence audit trail |
| 學術研究者 | 檢驗 Expert Data 是否提升 AI 決策品質 | 可重建的事件資料、時間切分、多天期標籤與比較基準 |

### 3.2 次要使用者

- 模型風險管理、內稽與法遵人員：檢查資料來源、模型版本、SHAP 限制與引用證據。
- 基金評估或顧問機構：比較經理人同風格相對績效與配置一致性。
- 指導教授及口試委員：重現資料建構、模型訓練、前端事件選取與輸出稽核。

### 3.3 系統不應被定位成什麼

- 不是自動下單系統。
- 不是保證報酬的選股或擇時工具。
- Part6 prediction 不是因果效果估計。
- SHAP 不是經理人的心理或意圖證明。
- Part7 LLM 不是另一個報酬預測模型。

---

## 4. 本研究的預期貢獻

### 4.1 資料貢獻

將 CRSP 基金月資料、平衡型基金持股揭露、S&P 500、10 年期利率、個股 Beta、非個股型曝險與經理人資料，整理成以 `manager × fund × report_date` 為單位的行為事件資料。

### 4.2 方法貢獻

- 將專家實際行為而不只是專家意見納入模型。
- 將經理人能力放在同風格群組內比較，降低不同投資風格直接互比造成的偏差。
- 同時分析 3M、6M、9M、12M，避免只用單一 12M 結果掩蓋期限差異。
- 使用方向分類器加正、負幅度模型，分開處理「方向」與「幅度」。
- 用 TreeSHAP 解釋模型，再以 SHAP 向量聚類觀察決策邏輯異質性。
- 以 Part7 critic 主動找反證、structural break 與資料限制。

### 4.3 系統貢獻

Part1–Part7 不是七個互不相干的圖，而是一條保留使用者選取狀態的分析鏈。Part6 收到 Part1–Part5 的完整狀態 JSON；Part7 又以這些狀態、模型輸出與外部文件建立 evidence IDs，因此結果可以追溯到「使用者選了什麼、模型用了什麼、LLM 引用了什麼」。

### 4.4 目前尚未被證明的主張

這套系統已建立研究與診斷架構，但不能只因系統較複雜就宣稱優於簡單模型。目前 `models/action_effectiveness/v002/model_metadata.json` 的時間後段測試結果顯示部分天期 AUC 仍偏弱，尤其 6M、9M、12M 尚不足以支持生產級預測能力。論文應將「Expert Data 是否提升 AI」寫成待檢驗假說，並加入簡單 baseline、walk-forward、交易成本及統計顯著性比較。

---

## 5. 為何不能只用簡單方法

### 5.1 簡單平均經理人配置

問題：每位經理人影響力相同，忽略能力、樣本可靠度、風格與市場狀態。大型基金或同一經理人管理多檔基金也可能被重複計票。

本系統：每個 report date 先將經理人收斂為一票，再以同風格相對績效形成 expert weight，並另產生 AI-only、Expert-only、Human–AI 與 equal-weight 四組比較。

### 5.2 只看歷史 Sharpe Ratio

問題：Sharpe 無法表示經理人當期做了什麼，也無法分辨股債配置、產業輪動或風格偏離。

本系統：同時保留績效風險、配置、11 產業曝險、配置變化、跨資產防守與 style deviation。

### 5.3 單一線性迴歸

問題：可能無法表達利率、市場 regime、配置調整與風格之間的非線性交互作用。

本系統：樹模型負責非線性方向分類，Ridge 負責條件幅度，並保留可比較的簡單模型作為必要 baseline，而不是完全排除簡單方法。

### 5.4 只預測未來 12M

問題：配置行為可能短期錯、長期對，或反之；單一天期會把期限結構壓成一個標籤。

本系統：同一事件同時建立 3M、6M、9M、12M 結果。

### 5.5 只看模型機率

問題：無法知道模型依賴哪一層特徵，也無法辨識模型是否在 regime shift 時沿用失效關係。

本系統：TreeSHAP、SHAP cluster、原始值逆向檢查及 Part7 critic 共同提供稽核。

### 5.6 直接請 LLM 預測

問題：LLM 可能用不明時間點的知識、混合事後資訊、產生無法查證的理由或過度自信。

本系統：LLM 只擔任 critic；Part6 prediction 不准被 LLM 偷偷重算，所有本地證據使用 `E###`，外部證據需有來源，並強制輸出反證與限制。

### 5.7 複雜系統仍需簡單 baseline

「不只用簡單方法」不等於「不測簡單方法」。完整實證至少應比較：

- S&P 500 或固定 60/40。
- 經理人配置等權平均。
- 僅歷史績效加權。
- Logistic／linear baseline。
- 不含 Expert Data 的 AI。
- 含 Expert Data 的 AI。
- Expert-only、AI-only、Human–AI。

只有在相同 point-in-time 資料、相同時間切分、成本與風險條件下顯著改善，才能支持系統的增額貢獻。

---

## 6. 你的 Manager Style 正式定義

Style 最初由根目錄 `build_manager_action_groundtruth_complete.py` 的 `build_manager_styles()` 建立。它先在經理人層級彙整 3 年資訊，再將指標轉為橫斷面 percentile score。

### 6.1 三個核心 Style scores

| Score | 組成 | 高分含義 |
|---|---|---|
| `manager_defensive_score` | max drawdown score、低波動 score、低費用 score 的平均 | 下檔控制、低波動、低費用較佳 |
| `manager_flow_score` | 平均淨流量 score、平均 MTNA score 的平均 | 資金流與基金規模支持較強 |
| `manager_growth_tilt_score` | 年化報酬 score、平均超額報酬 score、Sharpe score 的平均 | 報酬與風險調整績效較偏成長／進取 |

其中高報酬、超額報酬、Sharpe、beat rate、較佳 drawdown、flow、tenure、MTNA 使用正向 percentile；波動與費用使用反向 percentile。

可靠度另定義為：

```text
manager_reliability_score
= min(1, log(1 + manager_obs_count) / log(1 + max_manager_obs_count))
```

### 6.2 分組規則與順序

規則依下列順序判斷；先符合者先分類：

| Style | 規則 | Part6 顏色 |
|---|---|---|
| Defensive / risk-control style | defensive ≥ 0.66 且 growth < 0.66 | 藍 `#2f6b9a` |
| High-return / high-flow style | growth ≥ 0.66 且 flow ≥ 0.55 | 紅 `#c94c4c` |
| Equity-tilted / growth style | growth ≥ 0.66 | 橘 `#e28a2b` |
| Flow-supported core style | flow ≥ 0.66 | 綠 `#249b8a` |
| Balanced core style | 以上皆未符合 | 紫 `#7655a6` |
| Unknown style | 找不到可連結經理人或 style | 灰 `#7b8790` |

這是研究者定義的 rule-based expert taxonomy，不是 KMeans 自動分群。門檻 0.66／0.55 應在 robustness test 中測試其他門檻或資料驅動分群結果。

### 6.3 Style deviation 的不同意義

`rolling_style_deviation_score` 衡量單一事件在以下曝險上，相對該經理人事件日前 36 個月歷史基準的平均絕對標準化偏離：

- 股票配置、portfolio beta、科技曝險。
- bond/money 曝險、indirect equity 曝險。
- 11 個 GICS 產業曝險。

目前程式對每個 `report_date = T` 僅使用 `[T-36 months, T)` 內同一 manager 的事件。所有 T 當日報告一律一起排除，因此同日另一檔基金也不會進入 baseline。rolling mean 與 rolling standard deviation 都只使用這個事件前視窗；當期值與 rolling mean 的絕對差除以 rolling standard deviation，再跨特徵平均。

系統另外保留以下 Evidence 3 指標：

- `delta_stock`、`delta_beta`、`delta_technology`。
- 11 個 `delta_sector_*` 與其 L1 rotation 摘要 `delta_sector_exposure`。
- `delta_bond_money`、`delta_indirect_equity`。
- `nonstock_total_exposure = bond_money_exposure + indirect_equity_exposure`。
- `delta_nonstock_total_exposure`。
- `rolling_sector_deviation_score`：只彙整 11 個產業曝險偏離。
- `rolling_cross_asset_deviation_score`：彙整股票、beta、科技、bond/money、indirect equity 與 nonstock 曝險偏離。
- `rolling_action_deviation_score`：彙整當期配置 deltas 相對過去 36 個月 action baseline 的偏離。

注意：程式碼已改成嚴格 36 個月，但既有 CSV 與 model artifacts 不會自動改寫。必須重新執行資料建構與模型訓練後，新定義才會進入 Part6/Part7 實際結果。

### 6.4 SHAP cluster 不是 Manager Style

Part6 將同一天期的 TreeSHAP 向量標準化後用 KMeans 分群，PCA 只負責二維顯示。cluster 名稱由主導 SHAP feature 規則命名，例如 Technology Allocation、High-Rotation、Cross-Asset Defensive、Style-Drift、Flow-Sensitive 或 Mixed Allocation Logic。

修改後的圖表意義是：

- 點的位置：SHAP 決策邏輯在 PCA 空間的位置。
- 點的顏色：`manager_style_group`。
- 點的大小：該 SHAP cluster 中模型預測為 `large_win` 的占比。
- 點擊事件：仍選取 SHAP cluster，更新下方 fidelity 圖。

因此可觀察「不同經理人 style 是否落在相同／不同 AI decision logic」，但不能因兩點靠近就宣稱兩位經理人的真實策略完全相同。

---

## 7. Part1–Part7 使用者執行流程

### Part1：基金報酬相對 S&P 500

1. 載入 before/after 2010 的平衡型基金月資料及 S&P 500。
2. 建立 3 年 trailing 基金與 benchmark 指標。
3. 使用散點圖框選 A、B 樣本。
4. 按下套用後，基金 membership 傳入後續 Parts。

輸出：選定基金、A/B 樣本、點位與統計摘要。

### Part2：基金層級因子視覺篩選

1. 在基金層級因子圖建立一個或多個範圍。
2. 選擇 AND／OR 邏輯。
3. 套用後產生符合區域的基金與月資料。

輸出：factor regions、篩選後基金／月資料表。這是可解釋的 human-in-the-loop cohort definition，不是自動模型挑樣本。

### Part3：經理人選取

1. 從 Part2 的基金集合連結 manager name。
2. 檢視經理人資料與候選清單。
3. 選定要追蹤的經理人。

輸出：selected managers 與 manager details。

### Part4：經理人能力與 Style Drift

1. 比較報酬、超額報酬、Sharpe、drawdown、flow、費用、規模等 indicators。
2. 使用 radar 或群組圖比較經理人。
3. 觀察年度或事件期 style drift。

輸出：`manager_records_raw`、scores、style drift rows。Part6 的 expert performance score 會使用這些資料。

### Part5：真實配置與持股動作

1. 依 Part4 經理人限制其平衡型基金報告。
2. 連結 10 年期殖利率、股票／債券／現金配置。
3. 對公司股票連結年度 trailing beta 與 sector。
4. 將債券、貨幣市場、基金型曝險放在 non-individual exposure 分析。
5. 比較相鄰 report date，建立新增、加碼、減碼及權重差。
6. 選取 report rows，成為 Part6 anchor events。

輸出：selected reports、全部持股明細、stock action rows、allocation state。

### Part6：多天期 AI 與專家協作診斷

1. 使用者按 **Run Backend Analysis**。
2. `static/app.js` 將 Part1–Part5 狀態與 Part6 天期／日期送到 `/api/ml/analyze-visual-state`。
3. API 保存完整 payload 及各 Part latest JSON。
4. `feature_builder.py` 將 Part5 report key 優先對齊 ML event；若失敗才依 manager/fund fallback。
5. `prediction_service.py` 對 3M、6M、9M、12M 分別預測正向機率、條件幅度與五分類。
6. `shap_service.py` 產生 TreeSHAP、global importance、SHAP KMeans/PCA 與 fidelity raw values。
7. `expert_collaboration_service.py` 計算同 style 相對績效權重、AI confidence 與 Human–AI 配置。
8. SHAP event picker 可同時選擇最多 8 個 manager-report events；比較圖以跨所選事件總絕對 SHAP 最高的 14 個特徵對齊顯示。
9. Fidelity 箱型圖以全部回傳事件的 feature mean/std 標準化，避免殖利率、報酬率與 manager score 因量尺不同而被壓縮；hover 仍保留 raw value。
10. 前端列出 prediction、SHAP、股債建議、經理人貢獻與所有實際新增／加碼／減碼持股。

### Part7：Evidence-Grounded LLM Critic

1. 選擇一至八個 Part6 events 與天期；預設跟隨 Part6 SHAP 多選結果。
2. backend 從 `visual_state_latest.json` 與 `backend_ml_latest.json` 建立 evidence chunks。
3. 檢索 `data/part7_knowledge/` 的同期文件。
4. 有 API Key 且安裝 OpenAI SDK 時，可使用 Responses API web search 補充新聞、FOMC、基金報告及 commentary。
5. LLM 只能批判 Part6 解釋，不直接重做 prediction；多事件時必須先逐事件判斷，再比較共同與相反的 SHAP/action evidence，不可先平均經理人。
6. Structured Output 必須列出支持證據、反證、structural breaks、資料限制、過度解釋風險、專家問題、verdict 與 citations。
7. 沒有 Key 時回傳 Preview，供檢查 evidence IDs 與 prompt。

---

## 8. 程式啟動後的資料流

```text
Browser
  static/index.html
  static/app.js
      │
      ├─ GET CSV aliases / static files
      │
      ├─ POST /api/ml/analyze-visual-state
      │       │
      │       ├─ save Part1–Part5 JSON
      │       ├─ backend/feature_builder.py
      │       ├─ backend/prediction_service.py
      │       ├─ backend/shap_service.py
      │       └─ backend/expert_collaboration_service.py
      │
      └─ POST /api/part7/critic
              │
              ├─ saved visual + Part6 evidence
              ├─ data/part7_knowledge local RAG
              └─ backend/part7_rag_service.py → OpenAI Responses API（可選）

main.py → Uvicorn → api_server.py → FastAPI routes
```

---

## 9. 有作用的主要檔案

### 9.1 Runtime 與 API

| 檔案 | 作用 |
|---|---|
| `main.py` | 以 Uvicorn 在 `127.0.0.1:8000` 啟動 `api_server:app`，開發模式使用 reload |
| `api_server.py` | FastAPI、CSV aliases、靜態檔服務、Part6 與 Part7 endpoints、JSON audit 保存 |
| `requirements.txt` | Python runtime、ML、SHAP、XGBoost、OpenAI SDK 依賴 |
| `.env.example` | Part7 環境變數範例；程式不會自動讀取此檔，仍需在 shell 或部署環境設定 |

### 9.2 Frontend

| 檔案 | 作用 |
|---|---|
| `static/index.html` | Part1–Part7 DOM、控制項、圖表及表格容器 |
| `static/app.js` | CSV 載入、所有互動篩選、指標計算、Part5 actions、backend payload、Part6/7 render |
| `static/style.css` | 全站與各 Part 視覺樣式 |

### 9.3 Part6 backend

| 檔案 | 作用 |
|---|---|
| `backend/feature_builder.py` | 從 Part1–Part5 state 對齊 ML event、日期範圍與 metadata |
| `backend/prediction_service.py` | 載入四個天期 bundle，產生方向機率、幅度與五分類 |
| `backend/shap_service.py` | TreeSHAP、SHAP KMeans、PCA display、cluster fidelity；現在也傳遞 manager style |
| `backend/expert_collaboration_service.py` | 同風格相對 expert weight、AI weight、Human–AI 股債現金配置 |
| `backend/shap.py` | 舊有／輔助 SHAP 模組；目前主要 Part6 route 使用 `shap_service.py` |

### 9.4 Part7 backend

| 檔案 | 作用 |
|---|---|
| `backend/part7_rag_service.py` | visual evidence、local RAG、prompt、Responses API、Structured Output |
| `prompts/part7_evidence_grounded_critic.md` | critic 的不可違反規則與輸出要求 |
| `data/part7_knowledge/README.md` | 本地 macro/news/FOMC/fund/commentary 文件格式 |
| `README_PART7.md` | Part7 簡要安裝與執行說明 |

### 9.5 Offline 資料與模型建構

| 檔案 | 作用 |
|---|---|
| `build_manager_action_groundtruth_complete.py` | 從基金與持股原始資料建立 base ground truth、manager style、market regime、actions |
| `scripts/modeling/build_manager_action_groundtruth_complete.py` | 將 base ground truth 擴充為 leakage-aware 3Y features 與 3/6/9/12M labels |
| `scripts/modeling/train_action_effectiveness_model.py` | 訓練四個天期的 tree direction classifier、positive/negative Ridge 與 portable fallback |
| `scripts/processing/extract_company_stocks_for_beta copy.py` | 從持股中建立公司股票 beta universe（需要重建 Part5 beta 時使用） |
| `scripts/processing/calculate_yearly_trailing_beta_for_part5_with_sector.py` | 下載／快取股價並計算年度 1Y/3Y/5Y beta 與 sector；內含舊絕對路徑，跨機器執行前需修改 |
| `scripts/processing/preprocess_part5_excluded_two_groups.py` | 將非公司持股拆為債券／信用／貨幣與 equity-fund-like；內含舊絕對路徑，執行前需修改 |

### 9.6 模型 artifacts

`models/action_effectiveness/v002/` 內每個 3M、6M、9M、12M 天期包含：

- `dual_stage_model_{h}m.pkl`：主要 bundle。
- `dual_stage_model_{h}m_sklearn.pkl`：未安裝 XGBoost 時的 portable fallback。
- `direction_model_{h}m.pkl`。
- `positive_ridge_{h}m.pkl`、`negative_ridge_{h}m.pkl`。
- `feature_columns.json`、`model_metadata.json`。

---

## 10. 從零開始：完整重建與執行

以下命令都在專案根目錄執行。

### Step 0：建立環境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若 PowerShell 禁止啟動 script，可在目前 session 使用：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 1：放置必要資料

至少確認：

```text
data/crsp/fund_level/balanced_before2010.csv
data/crsp/fund_level/balanced_after2010.csv
data/crsp/holdings_raw/stock berfore 2010_new___.csv
data/crsp/holdings_raw/stock between 2010_2014_new___.csv
data/crsp/holdings_raw/stock between 2015_2019_new___.csv
data/crsp/holdings_raw/stock between 2020_2026_new___.csv
data/market/sp500_monthly_returns_1871_2026.csv
data/market/FRB_H15.csv
data/part5_equity_beta/part5_yearly_trailing_stock_beta.csv
data/part5_non_individual_holdings/*.csv
```

### Step 2：建立 base manager-action ground truth

必須明確指定 output 到 `data/derived`，避免根目錄 builder 的舊 default 寫到 `data/outputs`：

```powershell
python .\build_manager_action_groundtruth_complete.py `
  --data-root .\data `
  --output-dir .\data\derived\manager_action_groundtruth
```

此步建立 manager style、配置、持股 actions、market regime 與 base outcomes。

### Step 3：建立多天期 leakage-aware dataset

```powershell
python .\scripts\modeling\build_manager_action_groundtruth_complete.py --data-root .\data
```

主要輸出：

```text
data/derived/manager_action_groundtruth/manager_action_ground_truth.csv
data/derived/manager_action_groundtruth/manager_action_ground_truth_trailing3y_multi_horizon.csv
data/derived/prediction/part6_prediction_dataset_trailing3y_multi_horizon.csv
data/derived/prediction/part6_prediction_dataset.csv
```

檢查 `manager_action_ground_truth_audit.json` 中：

- `leakage_check_passed_counts`。
- 四個 horizon 的 label counts。
- rows 與缺失值。
- 正負類別是否嚴重不平衡。

### Step 4：訓練 Part6 模型

```powershell
python .\scripts\modeling\train_action_effectiveness_model.py --data-root .\data
```

若已安裝 XGBoost，產生主要 XGBoost bundle；程式也會建立 sklearn fallback。完成後查看：

```text
models/action_effectiveness/v002/model_metadata.json
models/action_effectiveness/v002/feature_columns.json
```

### Step 5：準備 Part7（可選）

沒有 API Key 也能使用 Preview。若要 Live：

```powershell
$env:OPENAI_API_KEY="你的 API key"
$env:OPENAI_MODEL="gpt-5.6"
```

不要把真實 key 寫進 `.env.example`、JavaScript、Git 或 screenshot。

將 point-in-time 文件放入：

```text
data/part7_knowledge/
```

格式參照 `data/part7_knowledge/README.md`。

### Step 6：啟動 API 與網頁

```powershell
python .\main.py
```

開啟：

```text
http://127.0.0.1:8000
```

先檢查：

```text
http://127.0.0.1:8000/api/health
http://127.0.0.1:8000/api/files
http://127.0.0.1:8000/api/part7/status
```

### Step 7：依序操作 Part1–Part7

1. 載入預設 CSV。
2. Part1 框選並套用 A/B。
3. Part2 設定因子範圍與 AND/OR，套用。
4. Part3 選經理人。
5. Part4 檢查 indicator、radar、群組與 style drift。
6. Part5 選 manager mode、基金報告與持股事件。
7. Part6 選 3M/6M/9M/12M，按 **Run Backend Analysis**。
8. 檢查 prediction、SHAP、style-colored map、fidelity、expert allocation 與實際持股動作。
9. Part7 選事件，按 **Run Evidence Critic**。
10. 保存所選條件、backend JSON、圖表及模型版本，作為論文可重現附件。

---

## 11. API endpoints 與輸出

| Endpoint | 方法 | 作用 |
|---|---|---|
| `/api/health` | GET | API、static、data、backend import 健康狀態 |
| `/api/files` | GET | 顯示 CSV aliases 與 Part6 artifacts 是否存在 |
| `/api/ml/analyze-visual-state` | POST | 保存 Part1–Part5 state、執行 Part6 prediction/SHAP/expert collaboration |
| `/api/part7/status` | GET | Part7 model、SDK、API Key、local knowledge 狀態 |
| `/api/part7/critic` | POST | 執行 Part7 Preview 或 Live critic |

重要輸出：

```text
outputs/backend_payloads/visual_state_latest.json
outputs/backend_payloads/part1_latest.json ... part5_latest.json
outputs/backend_payloads/backend_ml_latest.json
outputs/part7/part7_critic_latest.json
```

timestamped files 是 audit trail；`latest` 是前端與 Part7 使用的最近一次狀態。

---

## 12. 論文與口試必須主動揭露的限制

1. Holdings report date 不必然是實際交易日，需處理 availability/disclosure lag。
2. `future_excess_return` 是歷史 action-outcome association，不是經理人行為的因果效果。
3. S&P 500 未必是所有平衡型基金最適 benchmark；應加入平衡型 benchmark 或自訂股債 benchmark。
4. 目前 expert outcome 是 proxy，不是完整可交易 backtest；仍缺債券 total-return index、成本、稅、turnover 與滑價。
5. Style taxonomy 門檻由研究者設定，需 robustness tests。
6. Style baseline 已定義為嚴格 event-time trailing 36M；實證時仍需確認資料已重建、同日事件確實排除且每筆事件有足夠 `style_obs_count`。
7. TreeSHAP 解釋模型，不解釋真實世界因果。
8. SHAP cluster 名稱是 heuristic，PCA 只用於顯示。
9. 目前模型時間後段 AUC 顯示多個天期表現仍弱；不可挑選最好天期後宣稱全面有效。
10. Part7 的網路資料可能有來源缺漏、發布時間錯置或 prompt injection；必須保留引用並由人類複核。

---

## 13. 建議的正式實證順序

1. 鎖定 point-in-time dataset 與資料版本。
2. 使用 expanding-window 或 rolling-window walk-forward validation。
3. 先報告簡單 baseline。
4. 做 feature ablation：拿掉 manager style、actions、macro、holdings 等層。
5. 比較 Expert-only、AI-only、Human–AI、equal-weight。
6. 分 regime、style、horizon 報告結果。
7. 加入 turnover、成本、drawdown 與風險調整績效。
8. 使用 bootstrap／Diebold–Mariano 或合適的 panel inference 檢驗差異。
9. 把 Part7 評估為 critic quality，例如 citation correctness、counterevidence recall、human usefulness；不要用它的文字流暢度代替投資績效。

只有完成上述比較後，才可將「系統設計貢獻」進一步提升為「實證績效貢獻」。
