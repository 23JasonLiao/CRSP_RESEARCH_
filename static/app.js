"use strict";

// 3Y-only backend version: frontend keeps Part1-Part5 interactions, but Part6 uses backend ML/SHAP only.

const DEFAULT_FUND_FILES = ["balanced_before2010.csv", "balanced_after2010.csv"];
const DEFAULT_SP500_FILE = "sp500_monthly_returns_1871_2026.csv";

const MS_PER_DAY = 24 * 60 * 60 * 1000;
const MS_PER_YEAR = 365.25 * MS_PER_DAY;
const RISK_FREE_RATE = 0.01;

const COLORS = {
  a: "rgba(31, 111, 178, 0.78)",
  aDark: "rgba(13, 71, 161, 0.96)",
  b: "rgba(230, 111, 71, 0.66)",
  bDark: "rgba(183, 63, 28, 0.95)",
  both: "rgba(211, 47, 47, 0.95)"
};

const HORIZONS = {
  y3: { label: "3年", title: "Trailing 3年" }
};

const HORIZON_WINDOWS = {
  y3: 36
};

const HORIZON_MONTH_COUNTS = {
  ...HORIZON_WINDOWS
};

const ROLLING_FACTOR_KEYS = ["net_flow", "mtna", "exp_ratio", "mgmt_fee", "turn_ratio", "age", "tenure"];

const LEVELS = [
  { key: "fund", label: "基金層級因子（與 Part6 ML 相關）", defaultCount: 14 }
];

const FEATURES = {
  monthly: [
    ["mret", "基金月報酬率"],
    ["sp500_ret", "S&P500月報酬率"],
    ["excess_ret", "基金超額月報酬"],
    ["abs_mret", "基金月報酬絕對值"],
    ["net_flow", "Net Flow"],
    ["flow_abs", "Flow絕對值"],
    ["mtna", "MTNA資產規模"],
    ["exp_ratio", "費用率"],
    ["tenure", "管理年資"],
    ["turn_ratio", "換手率"],
    ["year", "觀測年份"],
    ["family_avg_return", "家族平均報酬率"],
    ["family_avg_fee", "家族平均費用率"],
    ["family_avg_mtna", "家族平均規模"],
    ["family_avg_flow", "家族平均淨申購"]
  ],
  fund: [
    ["obs", "觀測月數"],
    ["mean_return", "平均月報酬率"],
    ["annual_return", "年化報酬率"],
    ["cumulative_return", "累積報酬率"],
    ["annual_volatility", "年化波動度"],
    ["sharpe", "Sharpe Ratio"],
    ["sortino", "Sortino Ratio"],
    ["max_drawdown", "Max Drawdown"],
    ["win_rate", "月勝率"],
    ["beat_sp500_rate", "打敗S&P500月比例"],
    ["annual_excess", "年化超額報酬"],
    ["tracking_error", "Tracking Error"],
    ["information_ratio", "Information Ratio"],
    ["beta", "Beta vs S&P500"],
    ["alpha", "Alpha年化近似"],
    ["upside_capture", "Upside Capture"],
    ["avg_mtna", "平均MTNA"],
    ["avg_net_flow", "平均Net Flow"],
    ["sum_net_flow", "累積Net Flow"],
    ["avg_exp_ratio", "平均費用率"],
    ["avg_tenure", "平均管理年資"],
    ["avg_turnover", "平均換手率"]
  ],
  family: [
    ["obs", "觀測點數"],
    ["fund_count", "基金數"],
    ["manager_count", "經理人數"],
    ["avg_return", "家族平均報酬率"],
    ["avg_excess", "家族平均超額報酬"],
    ["avg_fee", "家族平均費用率"],
    ["avg_mtna", "家族平均規模"],
    ["avg_flow", "家族平均淨申購"],
    ["sum_flow", "家族累積淨申購"],
    ["avg_turnover", "家族平均換手率"],
    ["avg_tenure", "家族平均管理年資"]
  ]
};

const PART2_DISPLAY_FEATURES = {
  fund: [
    ["annual_return", "年化報酬率"],
    ["annual_volatility", "年化波動度"],
    ["sharpe", "Sharpe Ratio"],
    ["max_drawdown", "Max Drawdown"],
    ["beat_sp500_rate", "打敗S&P500比例"],
    ["annual_excess", "年化超額報酬"],
    ["tracking_error", "Tracking Error"],
    ["information_ratio", "Information Ratio"],
    ["beta", "Beta vs S&P500"],
    ["avg_mtna", "平均MTNA"],
    ["avg_net_flow", "平均Net Flow"],
    ["avg_exp_ratio", "平均費用率"],
    ["avg_turnover", "平均換手率"],
    ["avg_tenure", "平均管理年資"]
  ]
};

const PART2_TABLE_LIMIT = 500;

const PART2_TABLE_BASE_COLUMNS = {
  monthly: [
    { key: "caldt", label: "日期" },
    { key: "crsp_fundno", label: "基金代號" },
    { key: "fund_name", label: "基金名稱" },
    { key: "mgmt_name", label: "基金家族" },
    { key: "mgr_name", label: "經理人" }
  ],
  fund: [
    { key: "crsp_fundno", label: "基金代號" },
    { key: "fund_name", label: "基金名稱" },
    { key: "mgmt_name", label: "基金家族" },
    { key: "mgr_name", label: "主要經理人" }
  ],
  family: [
    { key: "mgmt_name", label: "基金家族" }
  ]
};

const PART2_TABLE_FORMATS = {
  mret: "pct",
  sp500_ret: "pct",
  excess_ret: "pct",
  abs_mret: "pct",
  net_flow: "money",
  flow_abs: "money",
  mtna: "money",
  exp_ratio: "pct",
  tenure: "years",
  turn_ratio: "pct",
  year: "int",
  family_avg_return: "pct",
  family_avg_fee: "pct",
  family_avg_mtna: "money",
  family_avg_flow: "money",
  obs: "int",
  mean_return: "pct",
  annual_return: "pct",
  cumulative_return: "pct",
  annual_volatility: "pct",
  sharpe: "num",
  sortino: "num",
  max_drawdown: "pct",
  win_rate: "pct",
  beat_sp500_rate: "pct",
  annual_excess: "pct",
  tracking_error: "pct",
  information_ratio: "num",
  beta: "num",
  alpha: "pct",
  upside_capture: "pct",
  avg_mtna: "money",
  avg_net_flow: "money",
  sum_net_flow: "money",
  avg_exp_ratio: "pct",
  avg_tenure: "years",
  avg_turnover: "pct",
  fund_count: "int",
  manager_count: "int",
  avg_return: "pct",
  avg_excess: "pct",
  avg_fee: "pct",
  avg_flow: "money",
  sum_flow: "money"
};
const RADAR_METRICS = [
  { label: "年化報酬", rawKey: "annual_return", higherBetter: true, format: "pct" },
  { label: "平均超額報酬", rawKey: "avg_excess", higherBetter: true, format: "pct" },
  { label: "Sharpe Ratio", rawKey: "sharpe", higherBetter: true, format: "num" },
  { label: "贏 S&P500", rawKey: "beat_rate", higherBetter: true, format: "pct" },
  { label: "Max Drawdown", rawKey: "max_drawdown", higherBetter: true, format: "pct" },
  { label: "年化波動", rawKey: "annual_volatility", higherBetter: false, format: "pct" },
  { label: "費用率", rawKey: "avg_fee", higherBetter: false, format: "pct" },
  { label: "Flow", rawKey: "avg_flow", higherBetter: true, format: "money" },
  { label: "經理人任期", rawKey: "avg_tenure", higherBetter: true, format: "years" },
  { label: "MTNA", rawKey: "avg_mtna", higherBetter: true, format: "money" }
];

const INDICATOR_TICK_VALUES = [0, 0.25, 0.5, 0.75, 1];
const INDICATOR_TICK_TEXT = ["0", "25", "50", "75", "100"];
const INDICATOR_COLORS = ["#1f6fb2", "#e66f47", "#2d8a63", "#8054b8", "#b58b00", "#008c95", "#bf4b75", "#6b7480"];

const PART5_YIELD_FILE = "FRB_H15.csv";
const PART5_BETA_FILE = "part5_yearly_trailing_stock_beta.csv";
const PART5_EXCLUDED_SUMMARY_FILE = "part5_excluded_two_group_summary.csv";
const PART5_EXCLUDED_PANEL_FILE = "part5_excluded_two_group_active_year_panel.csv";
const PART5_EXCLUDED_ENRICHED_FILE = "part5_excluded_two_group_enriched.csv";
const PART5_EXCLUDED_TOP_ITEMS_FILE = "part5_excluded_two_group_top_items.csv";
const PART5_EXCLUDED_REMOVED_FILE = "part5_excluded_individual_stock_like_removed_audit.csv";
const PART5_STOCK_SOURCES = [
  {
    key: "before2010",
    label: "2010 年以前",
    file: "stock berfore 2010_new___.csv"
  },
  {
    key: "y2010_2014",
    label: "2010-2014",
    file: "stock between 2010_2014_new___.csv"
  },
  {
    key: "y2015_2019",
    label: "2015-2019",
    file: "stock between 2015_2019_new___.csv"
  },
  {
    key: "y2020_2026",
    label: "2020-2026",
    file: "stock between 2020_2026_new___.csv"
  }
];
const PART5_TOP_HOLDINGS_LIMIT = 15;
const PART6_TARGETS = {
  next1: { label: "下一期", months: 1 },
  next12: { label: "未來12期", months: 12 }
};

const state = {
  rows: [],
  activeRows: [],
  selectedMgmt: new Set(),
  horizon: "y3",
  selectionMode: "single",
  manualMonthlySp500: annualToMonthly(0.1),
  latestP1Box: null,
  p1BoxA: null,
  p1BoxB: null,
  p1Applied: false,
  rawA: [],
  rawB: [],
  latestP2Region: null,
  pendingP2Regions: [],
  appliedP2Regions: [],
  part2Logic: "and",
  selectedFeatures: {},
  part2TablesA: null,
  part2TablesB: null,
  part3RawA: [],
  part3RawB: [],
  latestManagers: [],
  pendingManagers: new Set(),
  radarRecords: [],
  part4View: "indicator",
  part5: {
    loaded: false,
    loading: false,
    yieldRows: [],
    yieldMonthMap: new Map(),
    yieldYearMap: new Map(),
    stockBetaRows: [],
    stockBetaMap: new Map(),
    excludedSummaryRows: [],
    excludedPanelRows: [],
    excludedEnrichedRows: [],
    excludedTopRows: [],
    excludedRemovedRows: [],
    holdings: [],
    reports: [],
    reportKeys: new Set(),
    reportMap: new Map(),
    activeReportKey: "",
    activeHoldingKey: "",
    analysisMode: "all",
    brushedPeriodLabels: [],
    brushedReportKeys: [],
    brushedHoldingKeys: [],
    selectedManagerNames: [],
    part5BSelectedCategories: [],
    part5BSelectedYears: [],
    part5BSelectedItemKeys: [],
    modeMemory: defaultPart5ModeMemory(),
    managerMatchCache: null,
    managerFilteredCache: null
  },
  part6: {
    mode: "backend",
    backendStatus: "idle",
    backendResult: null,
    lastPayload: null,
    horizonMonths: 12,
    dateStart: "",
    dateEnd: "",
    dateDomain: null,
    activeCluster: 0
  }
};

const dom = {};

window.addEventListener("DOMContentLoaded", init);

function init() {
  [
    "loadStatus", "progressFill", "loadDefaultBtn", "fundCsvInput", "sp500CsvInput",
    "manualSp500Input", "loadUploadedBtn", "resetBtn", "mgmtFilter", "applyMgmtBtn",
    "clearMgmtBtn", "setABtn", "setBBtn", "applyPart1Btn", "clearPart1Btn",
    "part1SelectionStatus", "metricTotal", "metricA", "metricB", "metricBeatA",
    "part2Section", "part2Status", "cumulativeHistInput", "part2LogicSelect", "addPart2RegionBtn",
    "applyPart2Btn", "clearPart2Btn", "metricP2A", "metricP2FundsA", "metricP2B",
    "metricP2FundsB", "part2Grid", "part3Section", "part3Status", "managerTable",
    "addManagersBtn", "applyManagersBtn", "clearManagersBtn", "part4Section",
    "clearRadarBtn", "indicatorChart", "radarChart", "styleDriftChart", "groupTable", "recordTable",
    "loadPart5Btn", "resetPart5Btn", "part5Status", "part5AnalysisModeSelect",
    "part5PeriodSelect", "part5AggregationSelect", "part5RankInput", "part5LimitSelect",
    "part5SearchInput", "part5FocusSelect",
    "applyPart5FiltersBtn", "clearPart5FiltersBtn", "metricP5Reports", "metricP5Funds",
    "metricP5Holdings", "metricP5Yield", "part5OverviewChart", "part5AllocationChart",
    "part5TopHoldingsChart", "part5ReportScatterChart", "part5ReportHoldingsChart",
    "part5ReportDetailStatus", "part5ReportDetailTable", "part5HoldingExplainTable",
    "part5TopHoldingStatus", "part5TopHoldingDetailTable", "part5BrushStatus",
    "clearPart5BrushBtn", "part5ReportPickerStatus", "part5ReportPickerTable",
    "part5TopHoldingPickerStatus", "part5TopHoldingPickerTable", "part5ManagerPanel",
    "part5ManagerStatus", "part5ManagerSummaryCards", "part5ManagerChart", "part5ManagerTable",
    "part5BStatus", "clearPart5BBrushBtn",
    "part5BCaseCards", "part5BOverviewChart", "part5BYearChart", "part5BRateCaseChart",
    "part5BEquityCaseChart", "part5BTopItemsTable", "part5BRemovedAuditTable",
    "part5SummaryTable", "part5HoldingsTable", "part5Results",
    "part6Section", "part6ModeSelect", "part6WindowSelect", "part6TargetSelect", "part6RefreshBtn", "part6Status",
    "runBackendAnalysisBtn", "backendAnalysisStatus", "backendAnalysisSummary",
    "part6FrontendContent", "part6BackendContent", "part7RunAnalysisBtn", "part7Status", "part7ExplanationWorkspace",
    "part7EventSelect", "part7HorizonSelect", "part7ModelInput", "part7WebSearchInput", "part7QuestionInput", "part7RuntimeCards",
    "metricP6Rows", "metricP6Positive", "metricP6HighPositive", "metricP6AvgFuture",
    "part6BucketChart", "part6ScatterChart", "part6PortfolioChart",
    "part6PredictionRankChart", "part6ProbabilityHistogramChart", "part6ShapFeatureChart", "part6SingleEventShapChart",
    "part6CandidateTable", "part6BucketTable",
    "metricBackendPredictionCount", "metricBackendAvgProb", "metricBackendHighProb", "metricBackendTopProb",
    "part6PredictionRankChart", "part6ProbabilityHistogramChart", "part6StockActionChart", "part6ShapFeatureChart", "part6SingleEventShapChart",
    "part6PredictionResultTable", "part6StockActionTable", "part6ShapResultTable",
    "part6DateRangeLabel", "part6DateStartRange", "part6DateEndRange", "part6HorizonTabs",
    "part6ClusterMap", "part6ClusterSummary", "part6FidelityHint", "part6FidelityShapChart", "part6FidelityRawChart",
    "part6ShapEventSelect", "part6SelectAllShapEventsBtn", "part6ClearShapEventsBtn", "part6ShapSelectionHint",
    "part6ExpertLatestCards", "part6ExpertAllocationChart", "part6ExpertRecommendationTable",
    "part6ExpertManagerTable", "part6ExpertCaveat", "part6StockActionSummary"
  ].forEach(id => {
    dom[id] = document.getElementById(id);
  });

  dom.loadDefaultBtn.addEventListener("click", loadDefaultData);
  dom.loadUploadedBtn.addEventListener("click", loadUploadedData);
  dom.resetBtn.addEventListener("click", resetAll);
  dom.applyMgmtBtn.addEventListener("click", applyMgmtFilter);
  dom.clearMgmtBtn.addEventListener("click", clearMgmtFilter);
  dom.manualSp500Input.addEventListener("change", updateManualBenchmark);
  dom.setABtn.addEventListener("click", setRegionA);
  dom.setBBtn.addEventListener("click", setRegionB);
  dom.applyPart1Btn.addEventListener("click", applyPart1);
  dom.clearPart1Btn.addEventListener("click", clearPart1);
  dom.cumulativeHistInput.addEventListener("change", renderPart2);

  if (dom.part2LogicSelect) {
    dom.part2LogicSelect.addEventListener("change", () => {
      state.part2Logic = dom.part2LogicSelect.value || "and";
      renderPart2();
    });
  }

  dom.addPart2RegionBtn.addEventListener("click", addPart2Region);
  dom.applyPart2Btn.addEventListener("click", applyPart2);
  dom.clearPart2Btn.addEventListener("click", clearPart2);
  dom.addManagersBtn.addEventListener("click", addCurrentManagers);
  dom.applyManagersBtn.addEventListener("click", applyManagersToRadar);
  dom.clearManagersBtn.addEventListener("click", clearManagers);
  dom.clearRadarBtn.addEventListener("click", clearRadar);
  dom.loadPart5Btn.addEventListener("click", loadPart5Data);
  dom.resetPart5Btn.addEventListener("click", () => resetPart5Data(true));
  dom.applyPart5FiltersBtn.addEventListener("click", renderPart5);
  dom.clearPart5FiltersBtn.addEventListener("click", clearPart5Filters);
  if (dom.clearPart5BrushBtn) dom.clearPart5BrushBtn.addEventListener("click", clearPart5BrushSelection);
  if (dom.clearPart5BBrushBtn) dom.clearPart5BBrushBtn.addEventListener("click", clearPart5BSelection);
  if (dom.part5AnalysisModeSelect) dom.part5AnalysisModeSelect.addEventListener("change", () => {
    switchPart5AnalysisMode(dom.part5AnalysisModeSelect.value || "all");
  });
  dom.part5PeriodSelect.addEventListener("change", () => { state.part5.brushedPeriodLabels = []; renderPart5(); renderPart6(); });
  dom.part5AggregationSelect.addEventListener("change", () => { state.part5.brushedPeriodLabels = []; renderPart5(); renderPart6(); });
  dom.part5FocusSelect.addEventListener("change", () => { state.part5.brushedPeriodLabels = []; renderPart5(); renderPart6(); });
  dom.part5RankInput.addEventListener("change", () => { renderPart5(); renderPart6(); });
  dom.part5LimitSelect.addEventListener("change", () => { renderPart5(); renderPart6(); });
  dom.part5SearchInput.addEventListener("keydown", event => {
    if (event.key === "Enter") { renderPart5(); renderPart6(); }
  });
  if (dom.part6RefreshBtn) dom.part6RefreshBtn.style.display = "none";
  if (dom.runBackendAnalysisBtn) dom.runBackendAnalysisBtn.addEventListener("click", runBackendAnalysis);
  if (dom.part6ModeSelect) dom.part6ModeSelect.addEventListener("change", () => {
    state.part6.mode = dom.part6ModeSelect.value || "backend";
    renderPart6();
  });
  if (dom.part6WindowSelect) dom.part6WindowSelect.addEventListener("change", renderPart6);
  if (dom.part6TargetSelect) dom.part6TargetSelect.addEventListener("change", () => {
    state.part6.horizonMonths = Number(dom.part6TargetSelect.value) || 12;
    syncPart6HorizonTabs();
    if (state.part6.backendResult) runBackendAnalysis(); else renderPart6();
  });
  if (dom.part6HorizonTabs) dom.part6HorizonTabs.addEventListener("click", event => {
    const button = event.target.closest("button[data-horizon]");
    if (!button) return;
    state.part6.horizonMonths = Number(button.dataset.horizon) || 12;
    if (dom.part6TargetSelect) dom.part6TargetSelect.value = String(state.part6.horizonMonths);
    syncPart6HorizonTabs();
    if (state.part6.backendResult) runBackendAnalysis(); else renderPart6();
  });
  [dom.part6DateStartRange, dom.part6DateEndRange].forEach(input => {
    if (!input) return;
    input.addEventListener("input", updatePart6DateRangeFromControls);
    input.addEventListener("change", () => { updatePart6DateRangeFromControls(); if (state.part6.backendResult) runBackendAnalysis(); });
  });
  if (dom.part6ShapEventSelect) dom.part6ShapEventSelect.addEventListener("change", () => {
    part6ApplyShapEventSelection();
  });
  if (dom.part6SelectAllShapEventsBtn) dom.part6SelectAllShapEventsBtn.addEventListener("click", () => {
    Array.from(dom.part6ShapEventSelect.options).forEach((option, index) => { option.selected = index < 8; });
    part6ApplyShapEventSelection();
  });
  if (dom.part6ClearShapEventsBtn) dom.part6ClearShapEventsBtn.addEventListener("click", () => {
    Array.from(dom.part6ShapEventSelect.options).forEach((option, index) => { option.selected = index === 0; });
    part6ApplyShapEventSelection();
  });
  if (dom.part7RunAnalysisBtn) dom.part7RunAnalysisBtn.addEventListener("click", runPart7AnalysisPlaceholder);
  loadPart7Status();
  part7PopulateEvents();

  document.querySelectorAll("input[name='horizon']").forEach(input => {
    input.addEventListener("change", () => {
      state.horizon = input.value;
      resetPart1Selections();
      rebuildActiveRows();
      drawPart1Scatter();
      updatePart1Metrics();
    });
  });



  document.querySelectorAll("input[name='selectionMode']").forEach(input => {
    input.addEventListener("change", () => {
      state.selectionMode = getCurrentSelectionModeFromDom();

      if (state.selectionMode === "single") {
        state.p1BoxB = null;
        state.rawB = [];
      }

      resetAfterPart1();
      drawPart1Scatter();
      updatePart1Metrics();
      updateControlStates();
    });
  });

  document.querySelectorAll("input[name='part4View']").forEach(input => {
    input.addEventListener("change", () => {
      state.part4View = input.value;
      updatePart4View();
    });
  });

  updateManualBenchmark();
  updateControlStates();

  if (!window.Plotly || !window.Papa) {
    setStatus("Plotly 或 PapaParse 尚未載入，請確認網路可連到 CDN。");
    return;
  }

  if (window.location.protocol === "file:") {
    setStatus("目前用 file:// 開啟。可用上傳 CSV，或用本機伺服器開啟後自動載入內建 CSV。");
  } else {
    loadDefaultData();
  }
}

async function loadDefaultData() {
  if (!window.Papa) {
    setStatus("PapaParse 尚未載入。");
    return;
  }

  resetAll();
  setStatus("載入內建 S&P500 CSV...");
  setProgress(4);

  try {
    updateManualBenchmark();
    const sp500Map = await parseBenchmarkSource(DEFAULT_SP500_FILE, "S&P500");
    const rows = [];
    const counter = { value: 0 };

    for (const file of DEFAULT_FUND_FILES) {
      setStatus(`載入 ${file}...`);
      await parseFundSource(file, file, rows, sp500Map, counter);
    }

    finishDataLoad(rows);
  } catch (error) {
    let detail = "";
    try {
      const diag = await fetch("/api/files");
      if (diag.ok) {
        const info = await diag.json();
        const missing = (info.files || []).filter(item => !item.exists).map(item => item.public_name).slice(0, 8);
        if (missing.length) detail = `；缺少檔案：${missing.join(", ")}`;
      }
    } catch (_) {}
    setStatus(`內建 CSV 載入失敗：${error.message || error}${detail}`);
  }
}

async function loadUploadedData() {
  if (!window.Papa) {
    setStatus("PapaParse 尚未載入。");
    return;
  }

  const fundFiles = Array.from(dom.fundCsvInput.files || []);
  if (!fundFiles.length) {
    alert("請先選擇至少一個基金 CSV。");
    return;
  }

  resetAll();
  updateManualBenchmark();

  try {
    const sp500File = (dom.sp500CsvInput.files || [])[0];
    const sp500Map = sp500File
      ? await parseBenchmarkSource(sp500File, sp500File.name)
      : new Map();

    const rows = [];
    const counter = { value: 0 };
    for (const file of fundFiles) {
      setStatus(`載入 ${file.name}...`);
      await parseFundSource(file, file.name, rows, sp500Map, counter);
    }

    finishDataLoad(rows);
  } catch (error) {
    setStatus(`上傳 CSV 載入失敗：${error.message || error}`);
  }
}

function finishDataLoad(rows) {
  if (!rows.length) {
    setStatus("資料為空，請確認 CSV 欄位包含 caldt、mret、mgmt_name。");
    return;
  }

  setStatus("建立 Trailing 3 年 window...");
  setProgress(82);
  buildRollingHorizons(rows);

  state.rows = rows;
  populateMgmtFilter(rows);
  rebuildActiveRows();
  resetPart1Selections();
  drawPart1Scatter();
  updatePart1Metrics();
  updateControlStates();
  setProgress(100);
  renderPart6();
  setStatus(`完成：${formatInt(rows.length)} 筆基金月資料，${formatInt(state.activeRows.length)} 筆目前可分析。`);
}

function parseBenchmarkSource(source, label) {
  const sp500Map = new Map();
  let dateCol = null;
  let retCol = null;
  let count = 0;

  return parseCsv(source, label, chunkRows => {
    for (const row of chunkRows) {
      if (!dateCol || !retCol) {
        const keys = Object.keys(row);
        dateCol = findFirst(keys, ["caldt", "date", "month", "Date", "DATE"]);
        retCol = findFirst(keys, ["sp500_ret", "sp500_mret", "mret", "ret", "return", "Return"]);
      }
      if (!dateCol || !retCol) continue;

      const key = monthKeyFromValue(row[dateCol]);
      const ret = parseNumber(row[retCol]);
      if (key && Number.isFinite(ret)) {
        const old = sp500Map.get(key);
        if (!old) {
          sp500Map.set(key, { sum: ret, count: 1 });
        } else {
          old.sum += ret;
          old.count += 1;
        }
        count += 1;
      }
    }
    setStatus(`載入 ${label} benchmark：${formatInt(count)} 筆`);
  }).then(() => {
    const averaged = new Map();
    for (const [key, value] of sp500Map.entries()) {
      averaged.set(key, value.sum / value.count);
    }
    return averaged;
  });
}

function parseFundSource(source, label, targetRows, sp500Map, counter) {
  let count = 0;

  return parseCsv(source, label, chunkRows => {
    for (const raw of chunkRows) {
      const row = normalizeFundRow(raw, sp500Map, counter.value);
      counter.value += 1;
      if (row) {
        targetRows.push(row);
        count += 1;
      }
    }
    setStatus(`載入 ${label}：${formatInt(count)} 筆有效資料`);
  });
}

async function parseCsv(source, label, onChunk) {
  let parseSource = source;

  if (typeof source === "string") {
    setStatus(`讀取 ${label} 檔案...`);
    const response = await fetch(source);
    if (!response.ok) {
      throw new Error(`${label}: HTTP ${response.status}`);
    }
    parseSource = await response.blob();
  }

  return new Promise((resolve, reject) => {
    Papa.parse(parseSource, {
      header: true,
      dynamicTyping: false,
      skipEmptyLines: true,
      download: false,
      worker: false,
      chunkSize: 1024 * 1024 * 4,
      chunk(results) {
        onChunk(results.data || []);
        const cursor = results.meta && results.meta.cursor ? results.meta.cursor : 0;
        if (parseSource && parseSource.size && cursor) {
          setProgress(Math.min(80, (cursor / parseSource.size) * 80));
        }
      },
      complete() {
        resolve();
      },
      error(error) {
        reject(new Error(`${label}: ${error.message || error}`));
      }
    });
  });
}

function normalizeFundRow(raw, sp500Map, rowId) {
  const caldtMs = parseDateMs(raw.caldt);
  const monthKey = monthKeyFromMs(caldtMs);
  const mret = parseNumber(raw.mret);
  const mgmtName = cleanText(raw.mgmt_name) || cleanText(raw.mgr_name);

  if (!Number.isFinite(caldtMs) || !monthKey || !Number.isFinite(mret) || !mgmtName) {
    return null;
  }

  let sp500Ret = sp500Map.get(monthKey);
  if (!Number.isFinite(sp500Ret)) {
    sp500Ret = state.manualMonthlySp500;
  }

  const mgrDtMs = Number.isFinite(parseDateMs(raw.mgr_dt)) ? parseDateMs(raw.mgr_dt) : caldtMs;
  const tenure = Math.max(0, (caldtMs - mgrDtMs) / MS_PER_YEAR);
  const yearParts = datePartsFromMs(caldtMs);
  const netFlow = computeNetFlow(raw);

  const row = {
    rowId,
    caldtMs,
    caldt: isoDateFromMs(caldtMs),
    monthKey,
    year: yearParts.year,
    yearFloat: yearParts.year + (yearParts.month - 1) / 12,
    crsp_fundno: cleanText(raw.crsp_fundno) || `row-${rowId}`,
    fund_name: cleanText(raw.fund_name) || "Unknown Fund",
    mgmt_name: mgmtName,
    mgr_name: cleanText(raw.mgr_name) || "Unknown Manager",
    raw_mret: mret,
    raw_sp500_ret: sp500Ret,
    net_flow: netFlow,
    mtna: parseNumber(raw.mtna),
    exp_ratio: parseNumber(raw.exp_ratio),
    mgmt_fee: parseNumber(raw.mgmt_fee),
    turn_ratio: parseNumber(raw.turn_ratio),
    age: parseNumber(raw.age),
    tenure,
    policy: cleanText(raw.policy),
    lipper_class_name: cleanText(raw.lipper_class_name),
    horizons: {}
  };

  row.horizons.monthly = {
    x: sp500Ret,
    y: mret,
    count: 1,
    period_return: mret,
    max_drawdown: maxDrawdownFromMonthly([mret])
  };

  return row;
}

function computeNetFlow(raw) {
  const hasSalesColumns = Object.prototype.hasOwnProperty.call(raw, "new_sls") &&
    Object.prototype.hasOwnProperty.call(raw, "redemp");

  if (hasSalesColumns) {
    return zeroIfMissing(raw.new_sls) +
      zeroIfMissing(raw.rein_sls) +
      zeroIfMissing(raw.oth_sls) -
      zeroIfMissing(raw.redemp);
  }

  return parseNumber(raw.net_flow);
}

function buildRollingHorizons(rows) {
  const windows = [["y3", 36]];

  const fundGroups = groupBy(rows, row => row.crsp_fundno);
  const spRolls = buildSp500RollingMaps(rows, windows);

  for (const group of fundGroups.values()) {
    group.sort((a, b) => a.caldtMs - b.caldtMs);

    for (const [key, windowSize] of windows) {
      const minPeriods = Math.max(3, Math.floor(windowSize * 0.7));
      const queue = [];
      const returnQueue = [];
      const rowQueue = [];
      const factorSums = Object.fromEntries(ROLLING_FACTOR_KEYS.map(factorKey => [factorKey, 0]));
      const factorCounts = Object.fromEntries(ROLLING_FACTOR_KEYS.map(factorKey => [factorKey, 0]));
      let sum = 0;
      let count = 0;

      for (const row of group) {
        const logValue = safeLog1p(row.raw_mret);
        queue.push(logValue);
        returnQueue.push(row.raw_mret);
        rowQueue.push(row);
        if (Number.isFinite(logValue)) {
          sum += logValue;
          count += 1;
        }
        for (const factorKey of ROLLING_FACTOR_KEYS) {
          const value = row[factorKey];
          if (Number.isFinite(value)) {
            factorSums[factorKey] += value;
            factorCounts[factorKey] += 1;
          }
        }
        if (queue.length > windowSize) {
          const old = queue.shift();
          const oldRow = rowQueue.shift();
          returnQueue.shift();
          if (Number.isFinite(old)) {
            sum -= old;
            count -= 1;
          }
          for (const factorKey of ROLLING_FACTOR_KEYS) {
            const oldValue = oldRow[factorKey];
            if (Number.isFinite(oldValue)) {
              factorSums[factorKey] -= oldValue;
              factorCounts[factorKey] -= 1;
            }
          }
        }

        const hasEnoughData = count >= minPeriods;
        const y = hasEnoughData ? annualizeLogSum(sum, count) : NaN;
        const periodReturn = hasEnoughData ? safeExpm1(sum) : NaN;
        const x = spRolls[key].get(row.monthKey);
        row.horizons[key] = {
          x: Number.isFinite(x) ? x : NaN,
          y,
          count,
          period_return: periodReturn,
          max_drawdown: hasEnoughData ? maxDrawdownFromMonthly(returnQueue) : NaN,
          net_flow: factorCounts.net_flow ? factorSums.net_flow : NaN,
          avg_net_flow: factorCounts.net_flow ? factorSums.net_flow / factorCounts.net_flow : NaN,
          mtna: rollingAverage(factorSums, factorCounts, "mtna"),
          exp_ratio: rollingAverage(factorSums, factorCounts, "exp_ratio"),
          mgmt_fee: rollingAverage(factorSums, factorCounts, "mgmt_fee"),
          turn_ratio: rollingAverage(factorSums, factorCounts, "turn_ratio"),
          age: rollingAverage(factorSums, factorCounts, "age"),
          tenure: rollingAverage(factorSums, factorCounts, "tenure")
        };
      }
    }
  }
}

function rollingAverage(sums, counts, key) {
  return counts[key] ? sums[key] / counts[key] : NaN;
}

function buildSp500RollingMaps(rows, windows) {
  const monthMap = new Map();
  for (const row of rows) {
    if (!monthMap.has(row.monthKey) && Number.isFinite(row.raw_sp500_ret)) {
      monthMap.set(row.monthKey, safeLog1p(row.raw_sp500_ret));
    }
  }

  const months = Array.from(monthMap.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  const output = {};

  for (const [key, windowSize] of windows) {
    const minPeriods = Math.max(3, Math.floor(windowSize * 0.7));
    const queue = [];
    let sum = 0;
    let count = 0;
    const map = new Map();

    for (const [monthKey, logValue] of months) {
      queue.push(logValue);
      if (Number.isFinite(logValue)) {
        sum += logValue;
        count += 1;
      }
      if (queue.length > windowSize) {
        const old = queue.shift();
        if (Number.isFinite(old)) {
          sum -= old;
          count -= 1;
        }
      }
      map.set(monthKey, count >= minPeriods ? annualizeLogSum(sum, count) : NaN);
    }
    output[key] = map;
  }

  return output;
}

function populateMgmtFilter(rows) {
  const names = Array.from(new Set(rows.map(row => row.mgmt_name).filter(Boolean))).sort();
  dom.mgmtFilter.innerHTML = "";
  for (const name of names) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    dom.mgmtFilter.appendChild(option);
  }
}

function applyMgmtFilter() {
  state.selectedMgmt = new Set(Array.from(dom.mgmtFilter.selectedOptions).map(option => option.value));
  resetPart1Selections();
  rebuildActiveRows();
  drawPart1Scatter();
  updatePart1Metrics();
  renderPart6();
  setStatus(`公司篩選已套用：${state.selectedMgmt.size ? `${state.selectedMgmt.size} 家` : "全部"}`);
}

function clearMgmtFilter() {
  Array.from(dom.mgmtFilter.options).forEach(option => {
    option.selected = false;
  });
  state.selectedMgmt = new Set();
  resetPart1Selections();
  rebuildActiveRows();
  drawPart1Scatter();
  updatePart1Metrics();
  renderPart6();
}

function rebuildActiveRows() {
  const rows = [];
  const useFilter = state.selectedMgmt.size > 0;
  const baseRows = [];
  const isMonthly = state.horizon === "monthly";
  const periodMonths = HORIZON_MONTH_COUNTS[state.horizon] || 1;

  for (const base of state.rows) {
    if (useFilter && !state.selectedMgmt.has(base.mgmt_name)) continue;
    baseRows.push(base);
  }

  const sp500Roll = isMonthly
    ? null
    : buildSp500RollingMaps(baseRows, [[state.horizon, HORIZON_WINDOWS[state.horizon]]])[state.horizon];

  for (const base of baseRows) {

    const horizon = base.horizons[state.horizon];
    if (!horizon || !Number.isFinite(horizon.y)) continue;

    const mret = horizon.y;
    const sp500 = isMonthly
      ? horizon.x
      : sp500Roll.get(base.monthKey);
    if (!Number.isFinite(sp500)) continue;
    const factorSource = isMonthly ? base : horizon;
    const netFlow = isMonthly ? base.net_flow : horizon.net_flow;
    const mtna = factorSource.mtna;
    const expRatio = factorSource.exp_ratio;
    const mgmtFee = factorSource.mgmt_fee;
    const turnRatio = factorSource.turn_ratio;
    const age = factorSource.age;
    const tenure = factorSource.tenure;

    rows.push({
      rowId: base.rowId,
      horizon: state.horizon,
      horizon_label: HORIZONS[state.horizon].title,
      horizon_months: periodMonths,
      caldtMs: base.caldtMs,
      caldt: base.caldt,
      monthKey: base.monthKey,
      year: base.year,
      yearFloat: base.yearFloat,
      crsp_fundno: base.crsp_fundno,
      fund_name: base.fund_name,
      mgmt_name: base.mgmt_name,
      mgr_name: base.mgr_name,
      current_mret: base.raw_mret,
      current_sp500_ret: base.raw_sp500_ret,
      current_excess_ret: Number.isFinite(base.raw_mret) && Number.isFinite(base.raw_sp500_ret) ? base.raw_mret - base.raw_sp500_ret : NaN,
      current_net_flow: base.net_flow,
      current_mtna: base.mtna,
      current_exp_ratio: base.exp_ratio,
      current_mgmt_fee: base.mgmt_fee,
      current_turn_ratio: base.turn_ratio,
      current_age: base.age,
      current_tenure: base.tenure,
      fund_trailing_return: mret,
      sp500_trailing_return: sp500,
      fund_trailing_excess_return: mret - sp500,
      fund_trailing_period_return: horizon.period_return,
      fund_trailing_max_drawdown: horizon.max_drawdown,
      trailing_avg_net_flow: horizon.avg_net_flow,
      trailing_sum_net_flow: horizon.net_flow,
      trailing_avg_mtna: horizon.mtna,
      trailing_avg_exp_ratio: horizon.exp_ratio,
      trailing_avg_mgmt_fee: horizon.mgmt_fee,
      trailing_avg_turn_ratio: horizon.turn_ratio,
      trailing_avg_age: horizon.age,
      trailing_avg_tenure: horizon.tenure,
      mret,
      sp500_ret: sp500,
      x_ret: sp500,
      y_ret: mret,
      excess_ret: mret - sp500,
      abs_mret: Math.abs(mret),
      abs_sp500: Math.abs(sp500),
      period_return: horizon.period_return,
      window_max_drawdown: horizon.max_drawdown,
      net_flow: netFlow,
      avg_net_flow: isMonthly ? base.net_flow : horizon.avg_net_flow,
      flow_abs: Math.abs(netFlow),
      mtna,
      exp_ratio: expRatio,
      mgmt_fee: mgmtFee,
      turn_ratio: turnRatio,
      age,
      tenure,
      policy: base.policy,
      lipper_class_name: base.lipper_class_name,
      window_count: horizon.count
    });
  }

  state.activeRows = rows;
}

function drawPart1Scatter() {
  if (!window.Plotly) return;

  const plot = document.getElementById("scatterPlot");
  const rows = state.activeRows;

  if (!rows.length) {
    Plotly.purge(plot);
    updatePart1Status("沒有可顯示的資料");
    return;
  }

  const x = new Array(rows.length);
  const y = new Array(rows.length);
  const years = new Array(rows.length);
  const ids = new Array(rows.length);
  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    x[i] = row.x_ret;
    y[i] = row.y_ret;
    years[i] = row.yearFloat;
    ids[i] = row.rowId;
    if (row.x_ret < minVal) minVal = row.x_ret;
    if (row.y_ret < minVal) minVal = row.y_ret;
    if (row.x_ret > maxVal) maxVal = row.x_ret;
    if (row.y_ret > maxVal) maxVal = row.y_ret;
  }

  const traces = [
    {
      type: "scattergl",
      mode: "markers",
      x,
      y,
      customdata: ids,
      name: "全部資料",
      hoverinfo: "skip",
      marker: {
        color: years,
        colorscale: "Viridis",
        size: 4,
        opacity: 0.82,
        line: { width: 0 },
        showscale: true,
        colorbar: { title: "年份", len: 0.5, thickness: 12 }
      }
    },
    {
      type: "scatter",
      mode: "lines",
      x: [minVal, maxVal],
      y: [minVal, maxVal],
      name: "基金 = S&P500",
      hoverinfo: "skip",
      line: { color: "black", dash: "dot", width: 1.5 }
    }
  ];

  const shapes = [];
  if (state.p1BoxA) shapes.push(rectShape(state.p1BoxA, COLORS.aDark, "rgba(31,111,178,0.035)"));
  if (state.selectionMode === "compare" && state.p1BoxB) {
    shapes.push(rectShape(state.p1BoxB, COLORS.bDark, "rgba(230,111,71,0.035)"));
  }

  const xr = niceAxisRange(rows, "x_ret");
  const yr = niceAxisRange(rows, "y_ret");
  const horizonTitle = HORIZONS[state.horizon].title;
  const isMonthly = state.horizon === "monthly";

  const layout = {
    title: `Part 1：${horizonTitle} 基金報酬 vs S&P500`,
    dragmode: "select",
    hovermode: false,
    height: 560,
    margin: { l: 58, r: 24, t: 62, b: 56 },
    shapes,
    xaxis: {
      title: isMonthly ? "S&P500 月報酬" : `S&P500 ${HORIZONS[state.horizon].label} 年化報酬`,
      tickformat: ".1%",
      range: xr || undefined,
      zeroline: true,
      zerolinecolor: "rgba(0,0,0,0.25)"
    },
    yaxis: {
      title: isMonthly ? "基金月報酬" : `基金 ${HORIZONS[state.horizon].label} 年化報酬`,
      tickformat: ".1%",
      range: yr || undefined,
      zeroline: true,
      zerolinecolor: "rgba(0,0,0,0.25)"
    },
    legend: { orientation: "h", y: -0.18 }
  };

  Plotly.react(plot, traces, layout, {
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToAdd: ["select2d", "lasso2d"],
    modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"]
  });

  resetPlotlyHandler(plot, "plotly_selected", eventData => {
    const box = boxFromPlotlySelection(eventData);
    if (!box) return;
    state.latestP1Box = box;
    updatePart1Status(`已框選：x ${formatPct(box.x0)} ~ ${formatPct(box.x1)}，y ${formatPct(box.y0)} ~ ${formatPct(box.y1)}`);
  });
}

function setRegionA() {
  if (!state.latestP1Box) {
    alert("請先在 Part 1 圖上框選一個範圍。");
    return;
  }

  state.selectionMode = getCurrentSelectionModeFromDom();

  state.p1BoxA = { ...state.latestP1Box };
  state.rawA = filterByBox(state.activeRows, state.p1BoxA);
  state.p1Applied = false;

  resetAfterPart1();
  drawPart1Scatter();
  updatePart1Metrics();
  updateControlStates();
}

function setRegionB() {
  state.selectionMode = getCurrentSelectionModeFromDom();
  updateControlStates();

  if (state.selectionMode !== "compare") {
    alert("請先切換到 A/B 比較模式。");
    return;
  }
  if (!state.latestP1Box) {
    alert("請先在 Part 1 圖上框選一個範圍。");
    return;
  }
  state.p1BoxB = { ...state.latestP1Box };
  state.rawB = filterByBox(state.activeRows, state.p1BoxB);
  state.p1Applied = false;
  resetAfterPart1();
  drawPart1Scatter();
  updatePart1Metrics();
}

function applyPart1() {
  if (!state.p1BoxA) {
    alert("請先設定 A 區域。");
    return;
  }

  state.rawA = filterByBox(state.activeRows, state.p1BoxA);
  state.rawB = state.selectionMode === "compare" && state.p1BoxB
    ? filterByBox(state.activeRows, state.p1BoxB)
    : [];

  if (!state.rawA.length) {
    alert("A 區域沒有資料。");
    return;
  }
  if (state.selectionMode === "compare" && !state.rawB.length) {
    alert("B 區域沒有資料。");
    return;
  }

  state.p1Applied = true;
  state.latestP2Region = null;
  state.pendingP2Regions = [];
  state.appliedP2Regions = [];
  dom.part2Section.classList.remove("hidden");
  state.part2TablesA = buildLevelTables(state.rawA);
  state.part2TablesB = buildLevelTables(state.rawB);
  renderPart2();
  updatePart1Metrics();
}

function clearPart1() {
  resetPart1Selections();
  drawPart1Scatter();
  updatePart1Metrics();
}

function resetPart1Selections() {
  state.latestP1Box = null;
  state.p1BoxA = null;
  state.p1BoxB = null;
  state.p1Applied = false;
  state.rawA = [];
  state.rawB = [];
  resetAfterPart1();
  updatePart1Status("尚未框選");
}

function resetAfterPart1() {
  state.latestP2Region = null;
  state.pendingP2Regions = [];
  state.appliedP2Regions = [];
  state.part2TablesA = null;
  state.part2TablesB = null;
  state.part3RawA = [];
  state.part3RawB = [];
  state.latestManagers = [];
  state.pendingManagers = new Set();
  dom.part2Section.classList.add("hidden");
  dom.part3Section.classList.add("hidden");
  dom.part2Grid.innerHTML = "";
  dom.managerTable.innerHTML = "";
  if (window.Plotly) {
    Plotly.purge("managerChart");
  }
  syncPart4Memory();
}

function updatePart1Metrics() {
  const unitLabel = observationUnitLabel();
  dom.metricTotal.previousElementSibling.textContent = `總${unitLabel}`;
  dom.metricA.previousElementSibling.textContent = `A ${unitLabel}`;
  dom.metricB.previousElementSibling.textContent = `B ${unitLabel}`;
  dom.metricTotal.textContent = formatInt(state.activeRows.length);
  dom.metricA.textContent = state.rawA.length ? formatInt(state.rawA.length) : "-";
  dom.metricB.textContent = state.rawB.length ? formatInt(state.rawB.length) : "-";
  dom.metricBeatA.textContent = state.rawA.length
    ? formatPct(mean(state.rawA.map(row => row.y_ret > row.x_ret ? 1 : 0)))
    : "-";
}

function updatePart1Status(text) {
  dom.part1SelectionStatus.textContent = text;
}

function renderPart2() {
  if (dom.part2Section.classList.contains("hidden")) return;

  const compareMode = state.selectionMode === "compare" && state.rawB.length > 0;
  const tablesA = buildLevelTables(state.rawA);
  const tablesB = compareMode ? buildLevelTables(state.rawB) : emptyLevelTables();
  state.part2TablesA = tablesA;
  state.part2TablesB = tablesB;

  const activeRegions = [
    ...state.appliedP2Regions,
    ...state.pendingP2Regions,
    ...(state.latestP2Region ? [state.latestP2Region] : [])
  ];
  const selectedRawA = activeRegions.length
  ? filterRawByRegions(state.rawA, activeRegions, tablesA, state.part2Logic)
  : [];
  const selectedRawB = compareMode && activeRegions.length
  ? filterRawByRegions(state.rawB, activeRegions, tablesB, state.part2Logic)
  : [];
  const selectedTablesA = selectedRawA.length ? buildLevelTables(selectedRawA) : emptyLevelTables();
  const selectedTablesB = selectedRawB.length ? buildLevelTables(selectedRawB) : emptyLevelTables();
  const unitLabel = observationUnitLabel();

  const metricRowsA = activeRegions.length ? selectedRawA : state.rawA;
  const metricRowsB = compareMode
    ? (activeRegions.length ? selectedRawB : state.rawB)
    : [];

  dom.metricP2A.previousElementSibling.textContent = activeRegions.length
    ? `A Part2 篩選後${unitLabel}`
    : `A Part1 基礎${unitLabel}`;

  dom.metricP2FundsA.previousElementSibling.textContent = activeRegions.length
    ? "A Part2 篩選後基金數"
    : "A Part1 基礎基金數";

  dom.metricP2B.previousElementSibling.textContent = activeRegions.length
    ? `B Part2 篩選後${unitLabel}`
    : `B Part1 基礎${unitLabel}`;

  dom.metricP2FundsB.previousElementSibling.textContent = activeRegions.length
    ? "B Part2 篩選後基金數"
    : "B Part1 基礎基金數";

  dom.metricP2A.textContent = formatInt(metricRowsA.length);
  dom.metricP2FundsA.textContent = formatInt(uniqueCount(metricRowsA, row => row.crsp_fundno));

  dom.metricP2B.textContent = compareMode
    ? formatInt(metricRowsB.length)
    : "-";

  dom.metricP2FundsB.textContent = compareMode
    ? formatInt(uniqueCount(metricRowsB, row => row.crsp_fundno))
    : "-";
  

  dom.part2Status.textContent = part2StatusText();
  dom.part2Grid.innerHTML = "";
  const horizonTitle = HORIZONS[state.horizon].title;

  for (const level of LEVELS) {
    const allA = tablesA[level.key];
    const allB = compareMode ? tablesB[level.key] : [];
    const displayFeatureSource = PART2_DISPLAY_FEATURES[level.key] || FEATURES[level.key];
    const available = displayFeatureSource.filter(([key]) => hasNumericValues(allA, key));
    if (!available.length) continue;

    if (!state.selectedFeatures[level.key]) {
      state.selectedFeatures[level.key] = available.slice(0, level.defaultCount).map(([key]) => key);
    }
    state.selectedFeatures[level.key] = state.selectedFeatures[level.key].filter(key => available.some(([k]) => k === key));
    if (!state.selectedFeatures[level.key].length) {
      state.selectedFeatures[level.key] = available.slice(0, level.defaultCount).map(([key]) => key);
    }

    const block = document.createElement("div");
    block.className = "level-block";

    const head = document.createElement("div");
    head.className = "level-head";
    const title = document.createElement("h2");
    title.textContent = `Part 2：${level.label}（${horizonTitle}）`;
    head.appendChild(title);

    const select = document.createElement("select");
    select.className = "feature-select";
    select.multiple = true;
    select.size = Math.min(8, available.length);
    for (const [key, label] of available) {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = part2FeatureLabel(level.key, key, label);
      option.selected = state.selectedFeatures[level.key].includes(key);
      select.appendChild(option);
    }
    select.addEventListener("change", () => {
      state.selectedFeatures[level.key] = Array.from(select.selectedOptions).map(option => option.value);
      renderPart2();
    });
    head.appendChild(select);
    block.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "hist-grid";
    block.appendChild(grid);
    dom.part2Grid.appendChild(block);

    const featureLabels = new Map(available.map(([key, label]) => [key, part2FeatureLabel(level.key, key, label)]));
    for (const featureKey of state.selectedFeatures[level.key]) {
      const cell = document.createElement("div");
      cell.className = "hist-cell";
      const chart = document.createElement("div");
      chart.className = "chart";
      cell.appendChild(chart);
      grid.appendChild(cell);

      makeHistogram(
        chart,
        level.key,
        featureKey,
        featureLabels.get(featureKey) || featureKey,
        allA,
        allB,
        selectedTablesA[level.key],
        selectedTablesB[level.key],
        compareMode,
        activeRegions
      );
    }

    renderPart2ResultTables(
      block,
      level,
      available,
      allA,
      allB,
      selectedTablesA[level.key],
      selectedTablesB[level.key],
      compareMode,
      activeRegions.length > 0
    );
  }
}

function renderPart2ResultTables(block, level, available, allA, allB, selectedA, selectedB, compareMode, hasActiveRegions) {
  const tableStack = document.createElement("div");
  tableStack.className = "part2-table-stack";
  block.appendChild(tableStack);

  const columns = part2TableColumns(level.key, available);
  const rowsA = hasActiveRegions ? selectedA : allA;
  const horizonTitle = HORIZONS[state.horizon].title;
  appendPart2ResultTable(
    tableStack,
    `Part2 ${level.label} A ${hasActiveRegions ? "篩選結果" : "計算結果"}（${horizonTitle}）`,
    rowsA,
    columns
  );

  if (compareMode) {
    const rowsB = hasActiveRegions ? selectedB : allB;
    appendPart2ResultTable(
      tableStack,
      `Part2 ${level.label} B ${hasActiveRegions ? "篩選結果" : "計算結果"}（${horizonTitle}）`,
      rowsB,
      columns
    );
  }
}

function appendPart2ResultTable(parent, title, rows, columns) {
  const safeRows = rows || [];
  if (!safeRows.length) return;

  const container = document.createElement("div");
  container.className = "table-wrap part2-table-wrap";
  parent.appendChild(container);

  const limitedRows = safeRows.slice(0, PART2_TABLE_LIMIT);
  const titleSuffix = safeRows.length > PART2_TABLE_LIMIT ? `（前 ${formatInt(PART2_TABLE_LIMIT)} 筆）` : "";
  renderTable(container, limitedRows, columns, {
    title: `${title}${titleSuffix}`,
    expanded: false,
    count: safeRows.length
  });
}

function part2TableColumns(levelKey, available) {
  const baseColumns = PART2_TABLE_BASE_COLUMNS[levelKey] || [];
  const featureColumns = available.map(([key, label]) => ({
    key,
    label: part2FeatureLabel(levelKey, key, label),
    format: PART2_TABLE_FORMATS[key]
  }));
  return baseColumns.concat(featureColumns);
}

function makeHistogram(container, levelKey, featureKey, label, allA, allB, selectedA, selectedB, compareMode, regions) {
  const edges = sharedHistEdges([allA, compareMode ? allB : [], selectedA, compareMode ? selectedB : []], featureKey, 30);
  if (!edges) {
    Plotly.react(container, [], { title: label, height: 280 }, { displaylogo: false });
    return;
  }

  const centers = [];
  const widths = [];
  for (let i = 0; i < edges.length - 1; i += 1) {
    centers.push((edges[i] + edges[i + 1]) / 2);
    widths.push((edges[i + 1] - edges[i]) * 0.92);
  }

  const cumulative = dom.cumulativeHistInput.checked;
  const hasSelection = regions.length > 0;
  const traces = [];

  addBarTrace(traces, centers, widths, histogramCounts(allA, featureKey, edges, cumulative), hasSelection ? "A 全部" : "A", hasSelection ? "rgba(31,111,178,0.24)" : COLORS.a);
  if (compareMode) {
    addBarTrace(traces, centers, widths, histogramCounts(allB, featureKey, edges, cumulative), hasSelection ? "B 全部" : "B", hasSelection ? "rgba(230,111,71,0.22)" : COLORS.b);
  }
  if (selectedA && selectedA.length) {
    addBarTrace(traces, centers, widths, histogramCounts(selectedA, featureKey, edges, cumulative), "A 篩選", COLORS.aDark);
  }
  if (compareMode && selectedB && selectedB.length) {
    addBarTrace(traces, centers, widths, histogramCounts(selectedB, featureKey, edges, cumulative), "B 篩選", COLORS.bDark);
  }

  const shapes = regions
    .filter(region => region.level === levelKey && region.feature === featureKey)
    .map(region => {
      const [x0, x1] = sortedPair(region.xRange);
      return {
        type: "rect",
        xref: "x",
        yref: "paper",
        x0,
        x1,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(31, 111, 178, 0.10)",
        line: { color: COLORS.aDark, width: 2 }
      };
    });

  Plotly.react(container, traces, {
    title: label,
    height: 280,
    barmode: "overlay",
    dragmode: "select",
    hovermode: false,
    margin: { l: 46, r: 14, t: 46, b: 44 },
    shapes,
    xaxis: { title: label, zeroline: false },
    yaxis: { title: cumulative ? "累積筆數" : "筆數", zeroline: false },
    legend: { orientation: "h", y: -0.25, font: { size: 9 } }
  }, {
    displaylogo: false,
    modeBarButtonsToAdd: ["select2d", "lasso2d"],
    modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"]
  });

  resetPlotlyHandler(container, "plotly_selected", eventData => {
    const xRange = xRangeFromPlotlySelection(eventData);
    if (!xRange) return;
    state.latestP2Region = {
      level: levelKey,
      feature: featureKey,
      label,
      xRange
    };
    dom.part2Status.textContent = `目前範圍：${label} / ${formatNumber(xRange[0])} ~ ${formatNumber(xRange[1])}`;
  });
}

function addPart2Region() {
  if (!state.latestP2Region) {
    alert("請先在 Part 2 histogram 上框選一個範圍。");
    return;
  }

  const sig = regionSignature(state.latestP2Region);
  const exists = state.pendingP2Regions.some(region => regionSignature(region) === sig);
  if (!exists) {
    state.pendingP2Regions.push({ ...state.latestP2Region, xRange: [...state.latestP2Region.xRange] });
  }
  state.latestP2Region = null;
  renderPart2();
}

function applyPart2() {
  if (!state.pendingP2Regions.length) {
    alert("請先加入至少一個 Part 2 範圍。");
    return;
  }

  state.appliedP2Regions = state.pendingP2Regions.map(region => ({
    ...region,
    xRange: [...region.xRange]
  }));
  state.pendingP2Regions = [];
  state.latestP2Region = null;

  state.part3RawA = filterRawByRegions(
    state.rawA,
    state.appliedP2Regions,
    state.part2TablesA || buildLevelTables(state.rawA),
    state.part2Logic
 );

  state.part3RawB = state.selectionMode === "compare"
    ? filterRawByRegions(
        state.rawB,
        state.appliedP2Regions,
        state.part2TablesB || buildLevelTables(state.rawB),
       state.part2Logic
     )
   : [];

  if (!state.part3RawA.length && !state.part3RawB.length) {
    alert("Part 2 範圍沒有選到資料。");
    return;
  }

  dom.part3Section.classList.remove("hidden");
  renderPart2();
  drawManagerChart();
}

function clearPart2() {
  state.latestP2Region = null;
  state.pendingP2Regions = [];
  state.appliedP2Regions = [];
  state.part3RawA = [];
  state.part3RawB = [];
  state.latestManagers = [];
  state.pendingManagers = new Set();
  dom.part3Section.classList.add("hidden");
  dom.managerTable.innerHTML = "";
  if (window.Plotly) {
    Plotly.purge("managerChart");
  }
  syncPart4Memory();
  renderPart2();
}

function buildLevelTables(rawRows) {
  if (!rawRows || !rawRows.length) return emptyLevelTables();

  const familyStats = buildFamilyAverages(rawRows);
  const monthly = rawRows.map(row => {
    const fam = familyStats.get(row.mgmt_name) || {};
    return {
      ...row,
      family_avg_return: fam.avg_return,
      family_avg_fee: fam.avg_fee,
      family_avg_mtna: fam.avg_mtna,
      family_avg_flow: fam.avg_flow
    };
  });

  return {
    monthly,
    fund: buildFundTable(rawRows),
    family: buildFamilyTable(rawRows)
  };
}

function emptyLevelTables() {
  return { monthly: [], fund: [], family: [] };
}

function currentHorizonMeta() {
  return {
    key: state.horizon,
    isMonthly: state.horizon === "monthly",
    label: HORIZONS[state.horizon].label,
    title: HORIZONS[state.horizon].title,
    months: HORIZON_MONTH_COUNTS[state.horizon] || 1
  };
}

function horizonAnnualValue(value, meta = currentHorizonMeta()) {
  if (!Number.isFinite(value)) return NaN;
  return meta.isMonthly ? safeAnnualizedReturnFromMonthlyMean(value) : value;
}

function horizonVolatility(values, meta = currentHorizonMeta()) {
  const clean = values.filter(Number.isFinite);
  if (clean.length <= 1) return NaN;
  const scale = meta.isMonthly ? Math.sqrt(12) : 1;
  return sampleStd(clean) * scale;
}

function minValue(values) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? Math.min(...clean) : NaN;
}

function part2FeatureLabel(levelKey, key, fallback) {
  const meta = currentHorizonMeta();
  if (meta.isMonthly) return fallback;

  const period = meta.label;
  const map = {
    monthly: {
      mret: `基金${period}年化報酬率`,
      sp500_ret: `S&P500 ${period}年化報酬率`,
      excess_ret: `基金${period}年化超額報酬`,
      abs_mret: `基金${period}年化報酬絕對值`,
      net_flow: `${period}累積 Net Flow`,
      flow_abs: `${period}累積 Flow 絕對值`,
      mtna: `${period}平均 MTNA 資產規模`,
      exp_ratio: `${period}平均費用率`,
      tenure: `${period}平均管理年資`,
      turn_ratio: `${period}平均換手率`,
      year: "觀測結束年份",
      family_avg_return: `家族平均${period}年化報酬率`,
      family_avg_fee: `家族${period}平均費用率`,
      family_avg_mtna: `家族${period}平均規模`,
      family_avg_flow: `家族${period}平均累積淨申購`
    },
    fund: {
      obs: "觀測窗口數",
      mean_return: `平均${period}年化報酬率`,
      annual_return: `代表${period}年化報酬率`,
      cumulative_return: `平均${period}累積報酬率`,
      annual_volatility: `${period}年化報酬波動度`,
      max_drawdown: `${period}窗口 Max Drawdown`,
      win_rate: `${period}窗口勝率`,
      beat_sp500_rate: `打敗 S&P500 ${period}窗口比例`,
      annual_excess: `${period}年化超額報酬`,
      tracking_error: `${period}Tracking Error`,
      alpha: `${period}Alpha 近似`,
      avg_net_flow: `平均${period}累積 Net Flow`,
      sum_net_flow: `累積${period}窗口 Net Flow`,
      avg_exp_ratio: `${period}平均費用率`,
      avg_tenure: `${period}平均管理年資`,
      avg_turnover: `${period}平均換手率`
    },
    family: {
      obs: "觀測窗口數",
      avg_return: `家族平均${period}年化報酬率`,
      avg_excess: `家族平均${period}年化超額報酬`,
      avg_fee: `家族${period}平均費用率`,
      avg_mtna: `家族${period}平均規模`,
      avg_flow: `家族平均${period}累積淨申購`,
      sum_flow: `家族累積${period}窗口淨申購`,
      avg_turnover: `家族${period}平均換手率`,
      avg_tenure: `家族${period}平均管理年資`
    }
  };

  return (map[levelKey] && map[levelKey][key]) || fallback;
}

function observationUnitLabel(meta = currentHorizonMeta()) {
  return meta.isMonthly ? "筆數" : "窗口數";
}

function buildFundTable(rawRows) {
  const groups = groupBy(rawRows, row => row.crsp_fundno);
  const rows = [];
  const meta = currentHorizonMeta();

  for (const [fundId, group] of groups.entries()) {
    group.sort((a, b) => a.caldtMs - b.caldtMs);
    const returns = finiteValues(group, "mret");
    if (!returns.length) continue;

    const aligned = group.filter(row => Number.isFinite(row.mret) && Number.isFinite(row.sp500_ret));
    const rAligned = aligned.map(row => row.mret);
    const bAligned = aligned.map(row => row.sp500_ret);
    const excess = aligned.map(row => row.mret - row.sp500_ret);
    const meanM = mean(returns);
    const periodReturns = finiteValues(group, "period_return");
    const windowDrawdowns = finiteValues(group, "window_max_drawdown");
    const annualReturn = horizonAnnualValue(meanM, meta);
    const annualVol = horizonVolatility(returns, meta);
    const downside = returns.filter(value => value < 0);
    const downsideVol = horizonVolatility(downside, meta);
    const var95 = returns.length >= 2 ? percentile(returns, 0.05) : NaN;
    const annualExcess = excess.length ? horizonAnnualValue(mean(excess), meta) : NaN;
    const trackingError = horizonVolatility(excess, meta);

    const beta = rAligned.length > 1 ? safeBeta(rAligned, bAligned) : NaN;
    const alphaBase = Number.isFinite(beta) ? mean(rAligned) - beta * mean(bAligned) : NaN;

    rows.push({
      crsp_fundno: fundId,
      mgmt_name: firstDefined(group, "mgmt_name"),
      fund_name: firstDefined(group, "fund_name"),
      mgr_name: mode(group.map(row => row.mgr_name)),
      obs: returns.length,
      mean_return: meanM,
      annual_return: annualReturn,
      cumulative_return: meta.isMonthly ? safeCompoundReturn(returns) : mean(periodReturns),
      annual_volatility: annualVol,
      sharpe: annualVol > 0 ? (annualReturn - RISK_FREE_RATE) / annualVol : NaN,
      sortino: downsideVol > 0 ? (annualReturn - RISK_FREE_RATE) / downsideVol : NaN,
      max_drawdown: meta.isMonthly ? maxDrawdownFromMonthly(returns) : minValue(windowDrawdowns),
      win_rate: mean(returns.map(value => value > 0 ? 1 : 0)),
      beat_sp500_rate: rAligned.length ? mean(aligned.map(row => row.mret > row.sp500_ret ? 1 : 0)) : NaN,
      annual_excess: annualExcess,
      tracking_error: trackingError,
      information_ratio: trackingError > 0 ? annualExcess / trackingError : NaN,
      beta,
      alpha: horizonAnnualValue(alphaBase, meta),
      upside_capture: captureRatio(rAligned, bAligned, true),
      downside_capture: captureRatio(rAligned, bAligned, false),
      corr_sp500: safeCorr(rAligned, bAligned),
      var95,
      cvar95: Number.isFinite(var95) ? mean(returns.filter(value => value <= var95)) : NaN,
      skewness: returns.length >= 3 ? skewness(returns) : NaN,
      kurtosis: returns.length >= 4 ? kurtosis(returns) : NaN,
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_net_flow: mean(finiteValues(group, "net_flow")),
      sum_net_flow: sum(finiteValues(group, "net_flow")),
      avg_exp_ratio: mean(finiteValues(group, "exp_ratio")),
      avg_tenure: mean(finiteValues(group, "tenure")),
      avg_turnover: mean(finiteValues(group, "turn_ratio"))
    });
  }

  return rows;
}

function buildFamilyTable(rawRows) {
  const groups = groupBy(rawRows, row => row.mgmt_name);
  const rows = [];

  for (const [family, group] of groups.entries()) {
    rows.push({
      mgmt_name: family,
      obs: group.length,
      fund_count: uniqueCount(group, row => row.crsp_fundno),
      manager_count: uniqueCount(group, row => row.mgr_name),
      avg_return: mean(finiteValues(group, "mret")),
      avg_excess: mean(finiteValues(group, "excess_ret")),
      avg_fee: mean(finiteValues(group, "exp_ratio")),
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_flow: mean(finiteValues(group, "net_flow")),
      sum_flow: sum(finiteValues(group, "net_flow")),
      avg_turnover: mean(finiteValues(group, "turn_ratio")),
      avg_tenure: mean(finiteValues(group, "tenure"))
    });
  }

  return rows;
}

function buildFamilyAverages(rawRows) {
  const table = buildFamilyTable(rawRows);
  const map = new Map();
  for (const row of table) {
    map.set(row.mgmt_name, row);
  }
  return map;
}

function filterRawByRegions(rawRows, regions, tables, logic = "or") {
  if (!rawRows || !rawRows.length || !regions || !regions.length) return [];

  const rawById = new Map(rawRows.map(row => [row.rowId, row]));

  function rowIdsForRegion(region) {
    const table = (tables && tables[region.level]) || [];
    if (!table.length) return new Set();

    const [lo, hi] = sortedPair(region.xRange);

    const picked = table.filter(row => {
      const value = row[region.feature];
      return Number.isFinite(value) && value >= lo && value <= hi;
    });

    const ids = new Set();

    if (region.level === "monthly") {
      for (const row of picked) {
        if (rawById.has(row.rowId)) {
          ids.add(row.rowId);
        }
      }
    } else if (region.level === "fund") {
      const fundIds = new Set(picked.map(row => row.crsp_fundno));
      for (const row of rawRows) {
        if (fundIds.has(row.crsp_fundno)) {
          ids.add(row.rowId);
        }
      }
    } else if (region.level === "family") {
      const families = new Set(picked.map(row => row.mgmt_name));
      for (const row of rawRows) {
        if (families.has(row.mgmt_name)) {
          ids.add(row.rowId);
        }
      }
    }

    return ids;
  }

  const regionSets = regions.map(rowIdsForRegion);

  if (!regionSets.length) return [];

  let selectedRowIds;

  if (logic === "and") {
    selectedRowIds = new Set(rawRows.map(row => row.rowId));

    for (const ids of regionSets) {
      selectedRowIds = new Set(
        [...selectedRowIds].filter(rowId => ids.has(rowId))
      );

      if (!selectedRowIds.size) break;
    }
  } else {
    selectedRowIds = new Set();

    for (const ids of regionSets) {
      for (const rowId of ids) {
        selectedRowIds.add(rowId);
      }
    }
  }

  return rawRows.filter(row => selectedRowIds.has(row.rowId));
}

function drawManagerChart() {
  if (!window.Plotly) return;

  const compareMode = state.selectionMode === "compare" && state.part3RawB.length > 0;
  const horizonTitle = HORIZONS[state.horizon].title;
  const unitLabel = observationUnitLabel();
  const countA = countManagers(state.part3RawA);
  const countB = compareMode ? countManagers(state.part3RawB) : new Map();
  const managers = mergeManagerCounts(countA, countB, compareMode);

  dom.part3Status.textContent = part3StatusText();
  renderTable("managerTable", managers.slice(0, 500), [
    { key: "manager", label: "經理人" },
    { key: "countA", label: `A ${unitLabel}`, format: "int" },
    { key: "countB", label: `B ${unitLabel}`, format: "int" },
    { key: "total", label: "合計", format: "int" },
    { key: "group", label: "來源" }
  ], { title: `Part3 經理人明細（${horizonTitle}）`, expanded: false });

  Plotly.react("managerChart", [{
    type: "bar",
    x: managers.map(row => row.manager),
    y: managers.map(row => row.total),
    customdata: managers.map(row => row.manager),
    marker: { color: managers.map(row => row.color) },
    hovertemplate: `經理人：%{x}<br>${unitLabel}：%{y}<extra></extra>`,
    name: "經理人"
  }], {
    title: `${compareMode ? "Part 3：A/B 經理人比較" : "Part 3：A 區域經理人"}（${horizonTitle}）`,
    height: Math.max(560, Math.min(1400, 420 + managers.length * 6)),
    dragmode: "select",
    margin: { l: 60, r: 24, t: 58, b: 220 },
    xaxis: { tickangle: -50, automargin: true },
    yaxis: { title: unitLabel },
    showlegend: false
  }, {
    displaylogo: false,
    modeBarButtonsToAdd: ["select2d", "lasso2d"]
  });

  const plot = document.getElementById("managerChart");
  resetPlotlyHandler(plot, "plotly_selected", eventData => {
    if (!eventData || !eventData.points) return;
    state.latestManagers = Array.from(new Set(eventData.points.map(point => String(point.customdata)).filter(Boolean))).sort();
    dom.part3Status.textContent = part3StatusText();
  });
}

function addCurrentManagers() {
  if (!state.latestManagers.length) {
    alert("請先在 Part 3 圖上框選經理人。");
    return;
  }
  for (const manager of state.latestManagers) {
    state.pendingManagers.add(manager);
  }
  dom.part3Status.textContent = part3StatusText();
}

function applyManagersToRadar() {
  const managers = Array.from(state.pendingManagers).sort();
  if (!managers.length) {
    alert("請先加入至少一位經理人。");
    return;
  }

  const selectedRaw = state.selectionMode === "compare"
    ? state.part3RawA.concat(state.part3RawB)
    : state.part3RawA;

  // 全部市場基準：目前 horizon 下的所有可分析資料
  const marketRaw = state.activeRows;

  // selectedBase：目前 Part1~Part3 選到的經理人原始指標
  const selectedBase = buildManagerRadarBase(selectedRaw);

  // marketBase：全部市場經理人的指標分布
  const marketBase = buildManagerRadarBase(marketRaw);

  // 將 selectedBase 的 score 改成相對全部市場經理人
  const base = rescoreManagerBaseWithBenchmark(selectedBase, marketBase);

  const sourceLabel =
    `${state.selectionMode === "compare" ? "A/B 比較" : "單一區域"} | ` +
    `${HORIZONS[state.horizon].title} | 分數基準：全部市場經理人`;

  const sourceKey =
    `${sourceLabel}|${regionsSignature(state.appliedP2Regions)}|` +
    `selected=${selectedRaw.length}|market=${marketRaw.length}`;

  const existing = new Set(state.radarRecords.map(record => record.recordKey));
  let added = 0;

  for (const manager of managers) {
    const record = makeManagerRadarRecord(base, manager, sourceLabel, sourceKey);
    if (record && !existing.has(record.recordKey)) {
      state.radarRecords.push(record);
      existing.add(record.recordKey);
      added += 1;
    }
  }

  state.latestManagers = [];
  state.pendingManagers = new Set();
  dom.part4Section.classList.remove("hidden");
  drawRadar();
  dom.part3Status.textContent = `已加入 ${added} 位經理人到 Part4。`;
}

function clearManagers() {
  state.latestManagers = [];
  state.pendingManagers = new Set();
  dom.part3Status.textContent = part3StatusText();
}

function buildManagerRadarBase(rawRows) {
  const groups = groupBy(rawRows, row => row.mgr_name || "Unknown Manager");
  const rows = [];
  const meta = currentHorizonMeta();

  for (const [manager, group] of groups.entries()) {
    const returns = finiteValues(group, "mret");
    if (!returns.length) continue;

    const aligned = group.filter(row => Number.isFinite(row.mret) && Number.isFinite(row.sp500_ret));
    const excess = aligned.map(row => row.mret - row.sp500_ret);
    const annualReturn = horizonAnnualValue(mean(returns), meta);
    const annualVol = horizonVolatility(returns, meta);
    const windowDrawdowns = finiteValues(group, "window_max_drawdown");

    rows.push({
      manager,
      obs: group.length,
      fund_count: uniqueCount(group, row => row.crsp_fundno),
      family_count: uniqueCount(group, row => row.mgmt_name),
      mean_return: mean(returns),
      annual_return: annualReturn,
      avg_excess: excess.length ? mean(excess) : NaN,
      beat_rate: aligned.length ? mean(aligned.map(row => row.mret > row.sp500_ret ? 1 : 0)) : NaN,
      annual_volatility: annualVol,
      sharpe: annualVol > 0 ? (annualReturn - RISK_FREE_RATE) / annualVol : NaN,
      max_drawdown: meta.isMonthly ? maxDrawdownFromMonthly(returns) : minValue(windowDrawdowns),
      avg_fee: mean(finiteValues(group, "exp_ratio")),
      avg_flow: mean(finiteValues(group, "net_flow")),
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_tenure: mean(finiteValues(group, "tenure")),
      avg_turnover: mean(finiteValues(group, "turn_ratio"))
    });
  }

  for (const metric of RADAR_METRICS) {
    const values = rows.map(row => row[metric.rawKey]);
    for (const row of rows) {
      let score = percentileRankAverage(values, row[metric.rawKey]);
      if (!metric.higherBetter) score = 1 - score;
      row[`score_${metric.rawKey}`] = Number.isFinite(score) ? score : 0.5;
    }
  }

  return rows;
}


function rescoreManagerBaseWithBenchmark(targetRows, benchmarkRows) {
  if (!targetRows || !targetRows.length || !benchmarkRows || !benchmarkRows.length) {
    return targetRows;
  }

  for (const metric of RADAR_METRICS) {
    const benchmarkValues = benchmarkRows
      .map(row => row[metric.rawKey])
      .filter(Number.isFinite);

    for (const row of targetRows) {
      let score = percentileRankAverage(benchmarkValues, row[metric.rawKey]);

      if (!metric.higherBetter) {
        score = 1 - score;
      }

      row[`score_${metric.rawKey}`] = Number.isFinite(score) ? score : 0.5;
    }
  }

  return targetRows;
}

function makeManagerRadarRecord(baseRows, manager, sourceLabel, sourceKey) {
  const row = baseRows.find(item => String(item.manager) === String(manager));
  if (!row) return null;

  const scores = {};
  for (const metric of RADAR_METRICS) {
    scores[metric.label] = row[`score_${metric.rawKey}`] ?? 0.5;
  }

  return {
    recordKey: `${manager}__${sourceKey}`,
    manager,
    sourceLabel,
    scores,
    raw: { ...row }
  };
}

function drawIndicatorChart(groups) {
  if (!groups.length) return;

  const cols = groups.length > 1 ? 2 : 1;
  const rows = Math.ceil(groups.length / cols);
  const rowHeightPx = 390;
  const chartHeight = Math.max(460, rows * rowHeightPx);
  const colWidth = 1 / cols;
  const rowHeight = 1 / rows;
  const xGap = Math.min(0.05, colWidth * 0.12);
  const yGap = Math.min(0.055, rowHeight * 0.2);
  const annotationOffset = Math.min(0.035, yGap + 0.012);
  const metricLabels = RADAR_METRICS.map(metric => metric.label);
  const yTickVals = INDICATOR_TICK_VALUES.map(value => value * 100);
  const layout = {
    title: "Part 4：Manager Indicator 平行座標圖（同群比較）",
    height: chartHeight,
    margin: { l: 58, r: 32, t: 88, b: 54 },
    hovermode: "closest",
    showlegend: state.radarRecords.length <= 28,
    annotations: [],
    shapes: [],
    legend: { orientation: "h", y: -0.12, font: { size: 10 } }
  };
  const traces = [];

  groups.forEach((group, groupIndex) => {
    const axisSuffix = groupIndex === 0 ? "" : String(groupIndex + 1);
    const xAxisName = `xaxis${axisSuffix}`;
    const yAxisName = `yaxis${axisSuffix}`;
    const xRef = groupIndex === 0 ? "x" : `x${groupIndex + 1}`;
    const yRef = groupIndex === 0 ? "y" : `y${groupIndex + 1}`;
    const col = groupIndex % cols;
    const row = Math.floor(groupIndex / cols);
    const x0 = col * colWidth + xGap;
    const x1 = (col + 1) * colWidth - xGap;
    const yTop = 1 - row * rowHeight;
    const yBottom = 1 - (row + 1) * rowHeight;
    const y0 = yBottom + yGap;
    const y1 = yTop - yGap;

    layout[xAxisName] = {
      domain: [x0, x1],
      anchor: yRef,
      type: "category",
      categoryorder: "array",
      categoryarray: metricLabels,
      tickangle: -25,
      showgrid: false,
      zeroline: false,
      automargin: true
    };
    layout[yAxisName] = {
      domain: [y0, y1],
      anchor: xRef,
      title: groupIndex % cols === 0 ? "標準化分數" : "",
      range: [0, 100],
      tickvals: yTickVals,
      ticktext: INDICATOR_TICK_TEXT,
      gridcolor: "#edf1f5",
      zeroline: false
    };
    layout.annotations.push({
      text: `Group ${groupIndex + 1}（${group.length} 位）`,
      x: (x0 + x1) / 2,
      y: Math.min(1, y1 + annotationOffset),
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 14, color: "#334150" }
    });

    for (const label of metricLabels) {
      layout.shapes.push({
        type: "line",
        xref: xRef,
        yref: yRef,
        x0: label,
        x1: label,
        y0: 0,
        y1: 100,
        line: { color: "#dce2e8", width: 1 }
      });
    }

    group.forEach((record, recordIndex) => {
      const color = INDICATOR_COLORS[recordIndex % INDICATOR_COLORS.length];
      traces.push({
        type: "scatter",
        mode: "lines+markers",
        xaxis: xRef,
        yaxis: yRef,
        x: metricLabels,
        y: RADAR_METRICS.map(metric => (record.scores[metric.label] ?? 0.5) * 100),
        name: `${record.manager} | Group ${groupIndex + 1}`,
        legendgroup: `Group ${groupIndex + 1}`,
        line: { color, width: 2 },
        marker: { color, size: 5 },
        opacity: 0.84,
        hovertemplate: `${record.manager}<br>%{x}<br>分數：%{y:.1f}<extra></extra>`
      });
    });
  });

  dom.indicatorChart.style.minHeight = `${chartHeight}px`;
  Plotly.react(dom.indicatorChart, traces, layout, {
    displaylogo: false,
    responsive: true
  });
}

function drawRadar(updateViewAfterDraw = true) {
  if (!window.Plotly || !state.radarRecords.length) return;

  const groups = groupRadarRecords(state.radarRecords);
  drawIndicatorChart(groups);
  const cols = groups.length > 1 ? 2 : 1;
  const rows = Math.ceil(groups.length / cols);
  const rowHeightPx = 520;
  const chartHeight = Math.max(620, rows * rowHeightPx);
  const colWidth = 1 / cols;
  const rowHeight = 1 / rows;
  const xGap = Math.min(0.04, colWidth * 0.1);
  const yGap = Math.min(0.04, rowHeight * 0.18);
  const annotationOffset = Math.min(0.03, yGap + 0.01);
  const layout = {
    title: "Part 4：Radar 分數與 cosine similarity 群組",
    showlegend: true,
    height: chartHeight,
    margin: { l: 42, r: 42, t: 90, b: 50 },
    annotations: []
  };
  const traces = [];
  const theta = RADAR_METRICS.map(metric => metric.label);
  const thetaClosed = theta.concat(theta[0]);

  groups.forEach((group, index) => {
    const polarName = index === 0 ? "polar" : `polar${index + 1}`;
    const col = index % cols;
    const row = Math.floor(index / cols);
    const x0 = col * colWidth + xGap;
    const x1 = (col + 1) * colWidth - xGap;
    const yTop = 1 - row * rowHeight;
    const yBottom = 1 - (row + 1) * rowHeight;
    const y0 = yBottom + yGap;
    const y1 = yTop - yGap;

    layout[polarName] = {
      domain: { x: [x0, x1], y: [y0, y1] },
      radialaxis: { visible: true, range: [0, 1], tickformat: ".0%" }
    };
    layout.annotations.push({
      text: `Group ${index + 1}`,
      x: (x0 + x1) / 2,
      y: Math.min(1, y1 + annotationOffset),
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 14 }
    });

    for (const record of group) {
      const values = theta.map(label => record.scores[label] ?? 0.5);
      traces.push({
        type: "scatterpolar",
        subplot: polarName,
        r: values.concat(values[0]),
        theta: thetaClosed,
        fill: "toself",
        opacity: 0.62,
        name: `${record.manager} | Group ${index + 1}`,
        hovertemplate: "%{theta}<br>分數：%{r:.2f}<extra></extra>"
      });
    }
  });

  const radarChart = document.getElementById("radarChart");
  radarChart.style.minHeight = `${chartHeight}px`;
  Plotly.react(radarChart, traces, layout, { displaylogo: false });

  renderTable("groupTable", groups.map((group, index) => ({
    group: `Group ${index + 1}`,
    count: group.length,
    managers: group.map(record => record.manager).join("、")
  })), [
    { key: "group", label: "群組" },
    { key: "count", label: "經理人數", format: "int" },
    { key: "managers", label: "經理人" }
  ], { title: "Part4 相似群組", expanded: true });

  const recordRows = state.radarRecords.map(record => {
    const row = {
      manager: record.manager,
      sourceLabel: record.sourceLabel
    };
    for (const metric of RADAR_METRICS) {
      row[`score_${metric.rawKey}`] = record.scores[metric.label];
      row[metric.rawKey] = record.raw[metric.rawKey];
    }
    return row;
  });

  renderTable("recordTable", recordRows, [
    { key: "manager", label: "經理人" },
    { key: "sourceLabel", label: "來源" },
    ...RADAR_METRICS.map(metric => ({ key: `score_${metric.rawKey}`, label: `${metric.label}分數`, format: "pct" })),
    ...RADAR_METRICS.map(metric => ({ key: metric.rawKey, label: metric.label, format: metric.format }))
  ], { title: "Part4 分數與原始指標", expanded: false });
  if (updateViewAfterDraw) updatePart4View(false);
}

function groupRadarRecords(records) {
  const vectors = records.map(record => RADAR_METRICS.map(metric => record.scores[metric.label] ?? 0.5));
  const groups = [];
  const used = new Set();

  for (let i = 0; i < records.length; i += 1) {
    if (used.has(i)) continue;
    used.add(i);
    const group = [records[i]];
    const vi = vectors[i];

    for (let j = i + 1; j < records.length; j += 1) {
      if (used.has(j)) continue;
      if (cosineSimilarity(vi, vectors[j]) >= 0.92) {
        used.add(j);
        group.push(records[j]);
      }
    }
    groups.push(group);
  }

  return groups;
}

function updatePart4View(redraw = true) {
  const view = state.part4View || "indicator";
  const showIndicator = view === "indicator";
  const showRadar = view === "radar";
  const showDrift = view === "style_drift";

  if (dom.indicatorChart) dom.indicatorChart.classList.toggle("hidden", !showIndicator);
  if (dom.radarChart) dom.radarChart.classList.toggle("hidden", !showRadar);
  if (dom.styleDriftChart) dom.styleDriftChart.classList.toggle("hidden", !showDrift);

  // Plotly charts rendered while hidden sometimes appear blank after switching tabs.
  // Redraw the selected Part4 view after it becomes visible, then force a resize.
  if (redraw && state.radarRecords && state.radarRecords.length) {
    if (showIndicator) {
      drawIndicatorChart(groupRadarRecords(state.radarRecords));
    } else if (showRadar) {
      drawRadar(false);
    } else if (showDrift) {
      drawStyleDriftChart();
    }
  } else if (showDrift) {
    drawStyleDriftChart();
  }

  const activeChart = showIndicator ? dom.indicatorChart : (showRadar ? dom.radarChart : dom.styleDriftChart);
  if (window.Plotly && activeChart) {
    setTimeout(() => Plotly.Plots.resize(activeChart), 80);
  }
}

function syncPart4Memory() {
  if (state.radarRecords.length) {
    dom.part4Section.classList.remove("hidden");
    drawRadar();
    return;
  }

  dom.part4Section.classList.add("hidden");
  dom.groupTable.innerHTML = "";
  dom.recordTable.innerHTML = "";
  if (window.Plotly) {
    Plotly.purge("indicatorChart");
    Plotly.purge("radarChart");
    if (dom.styleDriftChart) Plotly.purge("styleDriftChart");
  }
}

function clearRadar() {
  state.radarRecords = [];
  syncPart4Memory();
}

async function loadPart5Data() {
  if (!window.Papa) {
    setPart5Status("PapaParse 尚未載入，無法解析 Part5 CSV。");
    return;
  }
  if (state.part5.loading) return;

  resetPart5Data(false);
  state.part5.loading = true;
  dom.loadPart5Btn.disabled = true;
  setPart5Status("載入 10 年期美債殖利率...");

  try {
    state.part5.yieldRows = await loadPart5YieldRows();
    buildPart5YieldMaps();

    setPart5Status("載入 Part5 個股 trailing beta 與 sector 資料...");
    state.part5.stockBetaRows = await loadPart5StockBetaRows();
    buildPart5StockBetaMap();

    for (const source of PART5_STOCK_SOURCES) {
      setPart5Status(`載入 ${source.label} 持股資料...`);
      await parsePart5StockSource(source);
    }

    finalizePart5BetaMetrics();
    setPart5Status("載入 Part5B non-individual holdings exposure 資料...");
    await loadPart5ExcludedTwoGroupData();
    state.part5.managerMatchCache = null;
    state.part5.managerFilteredCache = null;
    state.part5.loaded = true;
    state.part5.loading = false;
    dom.loadPart5Btn.disabled = false;
    const betaMatched = state.part5.holdings.filter(row => row.hasStockBeta).length;
    setPart5Status(`完成：${formatInt(state.part5.reports.length)} 個基金報告、${formatInt(state.part5.holdings.length)} 筆持股明細；其中 ${formatInt(betaMatched)} 筆已接上個股 beta/sector，Part5B 載入 ${formatInt(state.part5.excludedEnrichedRows.length)} 筆 non-individual exposure。`);
    renderPart5();
    renderPart6();
  } catch (error) {
    state.part5.loading = false;
    dom.loadPart5Btn.disabled = false;
    setPart5Status(`Part5 載入失敗：${error.message || error}`);
  }
}

async function loadPart5YieldRows() {
  const text = await fetchText(PART5_YIELD_FILE);
  const parsed = Papa.parse(text, { header: false, skipEmptyLines: true });
  const rows = [];

  for (const item of parsed.data || []) {
    const dateText = cleanText(item[0]);
    if (!/^\d{4}-\d{2}$/.test(dateText)) continue;
    const value = parseNumber(item[1]);
    if (!Number.isFinite(value)) continue;

    const dateMs = parseDateMs(dateText);
    const parts = datePartsFromMs(dateMs);
    rows.push({
      monthKey: monthKeyFromMs(dateMs),
      dateMs,
      year: parts.year,
      yield10y: value
    });
  }

  return rows;
}

function buildPart5YieldMaps() {
  const byYear = new Map();
  state.part5.yieldMonthMap = new Map();

  for (const row of state.part5.yieldRows) {
    state.part5.yieldMonthMap.set(row.monthKey, row.yield10y);
    if (!byYear.has(row.year)) byYear.set(row.year, []);
    byYear.get(row.year).push(row.yield10y);
  }

  state.part5.yieldYearMap = new Map();
  for (const [year, values] of byYear.entries()) {
    state.part5.yieldYearMap.set(year, mean(values));
  }
}

async function loadPart5StockBetaRows() {
  const rows = [];
  let count = 0;
  await parseCsv(PART5_BETA_FILE, "Part5 yearly trailing stock beta", chunkRows => {
    for (const raw of chunkRows) {
      const row = normalizePart5StockBetaRow(raw);
      if (!row) continue;
      rows.push(row);
      count += 1;
    }
    setPart5Status(`載入個股 beta / sector 資料：${formatInt(count)} 筆`);
  });
  return rows;
}

function normalizePart5StockBetaRow(raw) {
  const year = parseNumber(raw.year);
  const holdingTicker = cleanText(raw.holding_ticker);
  const yahooTicker = cleanText(raw.yahoo_ticker);
  if (!Number.isFinite(year) || (!holdingTicker && !yahooTicker)) return null;

  return {
    holding_ticker: holdingTicker,
    yahoo_ticker: yahooTicker,
    primary_security_name: cleanText(raw.primary_security_name),
    sector: cleanText(raw.sector) || "Unknown",
    industry: cleanText(raw.industry) || "Unknown",
    quote_type: cleanText(raw.quote_type),
    sector_source: cleanText(raw.sector_source),
    year: Math.round(year),
    beta_end_month: cleanText(raw.beta_end_month),
    beta_y1: parseNumber(raw.beta_y1),
    beta_y3: parseNumber(raw.beta_y3),
    beta_y5: parseNumber(raw.beta_y5),
    corr_sp500_y1: parseNumber(raw.corr_sp500_y1),
    corr_sp500_y3: parseNumber(raw.corr_sp500_y3),
    corr_sp500_y5: parseNumber(raw.corr_sp500_y5),
    stock_return_y1: parseNumber(raw.stock_return_y1),
    stock_return_y3: parseNumber(raw.stock_return_y3),
    stock_return_y5: parseNumber(raw.stock_return_y5),
    sp500_return_y1: parseNumber(raw.sp500_return_y1),
    sp500_return_y3: parseNumber(raw.sp500_return_y3),
    sp500_return_y5: parseNumber(raw.sp500_return_y5)
  };
}

function buildPart5StockBetaMap() {
  state.part5.stockBetaMap = new Map();
  for (const row of state.part5.stockBetaRows || []) {
    for (const ticker of part5TickerAliases(row.holding_ticker).concat(part5TickerAliases(row.yahoo_ticker))) {
      const key = part5BetaMapKey(ticker, row.year);
      if (key && !state.part5.stockBetaMap.has(key)) {
        state.part5.stockBetaMap.set(key, row);
      }
    }
  }
}

function part5TickerAliases(value) {
  const key = part5TickerKey(value);
  if (!key) return [];
  const aliases = new Set([key]);
  aliases.add(key.replace(/\./g, "-"));
  aliases.add(key.replace(/-/g, "."));
  return Array.from(aliases).filter(Boolean);
}

function part5TickerKey(value) {
  return cleanText(value).toUpperCase().replace(/\s+/g, "");
}

function part5BetaMapKey(tickerKey, year) {
  if (!tickerKey || !Number.isFinite(year)) return "";
  return `${tickerKey}|${Math.round(year)}`;
}

function lookupPart5StockBeta(row) {
  const years = [row.year, row.year - 1, row.year + 1].filter(Number.isFinite);
  const tickers = part5TickerAliases(row.holding_ticker);
  for (const year of years) {
    for (const ticker of tickers) {
      const betaRow = state.part5.stockBetaMap.get(part5BetaMapKey(ticker, year));
      if (betaRow) return betaRow;
    }
  }
  return null;
}

function enrichPart5HoldingWithBeta(row) {
  const betaRow = lookupPart5StockBeta(row);
  if (!betaRow) return row;
  row.hasStockBeta = true;
  row.yahoo_ticker = betaRow.yahoo_ticker;
  row.primary_security_name = betaRow.primary_security_name;
  row.sector = betaRow.sector || "Unknown";
  row.industry = betaRow.industry || "Unknown";
  row.betaYear = betaRow.year;
  row.stock_beta = betaRow.beta_y1;
  row.stock_beta_y1 = betaRow.beta_y1;
  row.stock_beta_y3 = betaRow.beta_y3;
  row.stock_beta_y5 = betaRow.beta_y5;
  row.corr_sp500_y1 = betaRow.corr_sp500_y1;
  row.stock_return_y1 = betaRow.stock_return_y1;
  row.sp500_return_y1 = betaRow.sp500_return_y1;
  row.weighted_beta = Number.isFinite(row.stock_beta) && Number.isFinite(row.holdingPct)
    ? row.stock_beta * row.holdingPct
    : NaN;
  row.beta_adjusted_holding_score = Number.isFinite(row.weighted_beta)
    ? Math.abs(row.weighted_beta)
    : NaN;
  return row;
}

function finalizePart5BetaMetrics() {
  const portfolioBetaByReport = new Map();
  for (const row of state.part5.holdings || []) {
    if (!Number.isFinite(row.weighted_beta)) continue;
    portfolioBetaByReport.set(row.reportKey, (portfolioBetaByReport.get(row.reportKey) || 0) + row.weighted_beta);
  }

  for (const row of state.part5.holdings || []) {
    const portfolioWeightedBeta = portfolioBetaByReport.get(row.reportKey);
    row.portfolio_weighted_beta = Number.isFinite(portfolioWeightedBeta) ? portfolioWeightedBeta : NaN;
    if (Number.isFinite(row.beta_adjusted_holding_score) && Number.isFinite(portfolioWeightedBeta) && Math.abs(portfolioWeightedBeta) > 1e-12) {
      row.beta_contribution_share = Math.abs(row.weighted_beta / portfolioWeightedBeta);
    } else {
      row.beta_contribution_share = NaN;
    }
  }

  for (const report of state.part5.reports || []) {
    const portfolioWeightedBeta = portfolioBetaByReport.get(report.reportKey);
    report.portfolio_weighted_beta = Number.isFinite(portfolioWeightedBeta) ? portfolioWeightedBeta : NaN;
  }
}

async function loadPart5ExcludedTwoGroupData() {
  const tasks = [
    ["excludedSummaryRows", PART5_EXCLUDED_SUMMARY_FILE, "Part5B two-group summary", normalizePart5ExcludedSummaryRow],
    ["excludedPanelRows", PART5_EXCLUDED_PANEL_FILE, "Part5B active-year panel", normalizePart5ExcludedPanelRow],
    ["excludedEnrichedRows", PART5_EXCLUDED_ENRICHED_FILE, "Part5B enriched items", normalizePart5ExcludedItemRow],
    ["excludedTopRows", PART5_EXCLUDED_TOP_ITEMS_FILE, "Part5B top items", normalizePart5ExcludedItemRow],
    ["excludedRemovedRows", PART5_EXCLUDED_REMOVED_FILE, "Part5B removed individual-stock audit", normalizePart5ExcludedItemRow]
  ];

  for (const [stateKey, file, label, normalizer] of tasks) {
    state.part5[stateKey] = await loadOptionalPart5Rows(file, label, normalizer);
  }
}

async function loadOptionalPart5Rows(file, label, normalizer) {
  const rows = [];
  let count = 0;
  try {
    await parseCsv(file, label, chunkRows => {
      for (const raw of chunkRows) {
        const row = normalizer(raw);
        if (!row) continue;
        rows.push(row);
        count += 1;
      }
      setPart5Status(`載入 ${label}：${formatInt(count)} 筆`);
    });
  } catch (error) {
    setPart5Status(`${label} 載入略過：${error.message || error}`);
  }
  return rows;
}

function normalizePart5ExcludedSummaryRow(raw) {
  const category = cleanText(raw.teacher_category);
  if (!category) return null;
  return {
    teacher_category: category,
    excluded_item_count: parseNumber(raw.excluded_item_count),
    holding_record_count: parseNumber(raw.holding_record_count),
    unique_portfolio_count: parseNumber(raw.unique_portfolio_count),
    high_confidence_items: parseNumber(raw.high_confidence_items),
    review_items: parseNumber(raw.review_items),
    first_report_dt: cleanText(raw.first_report_dt),
    last_report_dt: cleanText(raw.last_report_dt),
    firstReportMs: parseDateMs(raw.first_report_dt),
    lastReportMs: parseDateMs(raw.last_report_dt)
  };
}

function normalizePart5ExcludedPanelRow(raw) {
  const year = parseNumber(raw.year);
  const category = cleanText(raw.teacher_category);
  if (!Number.isFinite(year) || !category) return null;
  return {
    year: Math.round(year),
    teacher_category: category,
    active_item_count: parseNumber(raw.active_item_count),
    holding_record_count_proxy: parseNumber(raw.holding_record_count_proxy),
    unique_portfolio_count_proxy: parseNumber(raw.unique_portfolio_count_proxy),
    yield10y: state.part5.yieldYearMap.get(Math.round(year))
  };
}

function normalizePart5ExcludedItemRow(raw) {
  const category = cleanText(raw.teacher_category);
  const name = cleanText(raw.holding_security_name);
  const ticker = cleanText(raw.holding_ticker);
  if (!category || (!name && !ticker)) return null;
  const firstReportMs = parseDateMs(raw.first_report_dt);
  const lastReportMs = parseDateMs(raw.last_report_dt);
  return {
    teacher_category: category,
    teacher_subcategory: cleanText(raw.teacher_subcategory),
    exposure_dimension: cleanText(raw.exposure_dimension),
    holding_ticker: ticker,
    yahoo_ticker: cleanText(raw.yahoo_ticker),
    holding_security_name: name,
    item_label: ticker ? `${ticker} | ${name || ticker}` : name,
    exclude_reason: cleanText(raw.exclude_reason),
    holding_record_count: parseNumber(raw.holding_record_count),
    unique_portfolio_count: parseNumber(raw.unique_portfolio_count),
    first_report_dt: cleanText(raw.first_report_dt),
    last_report_dt: cleanText(raw.last_report_dt),
    firstReportMs,
    lastReportMs,
    firstYear: Number.isFinite(firstReportMs) ? datePartsFromMs(firstReportMs).year : NaN,
    lastYear: Number.isFinite(lastReportMs) ? datePartsFromMs(lastReportMs).year : NaN,
    classification_confidence: cleanText(raw.classification_confidence),
    interpretation: cleanText(raw.interpretation),
    is_company_universe_ticker: cleanText(raw.is_company_universe_ticker),
    is_individual_stock_like_removed: cleanText(raw.is_individual_stock_like_removed),
    use_in_part5b_two_group: cleanText(raw.use_in_part5b_two_group)
  };
}

function parsePart5StockSource(source) {
  let count = 0;
  return parseCsv(source.file, source.label, chunkRows => {
    for (const raw of chunkRows) {
      const row = normalizePart5StockRow(raw, source);
      if (!row) continue;

      state.part5.holdings.push(row);
      count += 1;

      if (!state.part5.reportKeys.has(row.reportKey)) {
        state.part5.reportKeys.add(row.reportKey);
        const report = {
          reportKey: row.reportKey,
          periodKey: row.periodKey,
          periodLabel: row.periodLabel,
          crsp_portno: row.crsp_portno,
          fund_ticker: row.fund_ticker,
          fund_name: row.fund_name,
          report_dt: row.report_dt,
          reportDateMs: row.reportDateMs,
          year: row.year,
          quarter: row.quarter,
          monthKey: row.monthKey,
          stockPct: row.stockPct,
          bondPct: row.bondPct,
          cashPct: row.cashPct,
          yield10y: row.yield10y
        };
        state.part5.reports.push(report);
        state.part5.reportMap.set(row.reportKey, report);
      } else {
        mergePart5ReportFromHolding(state.part5.reportMap.get(row.reportKey), row);
      }
    }
    setPart5Status(`載入 ${source.label} 持股資料：${formatInt(count)} 筆`);
  });
}

function mergePart5ReportFromHolding(report, row) {
  if (!report) return;
  if (!Number.isFinite(report.stockPct) && Number.isFinite(row.stockPct)) report.stockPct = row.stockPct;
  if (!Number.isFinite(report.bondPct) && Number.isFinite(row.bondPct)) report.bondPct = row.bondPct;
  if (!Number.isFinite(report.cashPct) && Number.isFinite(row.cashPct)) report.cashPct = row.cashPct;
  if (!Number.isFinite(report.yield10y) && Number.isFinite(row.yield10y)) report.yield10y = row.yield10y;
}

function normalizePart5StockRow(raw, source) {
  const reportDateMs = parseDateMs(raw.report_dt);
  if (!Number.isFinite(reportDateMs)) return null;

  const parts = datePartsFromMs(reportDateMs);
  const monthKey = monthKeyFromMs(reportDateMs);
  const quarter = Math.floor((parts.month - 1) / 3) + 1;
  const yield10y = state.part5.yieldMonthMap.get(monthKey) ?? state.part5.yieldYearMap.get(parts.year);
  const rank = parseNumber(raw.security_rank);

  const row = {
    periodKey: source.key,
    periodLabel: source.label,
    crsp_portno: cleanText(raw.crsp_portno),
    fund_ticker: cleanText(raw.fund_ticker),
    fund_name: cleanText(raw.fund_name),
    lipper_obj_name: cleanText(raw.lipper_obj_name),
    stockPct: parsePercentValue(raw.fund_percent_common_stock),
    bondPct: parsePercentValue(raw.fund_percent_bond),
    cashPct: parsePercentValue(raw.fund_percent_cash),
    report_dt: isoDateFromMs(reportDateMs),
    reportDateMs,
    monthKey,
    year: parts.year,
    quarter,
    reportKey: `${cleanText(raw.crsp_portno)}|${isoDateFromMs(reportDateMs)}`,
    security_rank: Number.isFinite(rank) ? rank : NaN,
    holdingPct: parsePercentValue(raw.holding_percent_tna),
    holding_market_val: parseNumber(raw.holding_market_val),
    crsp_company_key: cleanText(raw.crsp_company_key),
    holding_security_name: cleanText(raw.holding_security_name),
    holding_ticker: cleanText(raw.holding_ticker),
    holding_permno: cleanText(raw.holding_permno),
    sector: "Unknown",
    industry: "Unknown",
    yield10y
  };

  return enrichPart5HoldingWithBeta(row);
}


function defaultPart5ModeMemory() {
  const base = () => ({
    period: "all",
    aggregation: "year",
    focus: "all",
    rank: "10",
    limit: "250",
    search: "",
    activeReportKey: "",
    activeHoldingKey: "",
    brushedPeriodLabels: [],
    brushedReportKeys: [],
    brushedHoldingKeys: [],
    selectedManagerNames: [],
    part5BSelectedCategories: [],
    part5BSelectedYears: [],
    part5BSelectedItemKeys: []
  });
  return { all: base(), manager: base() };
}

function ensurePart5ModeMemory() {
  if (!state.part5.modeMemory) state.part5.modeMemory = defaultPart5ModeMemory();
  if (!state.part5.modeMemory.all) state.part5.modeMemory.all = defaultPart5ModeMemory().all;
  if (!state.part5.modeMemory.manager) state.part5.modeMemory.manager = defaultPart5ModeMemory().manager;
  return state.part5.modeMemory;
}

function capturePart5ModeState() {
  return {
    period: dom.part5PeriodSelect ? dom.part5PeriodSelect.value : "all",
    aggregation: dom.part5AggregationSelect ? dom.part5AggregationSelect.value : "year",
    focus: dom.part5FocusSelect ? dom.part5FocusSelect.value : "all",
    rank: dom.part5RankInput ? dom.part5RankInput.value : "10",
    limit: dom.part5LimitSelect ? dom.part5LimitSelect.value : "250",
    search: dom.part5SearchInput ? dom.part5SearchInput.value : "",
    activeReportKey: state.part5.activeReportKey || "",
    activeHoldingKey: state.part5.activeHoldingKey || "",
    brushedPeriodLabels: [...(state.part5.brushedPeriodLabels || [])],
    brushedReportKeys: [...(state.part5.brushedReportKeys || [])],
    brushedHoldingKeys: [...(state.part5.brushedHoldingKeys || [])],
    selectedManagerNames: [...(state.part5.selectedManagerNames || [])],
    part5BSelectedCategories: [...(state.part5.part5BSelectedCategories || [])],
    part5BSelectedYears: [...(state.part5.part5BSelectedYears || [])],
    part5BSelectedItemKeys: [...(state.part5.part5BSelectedItemKeys || [])]
  };
}

function restorePart5ModeState(snapshot) {
  const data = snapshot || defaultPart5ModeMemory().all;
  if (dom.part5PeriodSelect) dom.part5PeriodSelect.value = data.period || "all";
  if (dom.part5AggregationSelect) dom.part5AggregationSelect.value = data.aggregation || "year";
  if (dom.part5FocusSelect) dom.part5FocusSelect.value = data.focus || "all";
  if (dom.part5RankInput) dom.part5RankInput.value = data.rank || "10";
  if (dom.part5LimitSelect) dom.part5LimitSelect.value = data.limit || "250";
  if (dom.part5SearchInput) dom.part5SearchInput.value = data.search || "";
  state.part5.activeReportKey = data.activeReportKey || "";
  state.part5.activeHoldingKey = data.activeHoldingKey || "";
  state.part5.brushedPeriodLabels = [...(data.brushedPeriodLabels || [])];
  state.part5.brushedReportKeys = [...(data.brushedReportKeys || [])];
  state.part5.brushedHoldingKeys = [...(data.brushedHoldingKeys || [])];
  state.part5.selectedManagerNames = [...(data.selectedManagerNames || [])];
  state.part5.part5BSelectedCategories = [...(data.part5BSelectedCategories || [])];
  state.part5.part5BSelectedYears = [...(data.part5BSelectedYears || [])];
  state.part5.part5BSelectedItemKeys = [...(data.part5BSelectedItemKeys || [])];
}

function switchPart5AnalysisMode(nextMode) {
  const mode = nextMode === "manager" ? "manager" : "all";
  const memory = ensurePart5ModeMemory();
  const prevMode = state.part5.analysisMode === "manager" ? "manager" : "all";
  memory[prevMode] = capturePart5ModeState();
  state.part5.analysisMode = mode;
  if (dom.part5AnalysisModeSelect) dom.part5AnalysisModeSelect.value = mode;
  restorePart5ModeState(memory[mode]);
  state.part5.managerFilteredCache = null;
  renderPart5();
  renderPart6();
}

function renderPart5() {
  if (!state.part5.loaded) {
    if (dom.part5Results) dom.part5Results.classList.add("hidden");
    setPart5Status(state.part5.loading ? "Part5 資料載入中..." : "尚未載入 Part5 資料。");
    return;
  }

  if (dom.part5Results) dom.part5Results.classList.remove("hidden");
  state.part5.analysisMode = dom.part5AnalysisModeSelect ? (dom.part5AnalysisModeSelect.value || "all") : (state.part5.analysisMode || "all");

  const baseReports = filterPart5Reports({ useFocus: false, useBrush: false });
  populatePart5FocusSelect(baseReports);

  const reports = filterPart5Reports({ useFocus: true, useBrush: true });
  const holdingsForSummary = filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true });
  const detailRows = filterPart5Holdings({ useDetailFilters: true, useFocus: true, useBrush: true });
  syncPart5Selection(reports, holdingsForSummary);

  updatePart5Metrics(reports, detailRows);
  drawPart5Charts(reports, detailRows);
  renderPart5SummaryTable(reports, holdingsForSummary);
  renderPart5HoldingsTable(detailRows);
  renderPart5ReportDetail(reports);
  renderPart5TopHoldingDetail(holdingsForSummary);
  renderPart5FastPickers(reports, detailRows, holdingsForSummary);
  renderPart5ManagerPanel(reports, holdingsForSummary);
  renderPart5BExcludedInsights();
  updatePart5BrushStatus(reports, detailRows, holdingsForSummary);

  const modeLabel = state.part5.analysisMode === "manager" ? "Part4 經理人模式" : "全部基金模式";
  const brushText = state.part5.brushedPeriodLabels.length ? `，已套用 ${formatInt(state.part5.brushedPeriodLabels.length)} 個年/季框選` : "";
  setPart5Status(`${modeLabel}${brushText}：目前顯示 ${formatInt(reports.length)} 個基金報告、${formatInt(detailRows.length)} 筆排名內持股明細。`);
  renderPart6();
}

function isPart5ManagerMode() {
  return (dom.part5AnalysisModeSelect ? dom.part5AnalysisModeSelect.value : state.part5.analysisMode) === "manager";
}

function selectedPart4Managers() {
  const names = new Set();
  for (const record of state.radarRecords || []) {
    if (record && record.manager) names.add(String(record.manager));
  }
  return Array.from(names).sort();
}

function activePart5Managers() {
  const allManagers = selectedPart4Managers();
  const selected = (state.part5.selectedManagerNames || []).filter(manager => allManagers.includes(manager));
  return selected.length ? selected.slice().sort() : allManagers;
}

function part5ManagerSelectionSet() {
  const selected = state.part5.selectedManagerNames || [];
  return new Set(selected.filter(Boolean));
}

function normalizeLooseText(value) {
  return cleanText(value).toLowerCase().replace(/[^a-z0-9]+/g, " ").replace(/\s+/g, " ").trim();
}

function part5ManagerFundNameSet() {
  return getPart5ManagerMatchCache().fundNames;
}

function part5ReportMatchesManager(report, managerFundNames) {
  if (!managerFundNames || !managerFundNames.size) return false;
  const reportName = normalizeLooseText(report.fund_name);
  if (!reportName) return false;
  for (const fundName of managerFundNames) {
    if (!fundName) continue;
    if (reportName === fundName || reportName.includes(fundName) || fundName.includes(reportName)) return true;
  }
  return false;
}

function part5ManagerCacheSignature(managers) {
  return [
    managers.join("||"),
    state.rows ? state.rows.length : 0,
    state.part5.reports ? state.part5.reports.length : 0,
    state.part5.holdings ? state.part5.holdings.length : 0
  ].join("::");
}

function getPart5ManagerMatchCache() {
  const managers = selectedPart4Managers();
  const signature = part5ManagerCacheSignature(managers);
  if (state.part5.managerMatchCache && state.part5.managerMatchCache.signature === signature) {
    return state.part5.managerMatchCache;
  }

  const managerSet = new Set(managers);
  const managerNormSet = new Set(managers.map(normalizeLooseText));
  const managerToFundNames = new Map();
  const managerToFundIds = new Map();
  const managerFundSamples = new Map();
  const fundNames = new Set();
  const fundNameToManagers = new Map();

  for (const manager of managers) {
    managerToFundNames.set(manager, new Set());
    managerToFundIds.set(manager, new Set());
    managerFundSamples.set(manager, []);
  }

  for (const row of state.rows || []) {
    const manager = cleanText(row.mgr_name);
    const managerNorm = normalizeLooseText(manager);
    if (!managerNormSet.has(managerNorm)) continue;
    const canonicalManager = managerSet.has(manager)
      ? manager
      : managers.find(item => normalizeLooseText(item) === managerNorm);
    if (!canonicalManager) continue;

    const fundName = cleanText(row.fund_name);
    const fundNameNorm = normalizeLooseText(fundName);
    if (fundNameNorm) {
      managerToFundNames.get(canonicalManager).add(fundNameNorm);
      fundNames.add(fundNameNorm);
      if (!fundNameToManagers.has(fundNameNorm)) fundNameToManagers.set(fundNameNorm, new Set());
      fundNameToManagers.get(fundNameNorm).add(canonicalManager);
      const samples = managerFundSamples.get(canonicalManager);
      if (fundName && samples.length < 8 && !samples.includes(fundName)) samples.push(fundName);
    }
    const fundId = cleanText(row.crsp_fundno);
    if (fundId) managerToFundIds.get(canonicalManager).add(fundId);
  }

  const managerToReportKeys = new Map(managers.map(manager => [manager, new Set()]));
  const reportKeyToManagers = new Map();
  const reportKeys = new Set();
  const normalizedReportNames = new Map();
  const fundNameList = Array.from(fundNames);

  function addReportMatch(reportKey, manager) {
    if (!managerToReportKeys.has(manager)) return;
    managerToReportKeys.get(manager).add(reportKey);
    reportKeys.add(reportKey);
    if (!reportKeyToManagers.has(reportKey)) reportKeyToManagers.set(reportKey, new Set());
    reportKeyToManagers.get(reportKey).add(manager);
  }

  for (const report of state.part5.reports || []) {
    const reportName = normalizeLooseText(report.fund_name);
    if (!reportName) continue;
    normalizedReportNames.set(report.reportKey, reportName);

    const exactManagers = fundNameToManagers.get(reportName);
    if (exactManagers && exactManagers.size) {
      for (const manager of exactManagers) addReportMatch(report.reportKey, manager);
      continue;
    }

    for (const fundName of fundNameList) {
      if (!fundName) continue;
      if (reportName.includes(fundName) || fundName.includes(reportName)) {
        const matchedManagers = fundNameToManagers.get(fundName);
        if (!matchedManagers) continue;
        for (const manager of matchedManagers) addReportMatch(report.reportKey, manager);
      }
    }
  }

  const reports = [];
  for (const report of state.part5.reports || []) {
    if (reportKeys.has(report.reportKey)) reports.push(report);
  }
  reports.sort((a, b) => a.reportDateMs - b.reportDateMs);

  const holdings = [];
  const holdingsByReportKey = new Map();
  for (const holding of state.part5.holdings || []) {
    if (!reportKeys.has(holding.reportKey)) continue;
    holdings.push(holding);
    if (!holdingsByReportKey.has(holding.reportKey)) holdingsByReportKey.set(holding.reportKey, []);
    holdingsByReportKey.get(holding.reportKey).push(holding);
  }

  const cache = {
    signature,
    managers,
    fundNames,
    managerToFundNames,
    managerToFundIds,
    managerFundSamples,
    managerToReportKeys,
    reportKeyToManagers,
    reportKeys,
    reports,
    holdings,
    holdingsByReportKey,
    normalizedReportNames
  };
  state.part5.managerMatchCache = cache;
  return cache;
}

function getPart5ManagerWorkingSet() {
  const cache = getPart5ManagerMatchCache();
  const selectedManagers = activePart5Managers();
  const signature = [cache.signature, selectedManagers.join("||")].join("::active=");
  if (state.part5.managerFilteredCache && state.part5.managerFilteredCache.signature === signature) {
    return state.part5.managerFilteredCache;
  }

  const activeManagerSet = new Set(selectedManagers);
  const reportKeys = new Set();
  for (const manager of selectedManagers) {
    const keys = cache.managerToReportKeys.get(manager) || new Set();
    for (const key of keys) reportKeys.add(key);
  }

  const reports = [];
  for (const report of cache.reports || []) {
    if (reportKeys.has(report.reportKey)) reports.push(report);
  }

  const holdings = [];
  for (const reportKey of reportKeys) {
    const list = cache.holdingsByReportKey && cache.holdingsByReportKey.get(reportKey);
    if (list && list.length) holdings.push(...list);
  }

  const working = {
    signature,
    managers: selectedManagers,
    activeManagerSet,
    reportKeys,
    reports,
    holdings
  };
  state.part5.managerFilteredCache = working;
  return working;
}

function managerMatchedReportKeySet() {
  return getPart5ManagerWorkingSet().reportKeys;
}

function filterPart5Reports(options = {}) {
  const period = dom.part5PeriodSelect.value;
  const useFocus = options.useFocus !== false;
  const useBrush = options.useBrush !== false;
  const focusKey = useFocus ? dom.part5FocusSelect.value : "all";
  const mode = dom.part5AggregationSelect.value;
  const managerCache = isPart5ManagerMode() && options.useManager !== false ? getPart5ManagerWorkingSet() : null;
  const sourceReports = managerCache ? managerCache.reports : state.part5.reports;
  const brushLabels = useBrush ? new Set(state.part5.brushedPeriodLabels || []) : null;

  return sourceReports
    .filter(row => {
      if (period !== "all" && row.periodKey !== period) return false;
      if (focusKey !== "all" && part5AggregateLabel(row, mode) !== focusKey) return false;
      if (brushLabels && brushLabels.size && !brushLabels.has(part5AggregateLabel(row, mode))) return false;
      return true;
    })
    .sort((a, b) => a.reportDateMs - b.reportDateMs);
}

function filterPart5Holdings(options = {}) {
  const period = dom.part5PeriodSelect.value;
  const useDetailFilters = options.useDetailFilters !== false;
  const useFocus = options.useFocus !== false;
  const useBrush = options.useBrush !== false;
  const focusKey = useFocus ? dom.part5FocusSelect.value : "all";
  const mode = dom.part5AggregationSelect.value;
  const rankMax = parseNumber(dom.part5RankInput.value);
  const query = cleanText(dom.part5SearchInput.value).toLowerCase();
  const managerCache = isPart5ManagerMode() && options.useManager !== false ? getPart5ManagerWorkingSet() : null;
  const sourceHoldings = managerCache ? managerCache.holdings : state.part5.holdings;
  const brushLabels = useBrush ? new Set(state.part5.brushedPeriodLabels || []) : null;

  return sourceHoldings.filter(row => {
    if (period !== "all" && row.periodKey !== period) return false;
    if (focusKey !== "all" && part5AggregateLabel(row, mode) !== focusKey) return false;
    if (brushLabels && brushLabels.size && !brushLabels.has(part5AggregateLabel(row, mode))) return false;
    if (!useDetailFilters) return true;
    if (Number.isFinite(rankMax)) {
      if (!Number.isFinite(row.security_rank) || row.security_rank <= 0 || row.security_rank > rankMax) return false;
    }
    if (query) {
      const haystack = [
        row.crsp_portno,
        row.fund_ticker,
        row.fund_name,
        row.holding_ticker,
        row.holding_security_name,
        row.lipper_obj_name,
        row.sector,
        row.industry
      ].join(" ").toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  }).sort((a, b) => {
    if (a.reportDateMs !== b.reportDateMs) return b.reportDateMs - a.reportDateMs;
    if (a.crsp_portno !== b.crsp_portno) return String(a.crsp_portno).localeCompare(String(b.crsp_portno));
    return (a.security_rank || 999) - (b.security_rank || 999);
  });
}

function populatePart5FocusSelect(reports) {
  const current = dom.part5FocusSelect.value || "all";
  const mode = dom.part5AggregationSelect.value;
  const options = aggregatePart5Reports(reports, mode).map(row => row.label);

  dom.part5FocusSelect.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = mode === "quarter" ? "全部季度" : "全部年份";
  dom.part5FocusSelect.appendChild(allOption);

  for (const label of options) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    dom.part5FocusSelect.appendChild(option);
  }

  dom.part5FocusSelect.value = options.includes(current) ? current : "all";
}

function updatePart5Metrics(reports, detailRows) {
  dom.metricP5Reports.textContent = formatInt(reports.length);
  dom.metricP5Funds.textContent = formatInt(uniqueCount(reports, row => row.crsp_portno));
  dom.metricP5Holdings.textContent = formatInt(detailRows.length);
  dom.metricP5Yield.textContent = formatYield(mean(reports.map(row => row.yield10y)));
}

function drawPart5Charts(reports, detailRows) {
  if (!window.Plotly) return;

  const trendSourceReports = filterPart5Reports({ useFocus: true, useBrush: false });
  const trendRows = aggregatePart5Reports(trendSourceReports, dom.part5AggregationSelect.value);
  if (!trendRows.length) {
    Plotly.purge("part5OverviewChart");
    Plotly.purge("part5AllocationChart");
    Plotly.purge("part5TopHoldingsChart");
    Plotly.purge("part5ReportScatterChart");
    Plotly.purge("part5ReportHoldingsChart");
    if (dom.part5ReportDetailTable) dom.part5ReportDetailTable.innerHTML = "";
    if (dom.part5HoldingExplainTable) dom.part5HoldingExplainTable.innerHTML = "";
    if (dom.part5TopHoldingDetailTable) dom.part5TopHoldingDetailTable.innerHTML = "";
    if (dom.part5ReportPickerTable) dom.part5ReportPickerTable.innerHTML = "";
    if (dom.part5TopHoldingPickerTable) dom.part5TopHoldingPickerTable.innerHTML = "";
    return;
  }

  const x = trendRows.map(row => row.label);
  const stock = trendRows.map(row => row.stockPct);
  const bond = trendRows.map(row => row.bondPct);
  const cash = trendRows.map(row => row.cashPct);
  const stackStock = trendRows.map(row => row.stackStockPct);
  const stackBond = trendRows.map(row => row.stackBondPct);
  const stackCash = trendRows.map(row => row.stackCashPct);
  const stockBondGap = trendRows.map(row => row.stockBondGap);
  const yields = trendRows.map(row => row.yield10y);
  const selectedSet = new Set(state.part5.brushedPeriodLabels || []);

  const selectedShape = selectedSet.size ? [{
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: x.find(label => selectedSet.has(label)) || x[0],
    x1: [...x].reverse().find(label => selectedSet.has(label)) || x[x.length - 1],
    y0: 0,
    y1: 1,
    fillcolor: "rgba(31,111,178,0.08)",
    line: { color: "rgba(31,111,178,0.35)", width: 1 }
  }] : [];

  Plotly.react("part5OverviewChart", [
    { type: "scatter", mode: "lines+markers", x, y: stock, name: "平均股票比例", line: { color: "#1677c2", width: 2.4 }, marker: { size: 6 }, hovertemplate: "%{x}<br>股票：%{y:.2%}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x, y: bond, name: "平均債券比例", line: { color: "#12a59a", width: 2.4 }, marker: { size: 6 }, hovertemplate: "%{x}<br>債券：%{y:.2%}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x, y: stockBondGap, name: "股債差", line: { color: "#df6b57", width: 2.1, dash: "dot" }, marker: { size: 6 }, hovertemplate: "%{x}<br>股票-債券：%{y:.2%}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x, y: yields, name: "10 年期殖利率", yaxis: "y2", line: { color: "#d88c18", width: 2.6 }, marker: { size: 7 }, hovertemplate: "%{x}<br>殖利率：%{y:.2f}%<extra></extra>" }
  ], {
    title: "Part 5：10 年期殖利率與股債配置趨勢（可框選年/季影響下方）",
    height: 430,
    margin: { l: 58, r: 62, t: 62, b: 58 },
    dragmode: "select",
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.22 },
    shapes: selectedShape,
    uirevision: "part5-trend-nozoom",
    xaxis: { title: dom.part5AggregationSelect.value === "quarter" ? "季度" : "年份", automargin: true },
    yaxis: { title: "配置比例 / 股債差", tickformat: ".0%" },
    yaxis2: { title: "10 年期殖利率", overlaying: "y", side: "right", ticksuffix: "%", showgrid: false }
  }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
  attachPart5ChartDrilldown("part5OverviewChart");

  Plotly.react("part5AllocationChart", [
    { type: "bar", x, y: stackStock, name: "股票", marker: { color: "#1677c2" }, hovertemplate: "%{x}<br>股票：%{y:.2%}<extra></extra>" },
    { type: "bar", x, y: stackBond, name: "債券（含補足）", marker: { color: "#12a59a" }, hovertemplate: "%{x}<br>債券含補足：%{y:.2%}<extra></extra>" },
    { type: "bar", x, y: stackCash, name: "現金", marker: { color: "#d88c18" }, hovertemplate: "%{x}<br>現金：%{y:.2%}<extra></extra>" }
  ], {
    title: "Part 5：基金股 / 債 / 現金配置堆疊（不足 100% 由債券補足）",
    height: 390,
    margin: { l: 58, r: 28, t: 58, b: 58 },
    dragmode: "select",
    barmode: "stack",
    legend: { orientation: "h", y: -0.22 },
    shapes: selectedShape,
    uirevision: "part5-allocation-nozoom",
    xaxis: { automargin: true },
    yaxis: { title: "配置比例", tickformat: ".0%", range: [0, 1] }
  }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
  attachPart5ChartDrilldown("part5AllocationChart");

  drawPart5TopHoldingsChart(detailRows);
  drawPart5ReportScatter(reports, detailRows);
}

function aggregatePart5Reports(reports, mode) {
  const map = new Map();
  for (const report of reports) {
    const key = part5AggregateLabel(report, mode);
    const sortKey = part5AggregateSortKey(report, mode);
    if (!map.has(key)) {
      map.set(key, { label: key, sortKey, stockValues: [], bondValues: [], cashValues: [], stackStockValues: [], stackBondValues: [], stackCashValues: [], yieldValues: [] });
    }
    const row = map.get(key);
    if (Number.isFinite(report.stockPct)) row.stockValues.push(report.stockPct);
    if (Number.isFinite(report.bondPct)) row.bondValues.push(report.bondPct);
    if (Number.isFinite(report.cashPct)) row.cashValues.push(report.cashPct);
    const completed = completedPart5Allocation(report);
    if (completed) {
      row.stackStockValues.push(completed.stockPct);
      row.stackBondValues.push(completed.bondPct);
      row.stackCashValues.push(completed.cashPct);
    }
    if (Number.isFinite(report.yield10y)) row.yieldValues.push(report.yield10y);
  }
  return Array.from(map.values()).sort((a, b) => a.sortKey - b.sortKey).map(row => ({
    label: row.label,
    stockPct: mean(row.stockValues),
    bondPct: mean(row.bondValues),
    cashPct: mean(row.cashValues),
    stackStockPct: mean(row.stackStockValues),
    stackBondPct: mean(row.stackBondValues),
    stackCashPct: mean(row.stackCashValues),
    stockBondGap: mean(row.stockValues) - mean(row.bondValues),
    yield10y: mean(row.yieldValues)
  }));
}

function completedPart5Allocation(row) {
  const stockPct = Number.isFinite(row.stockPct) ? Math.max(0, row.stockPct) : 0;
  let bondPct = Number.isFinite(row.bondPct) ? Math.max(0, row.bondPct) : 0;
  const cashPct = Number.isFinite(row.cashPct) ? Math.max(0, row.cashPct) : 0;
  const total = stockPct + bondPct + cashPct;
  if (total <= 0) return null;
  if (total < 1) bondPct += 1 - total;
  return { stockPct, bondPct, cashPct };
}

function part5AggregateLabel(row, mode) {
  return mode === "quarter" ? `${row.year} Q${row.quarter}` : String(row.year);
}

function part5AggregateSortKey(row, mode) {
  return mode === "quarter" ? row.year * 10 + row.quarter : row.year;
}

function attachPart5ChartDrilldown(chartId) {
  const plot = document.getElementById(chartId);
  if (!plot) return;
  resetPlotlyHandler(plot, "plotly_selected", eventData => {
    const labels = part5SelectedXLabels(eventData);
    if (!labels.length) return;
    state.part5.brushedPeriodLabels = labels;
    state.part5.activeReportKey = "";
    state.part5.activeHoldingKey = "";
    state.part5.brushedReportKeys = [];
    state.part5.brushedHoldingKeys = [];
    renderPart5();
  });
  resetPlotlyHandler(plot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    if (!point || point.x == null) return;
    state.part5.brushedPeriodLabels = [String(point.x)];
    state.part5.activeReportKey = "";
    state.part5.activeHoldingKey = "";
    renderPart5();
  });
}

function part5SelectedXLabels(eventData) {
  if (!eventData || !eventData.points || !eventData.points.length) return [];
  const labels = [];
  const seen = new Set();
  for (const point of eventData.points) {
    if (point.x == null) continue;
    const label = String(point.x);
    if (!seen.has(label)) {
      seen.add(label);
      labels.push(label);
    }
  }
  return labels;
}

function part5SelectedCustomData(eventData) {
  if (!eventData || !eventData.points || !eventData.points.length) return [];
  const out = [];
  const seen = new Set();
  for (const point of eventData.points) {
    if (point.customdata == null) continue;
    const value = Array.isArray(point.customdata) ? point.customdata[0] : point.customdata;
    const key = String(value);
    if (!seen.has(key)) {
      seen.add(key);
      out.push(key);
    }
  }
  return out;
}

function part5SelectedManagerNames(eventData) {
  return part5SelectedCustomData(eventData).filter(Boolean).sort();
}

function part5SectorColor(sector, index = 0) {
  const known = [
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Industrials", "Communication Services", "Consumer Defensive", "Energy",
    "Basic Materials", "Real Estate", "Utilities", "Unknown"
  ];
  const palette = ["#1677c2", "#12a59a", "#d88c18", "#d13f31", "#8054b8", "#008c95", "#bf4b75", "#2d8a63", "#b58b00", "#6b7480", "#4959b8", "#8b949e"];
  const sectorIndex = known.indexOf(sector);
  return palette[(sectorIndex >= 0 ? sectorIndex : index) % palette.length];
}

function drawPart5ReportScatter(reports, detailRows = []) {
  if (!window.Plotly) return;
  const reportKeySet = new Set((reports || []).map(row => row.reportKey));
  const rows = (detailRows || []).filter(row =>
    reportKeySet.has(row.reportKey) &&
    row.hasStockBeta &&
    Number.isFinite(row.stockPct) &&
    Number.isFinite(row.holdingPct)
  );
  if (!rows.length) {
    Plotly.purge("part5ReportScatterChart");
    return;
  }

  const selectedKeys = new Set(state.part5.brushedReportKeys && state.part5.brushedReportKeys.length ? state.part5.brushedReportKeys : (state.part5.activeReportKey ? [state.part5.activeReportKey] : []));
  const sectors = Array.from(new Set(rows.map(row => row.sector || "Unknown"))).sort();
  const traces = sectors.map((sector, index) => {
    const sectorRows = rows.filter(row => (row.sector || "Unknown") === sector);
    return {
      type: "scattergl",
      mode: "markers",
      name: sector,
      x: sectorRows.map(row => row.stockPct),
      y: sectorRows.map(row => row.holdingPct),
      customdata: sectorRows.map(row => [row.reportKey, row.holding_ticker || row.holding_security_name || row.crsp_company_key, row.fund_ticker || row.crsp_portno, row.holding_security_name, row.stock_beta, row.weighted_beta, row.sector]),
      marker: {
        size: sectorRows.map(row => selectedKeys.has(row.reportKey) ? 16 : Math.max(6, Math.min(18, 6 + row.holdingPct * 120))),
        color: part5SectorColor(sector, index),
        opacity: sectorRows.map(row => selectedKeys.size && !selectedKeys.has(row.reportKey) ? 0.2 : 0.72),
        line: { color: sectorRows.map(row => selectedKeys.has(row.reportKey) ? "#d13f31" : "rgba(20,45,64,0.28)"), width: sectorRows.map(row => selectedKeys.has(row.reportKey) ? 3 : 0.5) }
      },
      hovertemplate: "Sector：%{customdata[6]}<br>基金：%{customdata[2]}<br>持股：%{customdata[1]} %{customdata[3]}<br>基金股票配置：%{x:.2%}<br>持股占TNA：%{y:.2%}<br>stock_beta：%{customdata[4]:.4f}<br>weighted_beta：%{customdata[5]:.6f}<extra></extra>"
    };
  });

  Plotly.react("part5ReportScatterChart", traces, {
    title: `Part 5：基金報告配置散點圖（${formatInt(rows.length)} 筆 beta 個股持股；可框選或點選）`,
    height: 430,
    margin: { l: 62, r: 28, t: 62, b: 58 },
    dragmode: "select",
    xaxis: { title: "基金股票配置", tickformat: ".0%", zeroline: false },
    yaxis: { title: "個股持股占 TNA", tickformat: ".0%", zeroline: false },
    hovermode: "closest",
    legend: { orientation: "h", y: -0.22 }
  }, { displaylogo: false, responsive: true, scrollZoom: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines"] });

  const plot = document.getElementById("part5ReportScatterChart");
  resetPlotlyHandler(plot, "plotly_selected", eventData => {
    const keys = part5SelectedCustomData(eventData);
    if (!keys.length) return;
    state.part5.brushedReportKeys = keys;
    state.part5.activeReportKey = keys[0];
    const selectedReports = part5FilterReportsByKeys(reports, keys);
    const allHoldingRows = filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true });
    renderPart5ReportDetail(reports);
    renderPart5ReportPicker(selectedReports, detailRows);
    renderPart5TopHoldingPicker(reports, detailRows, allHoldingRows);
    updatePart5BrushStatus(reports, detailRows, allHoldingRows);
    setPart5Status(`已框選 ${formatInt(keys.length)} 個基金報告，基金報告快速選取已切換成框選結果。`);
  });
  resetPlotlyHandler(plot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    if (!point || point.customdata == null) return;
    state.part5.brushedReportKeys = [];
    state.part5.activeReportKey = String(Array.isArray(point.customdata) ? point.customdata[0] : point.customdata);
    const selectedReports = part5FilterReportsByKeys(reports, [state.part5.activeReportKey]);
    renderPart5ReportDetail(reports);
    renderPart5ReportPicker(selectedReports, detailRows);
    setPart5Status("已點選基金報告，基金報告快速選取已切換成該基金報告；散點圖仍保留目前範圍的個股持股點。");
  });
}

function renderPart5ReportDetail(reports) {
  if (!dom.part5ReportDetailStatus || !dom.part5ReportDetailTable) return;
  const selectedKeys = state.part5.brushedReportKeys && state.part5.brushedReportKeys.length
    ? new Set(state.part5.brushedReportKeys)
    : (state.part5.activeReportKey ? new Set([state.part5.activeReportKey]) : new Set());
  const pickedReports = (reports || []).filter(row => selectedKeys.has(row.reportKey));
  if (!pickedReports.length) {
    dom.part5ReportDetailStatus.textContent = "可在基金報告散點圖框選一群基金報告，或用快速選取表格選單一基金報告";
    dom.part5ReportDetailTable.innerHTML = "";
    if (dom.part5HoldingExplainTable) dom.part5HoldingExplainTable.innerHTML = "";
    if (window.Plotly) Plotly.purge("part5ReportHoldingsChart");
    return;
  }
  const keySet = new Set(pickedReports.map(row => row.reportKey));
  const holdings = state.part5.holdings.filter(row => keySet.has(row.reportKey)).sort((a, b) => b.reportDateMs - a.reportDateMs || (a.security_rank || 999) - (b.security_rank || 999));
  dom.part5ReportDetailStatus.textContent = pickedReports.length === 1
    ? `已鎖定 ${pickedReports[0].fund_ticker || pickedReports[0].crsp_portno} / ${pickedReports[0].report_dt} / ${formatInt(holdings.length)} 筆持股`
    : `已框選 ${formatInt(pickedReports.length)} 個基金報告 / ${formatInt(holdings.length)} 筆持股`;
  renderTable(dom.part5ReportDetailTable, pickedReports.slice(0, 200), [
    { key: "periodLabel", label: "期間" }, { key: "report_dt", label: "報告日" }, { key: "crsp_portno", label: "Portno" }, { key: "fund_ticker", label: "基金Ticker" }, { key: "fund_name", label: "基金名稱" }, { key: "stockPct", label: "股票", format: "pct" }, { key: "bondPct", label: "債券", format: "pct" }, { key: "cashPct", label: "現金", format: "pct" }, { key: "portfolio_weighted_beta", label: "portfolio_weighted_beta", format: "num" }, { key: "yield10y", label: "10年殖利率", format: "yield" }
  ], { title: pickedReports.length === 1 ? "被選取的基金報告" : "框選基金報告集合", expanded: true, count: pickedReports.length });
  drawPart5ReportHoldingsChart(holdings);
  renderPart5HoldingExplainTable(holdings);
}

function drawPart5ReportHoldingsChart(rows) {
  if (!window.Plotly) return;
  const ranked = (rows || []).filter(row => Number.isFinite(row.holdingPct) || Number.isFinite(row.holding_market_val)).slice(0, 15).reverse();
  if (!ranked.length) { Plotly.purge("part5ReportHoldingsChart"); return; }
  Plotly.react("part5ReportHoldingsChart", [{
    type: "bar", orientation: "h",
    x: ranked.map(row => Number.isFinite(row.holdingPct) ? row.holdingPct : 0),
    y: ranked.map(row => row.holding_ticker ? `${row.holding_ticker} | ${row.holding_security_name || ""}` : row.holding_security_name),
    customdata: ranked.map(row => [row.holding_ticker || row.holding_security_name || row.crsp_company_key, row.sector || "Unknown", row.stock_beta, row.weighted_beta]),
    marker: { color: ranked.map((row, index) => (row.holding_ticker || row.holding_security_name || row.crsp_company_key) === state.part5.activeHoldingKey ? "#d13f31" : part5SectorColor(row.sector || "Unknown", index)) },
    hovertemplate: "%{y}<br>Sector：%{customdata[1]}<br>占TNA：%{x:.2%}<br>stock_beta：%{customdata[2]:.4f}<br>weighted_beta：%{customdata[3]:.6f}<extra></extra>"
  }], { title: "被選取基金報告的 Top 持股", height: 370, margin: { l: 180, r: 24, t: 58, b: 44 }, xaxis: { title: "持股占 TNA", tickformat: ".0%", zeroline: false }, yaxis: { automargin: true } }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"] });
  const plot = document.getElementById("part5ReportHoldingsChart");
  resetPlotlyHandler(plot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    const key = point && point.customdata ? cleanText(point.customdata[0]) : "";
    if (!key) return;
    state.part5.activeHoldingKey = key;
    state.part5.brushedHoldingKeys = [];
    renderPart5TopHoldingDetail(filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true }));
  });
}

function renderPart5HoldingExplainTable(rows) {
  if (!dom.part5HoldingExplainTable) return;
  const limited = (rows || []).slice(0, 60);
  if (!limited.length) { dom.part5HoldingExplainTable.innerHTML = ""; return; }
  renderTable(dom.part5HoldingExplainTable, limited, [
    { key: "security_rank", label: "排名", format: "int" }, { key: "holding_ticker", label: "持股Ticker" }, { key: "holding_security_name", label: "持股名稱" }, { key: "sector", label: "Sector" }, { key: "industry", label: "Industry" }, { key: "holdingPct", label: "持股占TNA", format: "pct" }, { key: "stock_beta", label: "stock_beta", format: "num" }, { key: "weighted_beta", label: "weighted_beta", format: "num" }, { key: "portfolio_weighted_beta", label: "portfolio_weighted_beta", format: "num" }, { key: "beta_adjusted_holding_score", label: "beta_adjusted_holding_score", format: "num" }, { key: "yield10y", label: "10年殖利率", format: "yield" }
  ], { title: "被選取基金報告持股明細", expanded: false, count: rows.length });
}

function drawPart5TopHoldingsChart(rows) {
  if (!window.Plotly) return;
  const ranked = aggregatePart5TopHoldings(rows).slice(0, PART5_TOP_HOLDINGS_LIMIT).reverse();
  if (!ranked.length) {
    Plotly.purge("part5TopHoldingsChart");
    if (dom.part5TopHoldingStatus) dom.part5TopHoldingStatus.textContent = "目前篩選條件下沒有 Top 持股";
    return;
  }
  const selectedKeys = new Set(state.part5.brushedHoldingKeys && state.part5.brushedHoldingKeys.length ? state.part5.brushedHoldingKeys : (state.part5.activeHoldingKey ? [state.part5.activeHoldingKey] : []));
  Plotly.react("part5TopHoldingsChart", [{
    type: "bar", orientation: "h",
    x: ranked.map(row => row.score), y: ranked.map(row => row.label),
    customdata: ranked.map(row => [row.searchKey, row.sector, row.avgStockBeta, row.totalWeightedBeta, row.avgHoldingPct]),
    marker: { color: ranked.map(row => selectedKeys.has(row.searchKey) ? "#d13f31" : row.color) },
    hovertemplate: "%{y}<br>Sector：%{customdata[1]}<br>beta_adjusted_holding_score：%{x:.6f}<br>平均 stock_beta：%{customdata[2]:.4f}<br>合計 weighted_beta：%{customdata[3]:.6f}<br>平均持股占TNA：%{customdata[4]:.2%}<extra></extra>"
  }], { title: "Part 5：細看 Top 持股（beta-adjusted，可框選或點選）", height: 410, dragmode: "select", margin: { l: 190, r: 24, t: 58, b: 50 }, xaxis: { title: "beta_adjusted_holding_score", zeroline: false }, yaxis: { automargin: true } }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
  const plot = document.getElementById("part5TopHoldingsChart");
  resetPlotlyHandler(plot, "plotly_selected", eventData => {
    const keys = part5SelectedCustomData(eventData);
    if (!keys.length) return;
    state.part5.brushedHoldingKeys = keys;
    state.part5.activeHoldingKey = keys[0];
    const currentReports = filterPart5Reports({ useFocus: true, useBrush: true });
    const allHoldingRows = filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true });
    const selectedHoldingRows = part5FilterHoldingsByKeys(allHoldingRows, keys);
    renderPart5TopHoldingDetail(allHoldingRows);
    renderPart5TopHoldingPicker(currentReports, rows, selectedHoldingRows);
    updatePart5BrushStatus(currentReports, rows, allHoldingRows);
    setPart5Status(`已框選 ${formatInt(keys.length)} 個 Top 持股，Top 持股快速選取已切換成框選結果。`);
  });
  resetPlotlyHandler(plot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    const key = point && point.customdata ? cleanText(point.customdata[0]) : "";
    if (!key) return;
    state.part5.brushedHoldingKeys = [];
    state.part5.activeHoldingKey = key;
    const currentReports = filterPart5Reports({ useFocus: true, useBrush: true });
    const allHoldingRows = filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true });
    const selectedHoldingRows = part5FilterHoldingsByKeys(allHoldingRows, [key]);
    renderPart5TopHoldingDetail(allHoldingRows);
    renderPart5TopHoldingPicker(currentReports, rows, selectedHoldingRows);
  });
}

function renderPart5TopHoldingDetail(rows) {
  if (!dom.part5TopHoldingStatus || !dom.part5TopHoldingDetailTable) return;
  const activeKeys = state.part5.brushedHoldingKeys && state.part5.brushedHoldingKeys.length ? state.part5.brushedHoldingKeys : (state.part5.activeHoldingKey ? [state.part5.activeHoldingKey] : []);
  if (!activeKeys.length) { dom.part5TopHoldingStatus.textContent = "可在 Top 持股圖框選多個持股，或用快速選取清單選單一持股"; dom.part5TopHoldingDetailTable.innerHTML = ""; return; }
  const keySet = new Set(activeKeys.map(k => cleanText(k).toLowerCase()));
  const picked = (rows || []).filter(row => row.hasStockBeta && [row.holding_ticker, row.holding_security_name, row.crsp_company_key].map(v => cleanText(v).toLowerCase()).some(v => keySet.has(v))).sort((a,b) => b.reportDateMs - a.reportDateMs || (a.security_rank || 999) - (b.security_rank || 999));
  if (!picked.length) { dom.part5TopHoldingStatus.textContent = `已選取 ${activeKeys.join("、")}，但目前範圍沒有明細`; dom.part5TopHoldingDetailTable.innerHTML = ""; return; }
  dom.part5TopHoldingStatus.textContent = activeKeys.length === 1 ? `已鎖定 ${activeKeys[0]}：${formatInt(picked.length)} 筆 / ${formatInt(uniqueCount(picked, row => row.crsp_portno))} 檔基金` : `已框選 ${formatInt(activeKeys.length)} 個持股：${formatInt(picked.length)} 筆明細`;
  renderTable(dom.part5TopHoldingDetailTable, picked.slice(0, 1000), [
    { key: "holding_ticker", label: "持股Ticker" }, { key: "holding_security_name", label: "持股名稱" }, { key: "sector", label: "Sector" }, { key: "industry", label: "Industry" }, { key: "report_dt", label: "報告日" }, { key: "fund_ticker", label: "基金Ticker" }, { key: "fund_name", label: "基金名稱" }, { key: "security_rank", label: "排名", format: "int" }, { key: "holdingPct", label: "持股占TNA", format: "pct" }, { key: "stock_beta", label: "stock_beta", format: "num" }, { key: "weighted_beta", label: "weighted_beta", format: "num" }, { key: "portfolio_weighted_beta", label: "portfolio_weighted_beta", format: "num" }, { key: "beta_adjusted_holding_score", label: "beta_adjusted_holding_score", format: "num" }, { key: "yield10y", label: "10年殖利率", format: "yield" }
  ], { title: activeKeys.length === 1 ? `Top 持股全部明細：${activeKeys[0]}` : `框選 Top 持股全部明細：${activeKeys.length} 個持股`, expanded: false, count: picked.length });
}

function aggregatePart5TopHoldings(rows) {
  const map = new Map();
  for (const row of rows || []) {
    if (!row.hasStockBeta || !Number.isFinite(row.beta_adjusted_holding_score)) continue;
    const key = row.holding_ticker || row.holding_security_name || row.crsp_company_key;
    if (!key) continue;
    if (!map.has(key)) map.set(key, { label: row.holding_ticker ? `${row.holding_ticker} | ${row.holding_security_name || ""}` : row.holding_security_name, searchKey: key, count: 0, marketValue: 0, holdingPctValues: [], stockBetaValues: [], weightedBetaValues: [], portfolioBetaValues: [], scoreValues: [], sectors: [], industries: [] });
    const item = map.get(key);
    item.count += 1;
    if (Number.isFinite(row.holding_market_val)) item.marketValue += row.holding_market_val;
    if (Number.isFinite(row.holdingPct)) item.holdingPctValues.push(row.holdingPct);
    if (Number.isFinite(row.stock_beta)) item.stockBetaValues.push(row.stock_beta);
    if (Number.isFinite(row.weighted_beta)) item.weightedBetaValues.push(row.weighted_beta);
    if (Number.isFinite(row.portfolio_weighted_beta)) item.portfolioBetaValues.push(row.portfolio_weighted_beta);
    if (Number.isFinite(row.beta_adjusted_holding_score)) item.scoreValues.push(row.beta_adjusted_holding_score);
    if (row.sector) item.sectors.push(row.sector);
    if (row.industry) item.industries.push(row.industry);
  }
  return Array.from(map.values()).map((row, index) => {
    const avgHoldingPct = mean(row.holdingPctValues);
    const sector = mode(row.sectors) || "Unknown";
    const industry = mode(row.industries) || "Unknown";
    const score = sum(row.scoreValues);
    const totalWeightedBeta = sum(row.weightedBetaValues);
    const avgStockBeta = mean(row.stockBetaValues);
    const avgPortfolioWeightedBeta = mean(row.portfolioBetaValues);
    return { ...row, avgHoldingPct, sector, industry, score, totalWeightedBeta, avgStockBeta, avgPortfolioWeightedBeta, color: part5SectorColor(sector, index) };
  }).sort((a, b) => b.score - a.score || b.totalWeightedBeta - a.totalWeightedBeta || b.marketValue - a.marketValue);
}

function renderPart5FastPickers(reports, detailRows, allHoldingRows) {
  renderPart5ReportPicker(reports || [], detailRows || []);
  renderPart5TopHoldingPicker(reports || [], detailRows || [], allHoldingRows || []);
}

function part5FilterReportsByKeys(reports, keys) {
  const keySet = new Set((keys || []).map(key => String(key)));
  if (!keySet.size) return reports || [];
  return (reports || []).filter(row => keySet.has(row.reportKey));
}

function part5FilterHoldingsByKeys(rows, keys) {
  const keySet = new Set((keys || []).map(key => cleanText(key).toLowerCase()).filter(Boolean));
  if (!keySet.size) return rows || [];
  return (rows || []).filter(row => {
    const values = [row.holding_ticker, row.holding_security_name, row.crsp_company_key]
      .map(value => cleanText(value).toLowerCase());
    return values.some(value => keySet.has(value));
  });
}

function renderPart5ReportPicker(reports, detailRows) {
  if (!dom.part5ReportPickerStatus || !dom.part5ReportPickerTable) return;
  const rows = (reports || []).slice().sort((a, b) => b.reportDateMs - a.reportDateMs).slice(0, 500);
  if (!rows.length) { dom.part5ReportPickerStatus.textContent = "目前範圍沒有可選基金報告"; dom.part5ReportPickerTable.innerHTML = ""; return; }
  dom.part5ReportPickerStatus.textContent = `可直接點選 ${formatInt(rows.length)} 個基金報告`;
  const activeReportKeys = new Set(state.part5.brushedReportKeys && state.part5.brushedReportKeys.length ? state.part5.brushedReportKeys : (state.part5.activeReportKey ? [state.part5.activeReportKey] : []));
  const body = rows.map(row => `<tr class="part5-click-row${activeReportKeys.has(row.reportKey) ? " active" : ""}" data-report-key="${escapeHtml(row.reportKey)}"><td><button type="button" class="part5-mini-button" data-report-key="${escapeHtml(row.reportKey)}">選取</button></td><td>${escapeHtml(row.report_dt || "-")}</td><td>${escapeHtml(row.fund_ticker || row.crsp_portno || "-")}</td><td>${escapeHtml(row.fund_name || "-")}</td><td>${escapeHtml(formatPct(row.stockPct))}</td><td>${escapeHtml(formatPct(row.bondPct))}</td><td>${escapeHtml(formatYield(row.yield10y))}</td></tr>`).join("");
  dom.part5ReportPickerTable.innerHTML = `<details class="table-panel" open><summary><span class="table-title">基金報告快速選取</span><span class="table-count">${formatInt(rows.length)} 筆</span></summary><div class="table-scroll part5-picker-scroll"><table><thead><tr><th>操作</th><th>報告日</th><th>基金</th><th>基金名稱</th><th>股票</th><th>債券</th><th>殖利率</th></tr></thead><tbody>${body}</tbody></table></div></details>`;
  dom.part5ReportPickerTable.onclick = event => {
    const target = event.target.closest("[data-report-key]");
    if (!target) return;
    const key = cleanText(target.getAttribute("data-report-key"));
    if (!key) return;
    state.part5.brushedReportKeys = [];
    state.part5.activeReportKey = key;
    renderPart5ReportDetail(reports);
    highlightPart5PickerRows(dom.part5ReportPickerTable, "data-report-key", key);
    setPart5Status("已用快速清單選取基金報告，下方 Drill-down 已更新。");
  };
}

function renderPart5TopHoldingPicker(reports, detailRows, allHoldingRows) {
  if (!dom.part5TopHoldingPickerStatus || !dom.part5TopHoldingPickerTable) return;
  const ranked = aggregatePart5TopHoldings(allHoldingRows || []).slice(0, 500);
  if (!ranked.length) { dom.part5TopHoldingPickerStatus.textContent = "目前範圍沒有可選 Top 持股"; dom.part5TopHoldingPickerTable.innerHTML = ""; return; }
  dom.part5TopHoldingPickerStatus.textContent = `可直接點選 ${formatInt(ranked.length)} 個持股`;
  const activeHoldingKeys = new Set(state.part5.brushedHoldingKeys && state.part5.brushedHoldingKeys.length ? state.part5.brushedHoldingKeys : (state.part5.activeHoldingKey ? [state.part5.activeHoldingKey] : []));
  const body = ranked.map((row, index) => `<tr class="part5-click-row${activeHoldingKeys.has(row.searchKey) ? " active" : ""}" data-holding-key="${escapeHtml(row.searchKey)}"><td><button type="button" class="part5-mini-button" data-holding-key="${escapeHtml(row.searchKey)}">選取</button></td><td>${formatInt(index + 1)}</td><td>${escapeHtml(row.label || row.searchKey)}</td><td>${escapeHtml(row.sector || "-")}</td><td>${formatInt(row.count)}</td><td>${escapeHtml(formatPct(row.avgHoldingPct))}</td><td>${escapeHtml(formatValue(row.avgStockBeta, "num"))}</td><td>${escapeHtml(formatValue(row.totalWeightedBeta, "num"))}</td><td>${escapeHtml(formatValue(row.score, "num"))}</td></tr>`).join("");
  dom.part5TopHoldingPickerTable.innerHTML = `<details class="table-panel" open><summary><span class="table-title">Top 持股快速選取</span><span class="table-count">${formatInt(ranked.length)} 檔持股</span></summary><div class="table-scroll part5-picker-scroll"><table><thead><tr><th>操作</th><th>排序</th><th>持股</th><th>Sector</th><th>出現筆數</th><th>平均占TNA</th><th>平均stock_beta</th><th>合計weighted_beta</th><th>beta_adjusted_score</th></tr></thead><tbody>${body}</tbody></table></div></details>`;
  dom.part5TopHoldingPickerTable.onclick = event => {
    const target = event.target.closest("[data-holding-key]");
    if (!target) return;
    const key = cleanText(target.getAttribute("data-holding-key"));
    if (!key) return;
    state.part5.brushedHoldingKeys = [];
    state.part5.activeHoldingKey = key;
    renderPart5TopHoldingDetail(allHoldingRows);
    highlightPart5PickerRows(dom.part5TopHoldingPickerTable, "data-holding-key", key);
    setPart5Status(`已用快速清單鎖定持股 ${key}。`);
  };
}

function highlightPart5PickerRows(container, attrName, activeKey) {
  if (!container) return;
  container.querySelectorAll(".part5-click-row").forEach(row => row.classList.toggle("active", row.getAttribute(attrName) === activeKey));
}

function syncPart5Selection(reports, allHoldings) {
  const reportKeys = new Set((reports || []).map(row => row.reportKey));
  if (state.part5.activeReportKey && !reportKeys.has(state.part5.activeReportKey)) state.part5.activeReportKey = "";
  state.part5.brushedReportKeys = (state.part5.brushedReportKeys || []).filter(key => reportKeys.has(key));
  const holdingKeys = new Set((allHoldings || []).flatMap(row => [row.holding_ticker, row.holding_security_name, row.crsp_company_key].map(cleanText).filter(Boolean)));
  if (state.part5.activeHoldingKey && !holdingKeys.has(state.part5.activeHoldingKey)) state.part5.activeHoldingKey = "";
  state.part5.brushedHoldingKeys = (state.part5.brushedHoldingKeys || []).filter(key => holdingKeys.has(key));
}

function updatePart5BrushStatus(reports, detailRows, allHoldingRows) {
  if (!dom.part5BrushStatus) return;
  const labels = state.part5.brushedPeriodLabels || [];
  const reportCount = (state.part5.brushedReportKeys || []).length;
  const holdingCount = (state.part5.brushedHoldingKeys || []).length;
  const mode = isPart5ManagerMode() ? "Part4 經理人模式" : "全部基金模式";
  const managerText = isPart5ManagerMode()
    ? ` / 經理人框選：${(state.part5.selectedManagerNames || []).length ? state.part5.selectedManagerNames.join("、") : "無，顯示全部 Part4 經理人"}`
    : "";
  dom.part5BrushStatus.textContent = `${mode}${managerText} / 年季框選：${labels.length ? labels.join("、") : "無"} / 基金報告框選：${reportCount || "無"} / Top持股框選：${holdingCount || "無"} / 目前下方資料：${formatInt((reports || []).length)} 報告、${formatInt((detailRows || []).length)} 排名內持股、${formatInt((allHoldingRows || []).length)} 全部持股`;
}

function clearPart5BrushSelection() {
  state.part5.brushedPeriodLabels = [];
  state.part5.brushedReportKeys = [];
  state.part5.brushedHoldingKeys = [];
  state.part5.selectedManagerNames = [];
  state.part5.managerFilteredCache = null;
  state.part5.activeReportKey = "";
  state.part5.activeHoldingKey = "";
  renderPart5();
  setPart5Status("已清除 Part5 框選；快速選取清單保留。");
}

function filterPart5ManagerPanelReports(cache) {
  const period = dom.part5PeriodSelect.value;
  const focusKey = dom.part5FocusSelect.value || "all";
  const mode = dom.part5AggregationSelect.value;
  const brushLabels = new Set(state.part5.brushedPeriodLabels || []);

  return (cache.reports || []).filter(row => {
    if (period !== "all" && row.periodKey !== period) return false;
    if (focusKey !== "all" && part5AggregateLabel(row, mode) !== focusKey) return false;
    if (brushLabels.size && !brushLabels.has(part5AggregateLabel(row, mode))) return false;
    return true;
  });
}

function filterPart5ManagerPanelHoldings(cache, reportKeys) {
  const keys = reportKeys || new Set();
  if (!keys.size) return [];
  const rows = [];
  for (const key of keys) {
    const list = cache.holdingsByReportKey && cache.holdingsByReportKey.get(key);
    if (list && list.length) rows.push(...list);
  }
  return rows;
}

function buildPart5ManagerRows(cache, reports, holdings) {
  const visibleReports = reports || [];
  const visibleHoldings = holdings || [];
  const visibleReportKeys = new Set(visibleReports.map(report => report.reportKey));
  const reportByKey = new Map(visibleReports.map(report => [report.reportKey, report]));
  const holdingCountByReportKey = new Map();

  for (const row of visibleHoldings) {
    holdingCountByReportKey.set(row.reportKey, (holdingCountByReportKey.get(row.reportKey) || 0) + 1);
  }

  return (cache.managers || []).map(manager => {
    const managerReportKeys = cache.managerToReportKeys.get(manager) || new Set();
    const matchedReports = [];
    let holdingCount = 0;
    for (const reportKey of managerReportKeys) {
      if (!visibleReportKeys.has(reportKey)) continue;
      const report = reportByKey.get(reportKey);
      if (report) matchedReports.push(report);
      holdingCount += holdingCountByReportKey.get(reportKey) || 0;
    }

    // Use the same completed-allocation rule as the Part5 stacked chart:
    // if stock/bond/cash do not sum to 100%, the missing portion is treated as bond.
    // This makes the manager table and manager scatter consistent with the allocation stack.
    const completedAllocations = matchedReports
      .map(report => completedPart5Allocation(report))
      .filter(Boolean);
    const avgStock = completedAllocations.length
      ? mean(completedAllocations.map(row => row.stockPct))
      : mean(matchedReports.map(row => row.stockPct));
    const avgBond = completedAllocations.length
      ? mean(completedAllocations.map(row => row.bondPct))
      : mean(matchedReports.map(row => row.bondPct));
    const stockBondRatio = Number.isFinite(avgStock) && Number.isFinite(avgBond) && avgBond > 0 ? avgStock / avgBond : NaN;
    const fundIds = cache.managerToFundIds.get(manager) || new Set();
    const samples = cache.managerFundSamples.get(manager) || [];
    return {
      manager,
      fundCount: fundIds.size,
      part5ReportCount: matchedReports.length,
      holdingCount,
      avgStock,
      avgBond,
      stockBondRatio,
      avgYield: mean(matchedReports.map(row => row.yield10y)),
      funds: samples.join("、")
    };
  });
}

function renderPart5ManagerPanel(reports, holdings) {
  if (!dom.part5ManagerPanel) return;
  const managerMode = isPart5ManagerMode();
  dom.part5ManagerPanel.classList.toggle("hidden", !managerMode);
  if (!managerMode) return;

  const cache = getPart5ManagerMatchCache();
  const allManagers = cache.managers || [];
  if (!allManagers.length) {
    dom.part5ManagerStatus.textContent = "Part4 尚未加入經理人；請先在 Part3 選經理人並加入 Part4";
    if (dom.part5ManagerSummaryCards) dom.part5ManagerSummaryCards.innerHTML = "";
    if (dom.part5ManagerTable) dom.part5ManagerTable.innerHTML = "";
    if (window.Plotly && dom.part5ManagerChart) Plotly.purge("part5ManagerChart");
    return;
  }

  // Manager chart/table should stay selectable across all Part4 managers, so it ignores the manager selection itself,
  // but still respects period, year/quarter brush, focus, and other Part5 time filters.
  const panelReports = filterPart5ManagerPanelReports(cache);
  const panelReportKeys = new Set(panelReports.map(report => report.reportKey));
  const panelHoldings = filterPart5ManagerPanelHoldings(cache, panelReportKeys);
  const managerRows = buildPart5ManagerRows(cache, panelReports, panelHoldings);
  const selectedManagerSet = part5ManagerSelectionSet();
  const activeManagers = activePart5Managers();
  const activeManagerLabel = selectedManagerSet.size
    ? `；目前鎖定 ${formatInt(selectedManagerSet.size)} 位經理人`
    : "；目前顯示全部 Part4 經理人";

  dom.part5ManagerStatus.textContent = `Part4 已選 ${formatInt(allManagers.length)} 位經理人；目前時間範圍可匹配 ${formatInt(panelReports.length)} 個基金報告${activeManagerLabel}`;

  if (dom.part5ManagerSummaryCards) {
    dom.part5ManagerSummaryCards.innerHTML = [
      { label: "Part4 經理人", value: formatInt(allManagers.length), text: "此模式使用快取後的 Part4 經理人對照結果，不會每次重掃全部資料。" },
      { label: "目前鎖定經理人", value: selectedManagerSet.size ? formatInt(selectedManagerSet.size) : "全部", text: "可在經理人散點圖點選或框選；下方所有 Part5 圖表與明細會同步更新。" },
      { label: "匹配基金報告", value: formatInt((reports || []).length), text: "這是目前真正套用到下方圖表、快速選取與明細表的基金報告數。" },
      { label: "平均殖利率", value: formatYield(mean((reports || []).map(row => row.yield10y))), text: "可搭配股債比觀察 Part4 經理人所在基金的市場環境。" }
    ].map(card => `<div class="part5-insight-card"><span>${escapeHtml(card.label)}</span><strong>${escapeHtml(card.value)}</strong><p>${escapeHtml(card.text)}</p></div>`).join("");
  }

  if (window.Plotly && dom.part5ManagerChart) {
    const scatterRows = managerRows.filter(row =>
      Number.isFinite(row.avgYield) && Number.isFinite(row.stockBondRatio)
    );
    if (!scatterRows.length) {
      Plotly.react("part5ManagerChart", [], {
        title: "Part4 經理人在 Part5 中的匹配程度：殖利率 vs 股債比",
        height: 360,
        margin: { l: 68, r: 36, t: 58, b: 62 },
        dragmode: "select",
        xaxis: { title: "10 年期殖利率 (%)" },
        yaxis: { title: "股債比（平均股票 / 平均債券）" },
        annotations: [{ text: "目前沒有足夠資料計算股債比與殖利率", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false }],
        uirevision: "part5-manager-select-fast"
      }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
    } else {
      Plotly.react("part5ManagerChart", [{
        type: "scattergl",
        mode: "markers+text",
        x: scatterRows.map(row => row.avgYield),
        y: scatterRows.map(row => row.stockBondRatio),
        text: scatterRows.map(row => row.manager),
        textposition: "top center",
        customdata: scatterRows.map(row => [row.manager, row.part5ReportCount, row.holdingCount, row.avgStock, row.avgBond]),
        marker: {
          size: scatterRows.map(row => selectedManagerSet.has(row.manager) ? Math.max(15, Math.min(32, 11 + Math.sqrt(row.part5ReportCount || 0) * 4)) : Math.max(9, Math.min(26, 7 + Math.sqrt(row.part5ReportCount || 0) * 4))),
          color: scatterRows.map(row => row.avgStock),
          colorscale: "Viridis",
          showscale: true,
          colorbar: { title: "平均股票", tickformat: ".0%", thickness: 12 },
          opacity: scatterRows.map(row => selectedManagerSet.size && !selectedManagerSet.has(row.manager) ? 0.22 : 0.84),
          line: { color: scatterRows.map(row => selectedManagerSet.has(row.manager) ? "#d13f31" : "rgba(20,45,64,0.35)"), width: scatterRows.map(row => selectedManagerSet.has(row.manager) ? 3 : 0.8) }
        },
        hovertemplate: "經理人：%{customdata[0]}<br>10年期殖利率：%{x:.2f}%<br>股債比：%{y:.2f}<br>平均股票：%{customdata[3]:.2%}<br>平均債券：%{customdata[4]:.2%}<br>匹配基金報告：%{customdata[1]}<br>持股明細：%{customdata[2]}<extra></extra>",
        name: "Part4 經理人"
      }], {
        title: "Part4 經理人在 Part5 中的匹配程度：殖利率 vs 股債比（可框選並影響下方）",
        height: 400,
        margin: { l: 72, r: 70, t: 62, b: 64 },
        dragmode: "select",
        hovermode: "closest",
        xaxis: { title: "10 年期殖利率 (%)", zeroline: false },
        yaxis: { title: "股債比（平均股票 / 平均債券）", zeroline: false },
        showlegend: false,
        uirevision: "part5-manager-select-fast"
      }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
    }

    const plot = document.getElementById("part5ManagerChart");
    resetPlotlyHandler(plot, "plotly_selected", eventData => {
      const names = part5SelectedManagerNames(eventData);
      if (!names.length) return;
      state.part5.selectedManagerNames = names;
      state.part5.managerFilteredCache = null;
      state.part5.activeReportKey = "";
      state.part5.activeHoldingKey = "";
      state.part5.brushedReportKeys = [];
      state.part5.brushedHoldingKeys = [];
      renderPart5();
      setPart5Status(`已框選 ${formatInt(names.length)} 位 Part4 經理人，下方 Part5 內容已同步更新。`);
    });
    resetPlotlyHandler(plot, "plotly_click", eventData => {
      const point = eventData && eventData.points && eventData.points[0];
      const manager = point && point.customdata ? cleanText(point.customdata[0]) : "";
      if (!manager) return;
      state.part5.selectedManagerNames = [manager];
      state.part5.managerFilteredCache = null;
      state.part5.activeReportKey = "";
      state.part5.activeHoldingKey = "";
      state.part5.brushedReportKeys = [];
      state.part5.brushedHoldingKeys = [];
      renderPart5();
      setPart5Status(`已鎖定 Part4 經理人：${manager}，下方 Part5 內容已同步更新。`);
    });
  }

  if (dom.part5ManagerTable) {
    renderTable(dom.part5ManagerTable, managerRows, [
      { key: "manager", label: "Part4 經理人" },
      { key: "fundCount", label: "Part1基金數", format: "int" },
      { key: "part5ReportCount", label: "匹配Part5報告", format: "int" },
      { key: "holdingCount", label: "持股明細", format: "int" },
      { key: "avgStock", label: "平均股票", format: "pct" },
      { key: "avgBond", label: "平均債券", format: "pct" },
      { key: "stockBondRatio", label: "股債比", format: "num" },
      { key: "avgYield", label: "平均10年殖利率", format: "yield" },
      { key: "funds", label: "Part1基金名稱樣本" }
    ], { title: `Part4 經理人與 Part5 基金報告對照表（目前下方套用：${activeManagers.join("、") || "全部"}）`, expanded: true, count: managerRows.length });
  }
}

function renderPart5SummaryTable(reports, holdings) {
  const map = new Map();
  for (const report of reports || []) {
    if (!map.has(report.periodKey)) map.set(report.periodKey, { periodKey: report.periodKey, period: report.periodLabel, years: [], funds: new Set(), reportCount: 0, holdingCount: 0, betaHoldingCount: 0, stockValues: [], bondValues: [], cashValues: [], yieldValues: [], portfolioBetaValues: [] });
    const row = map.get(report.periodKey);
    row.reportCount += 1;
    row.years.push(report.year);
    if (report.crsp_portno) row.funds.add(report.crsp_portno);
    if (Number.isFinite(report.stockPct)) row.stockValues.push(report.stockPct);
    if (Number.isFinite(report.bondPct)) row.bondValues.push(report.bondPct);
    if (Number.isFinite(report.cashPct)) row.cashValues.push(report.cashPct);
    if (Number.isFinite(report.yield10y)) row.yieldValues.push(report.yield10y);
    if (Number.isFinite(report.portfolio_weighted_beta)) row.portfolioBetaValues.push(report.portfolio_weighted_beta);
  }
  for (const holding of holdings || []) {
    const row = map.get(holding.periodKey);
    if (row) {
      row.holdingCount += 1;
      if (holding.hasStockBeta) row.betaHoldingCount += 1;
    }
  }
  const rows = Array.from(map.values()).map(row => ({ period: row.period, yearRange: row.years.length ? `${Math.min(...row.years)}-${Math.max(...row.years)}` : "-", reportCount: row.reportCount, fundCount: row.funds.size, holdingCount: row.holdingCount, betaHoldingCount: row.betaHoldingCount, avgYield: mean(row.yieldValues), avgStock: mean(row.stockValues), avgBond: mean(row.bondValues), avgCash: mean(row.cashValues), avgPortfolioWeightedBeta: mean(row.portfolioBetaValues) }));
  renderTable("part5SummaryTable", rows, [
    { key: "period", label: "期間" }, { key: "yearRange", label: "年份" }, { key: "reportCount", label: "基金報告數", format: "int" }, { key: "fundCount", label: "基金數", format: "int" }, { key: "holdingCount", label: "持股筆數", format: "int" }, { key: "betaHoldingCount", label: "beta持股數", format: "int" }, { key: "avgYield", label: "平均10年殖利率", format: "yield" }, { key: "avgStock", label: "平均股票", format: "pct" }, { key: "avgBond", label: "平均債券", format: "pct" }, { key: "avgCash", label: "平均現金", format: "pct" }, { key: "avgPortfolioWeightedBeta", label: "平均portfolio_beta", format: "num" }
  ], { title: "Part5 期間摘要", expanded: true });
}

function renderPart5HoldingsTable(rows) {
  const limit = Math.max(50, parseNumber(dom.part5LimitSelect.value) || 250);
  const limited = (rows || []).slice(0, limit);
  renderTable("part5HoldingsTable", limited, [
    { key: "periodLabel", label: "期間" }, { key: "report_dt", label: "報告日" }, { key: "fund_ticker", label: "基金Ticker" }, { key: "fund_name", label: "基金名稱" }, { key: "security_rank", label: "排名", format: "int" }, { key: "holding_ticker", label: "持股Ticker" }, { key: "holding_security_name", label: "持股名稱" }, { key: "sector", label: "Sector" }, { key: "industry", label: "Industry" }, { key: "holdingPct", label: "持股占TNA", format: "pct" }, { key: "stock_beta", label: "stock_beta", format: "num" }, { key: "weighted_beta", label: "weighted_beta", format: "num" }, { key: "portfolio_weighted_beta", label: "portfolio_weighted_beta", format: "num" }, { key: "beta_adjusted_holding_score", label: "beta_adjusted_holding_score", format: "num" }, { key: "stockPct", label: "基金股票", format: "pct" }, { key: "bondPct", label: "基金債券", format: "pct" }, { key: "cashPct", label: "基金現金", format: "pct" }, { key: "yield10y", label: "10年殖利率", format: "yield" }
  ], { title: `Part5 持股明細（前 ${formatInt(limited.length)} / ${formatInt((rows || []).length)} 筆）`, expanded: false, count: (rows || []).length });
}


function part5BSelectionKey(row) {
  const parts = [row.teacher_category, row.teacher_subcategory, row.holding_ticker, row.holding_security_name]
    .map(value => cleanText(value).toLowerCase())
    .filter(Boolean);
  return parts.join("|");
}

function ensurePart5BState() {
  if (!Array.isArray(state.part5.part5BSelectedCategories)) state.part5.part5BSelectedCategories = [];
  if (!Array.isArray(state.part5.part5BSelectedYears)) state.part5.part5BSelectedYears = [];
  if (!Array.isArray(state.part5.part5BSelectedItemKeys)) state.part5.part5BSelectedItemKeys = [];
}

function clearPart5BSelection() {
  ensurePart5BState();
  state.part5.part5BSelectedCategories = [];
  state.part5.part5BSelectedYears = [];
  state.part5.part5BSelectedItemKeys = [];
  if (dom.part5BStatus) dom.part5BStatus.textContent = "Part5B 內部框選已清除。";
  renderPart5BExcludedInsights();
  renderPart6();
}

function part5BActiveSelectionText(items, panelRows) {
  const modeLabel = isPart5ManagerMode() ? "Part4 經理人分析" : "全部基金 / 持股分析";
  const yearList = Array.from(part5BSelectedYears()).sort((a, b) => a - b);
  const brushText = (state.part5.brushedPeriodLabels || []).length
    ? `Part5 年/季框選：${state.part5.brushedPeriodLabels.join("、")}`
    : "Part5 年/季框選：無";
  const managerText = isPart5ManagerMode()
    ? ` / 經理人框選：${(state.part5.selectedManagerNames || []).length ? state.part5.selectedManagerNames.join("、") : "全部 Part4 經理人"}`
    : "";
  const yearsText = yearList.length ? ` / 連動年份：${yearList.join("、")}` : " / 連動年份：全部可用年份";
  return `${modeLabel}${managerText} / ${brushText}${yearsText}；Part5B 自動顯示 ${formatInt((items || []).length)} 個 item、${formatInt((panelRows || []).length)} 個 year-category panel row。`;
}

function applyPart5BSelectionPatch({ category = null, year = null, itemKey = null, multi = false } = {}) {
  // Part5B Demo A/B/C is intentionally driven by the active Part5 analysis state only.
  // It no longer creates an independent Part5B selection state. The upstream controls are:
  // - Part5 overview trend brush/click
  // - Part5 allocation stack brush/click
  // - Part5 manager-mode scatter brush/click
  // plus period/focus filters and the independent memory of each Part5 analysis mode.
  ensurePart5BState();
  state.part5.part5BSelectedCategories = [];
  state.part5.part5BSelectedYears = [];
  state.part5.part5BSelectedItemKeys = [];
  if (dom.part5BStatus) {
    dom.part5BStatus.textContent = "Part5B 目前由 Part5 分析模式與 Part5 圖表框選自動連動；不使用 Part5B 內部獨立框選。";
  }
  renderPart5BExcludedInsights();
  renderPart6();
}

function toggleArrayValue(array, value) {
  const index = array.findIndex(item => item === value);
  if (index >= 0) array.splice(index, 1);
  else array.push(value);
}

function filterPart5BByInternalSelection(rows) {
  // Part5B no longer narrows itself by internal selections. All narrowing comes from Part5's
  // current mode, period/focus settings, and upstream Part5 chart brushes.
  return rows || [];
}

function renderPart5BExcludedInsights() {
  ensurePart5BState();
  state.part5.part5BSelectedCategories = [];
  state.part5.part5BSelectedYears = [];
  state.part5.part5BSelectedItemKeys = [];
  const items = filterPart5BItems(state.part5.excludedEnrichedRows || []);
  const topItems = filterPart5BItems(state.part5.excludedTopRows || []);
  const panelRows = filterPart5BPanelRows(state.part5.excludedPanelRows || []);
  const removedRows = filterPart5BItems(state.part5.excludedRemovedRows || [], { ignoreUseFlag: true });
  renderPart5BCards(items, removedRows);
  drawPart5BCharts(items, topItems, panelRows);
  renderPart5BTables(topItems, removedRows);
  if (dom.part5BStatus) dom.part5BStatus.textContent = part5BActiveSelectionText(items, panelRows);
}

function part5BPeriodYearRange() {
  const period = dom.part5PeriodSelect ? dom.part5PeriodSelect.value : "all";
  if (period === "before2010") return [-Infinity, 2009];
  if (period === "y2010_2014") return [2010, 2014];
  if (period === "y2015_2019") return [2015, 2019];
  if (period === "y2020_2026") return [2020, 2026];
  return [-Infinity, Infinity];
}

function part5BSelectedYears() {
  const years = new Set();

  // Part5B follows the current Part5 view. This respects:
  // - all-fund mode vs Part4 manager mode
  // - independent memory of each mode
  // - period/focus selectors
  // - Part5 overview and allocation brushes
  // - manager scatter selection in manager mode
  if (state.part5.loaded && dom.part5PeriodSelect) {
    try {
      const reports = filterPart5Reports({ useFocus: true, useBrush: true });
      for (const report of reports || []) {
        if (Number.isFinite(report.year)) years.add(report.year);
      }
    } catch (error) {
      // Fall back to label parsing below.
    }
  }

  const labels = [];
  const focus = dom.part5FocusSelect ? dom.part5FocusSelect.value : "all";
  if (focus && focus !== "all") labels.push(focus);
  for (const label of state.part5.brushedPeriodLabels || []) labels.push(label);
  for (const label of labels) {
    const match = String(label).match(/^(\d{4})/);
    if (match) years.add(Number(match[1]));
  }
  return years;
}

function filterPart5BPanelRows(rows) {
  const [lo, hi] = part5BPeriodYearRange();
  const years = part5BSelectedYears();
  return (rows || []).filter(row => {
    if (!Number.isFinite(row.year) || row.year < lo || row.year > hi) return false;
    if (years.size && !years.has(row.year)) return false;
    return true;
  }).sort((a, b) => a.year - b.year || String(a.teacher_category).localeCompare(String(b.teacher_category)));
}

function filterPart5BItems(rows, options = {}) {
  const [lo, hi] = part5BPeriodYearRange();
  const years = part5BSelectedYears();
  return (rows || []).filter(row => {
    if (!options.ignoreUseFlag && row.use_in_part5b_two_group && String(row.use_in_part5b_two_group).toLowerCase() === "false") return false;
    if (Number.isFinite(row.firstYear) && Number.isFinite(row.lastYear)) {
      if (row.lastYear < lo || row.firstYear > hi) return false;
      if (years.size) {
        let overlaps = false;
        for (const year of years) {
          if (year >= row.firstYear && year <= row.lastYear) overlaps = true;
        }
        if (!overlaps) return false;
      }
    }
    return true;
  }).sort((a, b) => (b.holding_record_count || 0) - (a.holding_record_count || 0));
}

function summarizePart5BItems(items) {
  const map = new Map();
  for (const row of items || []) {
    const key = row.teacher_category || "Unknown";
    if (!map.has(key)) map.set(key, { teacher_category: key, excluded_item_count: 0, holding_record_count: 0, unique_portfolio_count: 0, high_confidence_items: 0, review_items: 0 });
    const item = map.get(key);
    item.excluded_item_count += 1;
    item.holding_record_count += Number.isFinite(row.holding_record_count) ? row.holding_record_count : 0;
    item.unique_portfolio_count += Number.isFinite(row.unique_portfolio_count) ? row.unique_portfolio_count : 0;
    if (row.classification_confidence === "high") item.high_confidence_items += 1;
    if (row.classification_confidence === "review") item.review_items += 1;
  }
  return Array.from(map.values()).sort((a, b) => b.holding_record_count - a.holding_record_count);
}

function renderPart5BCards(items, removedRows) {
  if (!dom.part5BCaseCards) return;
  const summary = summarizePart5BItems(items);
  const totalRecords = sum(summary.map(row => row.holding_record_count));
  const totalPortfolios = sum(summary.map(row => row.unique_portfolio_count));
  const highConfidence = sum(summary.map(row => row.high_confidence_items));
  const reviewItems = sum(summary.map(row => row.review_items));
  const indirectEquity = summary.find(row => row.teacher_category === "Equity Fund / Stock-fund-like");
  const bondMoney = summary.find(row => row.teacher_category === "Bond / Credit / Money-related");
  const cards = [
    { label: "Demo A 總覽", value: formatInt((items || []).length), text: `non-individual holdings，不是 beta pipeline 垃圾；record ${formatInt(totalRecords)} / portfolio proxy ${formatInt(totalPortfolios)}。` },
    { label: "Bond/Credit/Money", value: formatInt(bondMoney ? bondMoney.holding_record_count : 0), text: "可接 10Y yield、流動性、agency/MBS、cash/money market 與 credit exposure。" },
    { label: "Indirect Equity", value: formatInt(indirectEquity ? indirectEquity.holding_record_count : 0), text: "ETF、stock fund、equity portfolio 代表間接股票曝險，不等於公司個股 beta。" },
    { label: "資料治理", value: `${formatInt(highConfidence)} / ${formatInt(reviewItems)}`, text: `high / review 分類；另移除疑似個股 ${formatInt((removedRows || []).length)} 筆 audit item。` }
  ];
  dom.part5BCaseCards.innerHTML = cards.map(card => `<div class="part5-insight-card"><span>${escapeHtml(card.label)}</span><strong>${escapeHtml(card.value)}</strong><p>${escapeHtml(card.text)}</p></div>`).join("");
}

function drawPart5BCharts(items, topItems, panelRows) {
  if (!window.Plotly) return;
  const summary = summarizePart5BItems(items);
  if (!summary.length) {
    ["part5BOverviewChart", "part5BYearChart", "part5BRateCaseChart", "part5BEquityCaseChart"].forEach(id => Plotly.purge(id));
    return;
  }

  Plotly.react("part5BOverviewChart", [{
    type: "bar",
    x: summary.map(row => row.teacher_category),
    y: summary.map(row => row.holding_record_count),
    customdata: summary.map(row => [row.excluded_item_count, row.unique_portfolio_count, row.high_confidence_items, row.review_items]),
    marker: { color: summary.map((row, index) => row.teacher_category.startsWith("Bond") ? "#12a59a" : part5SectorColor("Technology", index)) },
    hovertemplate: "%{x}<br>holding records：%{y:,.0f}<br>items：%{customdata[0]:,.0f}<br>portfolio proxy：%{customdata[1]:,.0f}<br>high/review：%{customdata[2]:,.0f} / %{customdata[3]:,.0f}<extra></extra>"
  }], {
    title: "Part5B Demo A：兩大類 non-individual holdings exposure",
    height: 360,
    margin: { l: 68, r: 26, t: 58, b: 78 },
    xaxis: { automargin: true },
    yaxis: { title: "holding record count" }
  }, { displaylogo: false, responsive: true });
  const overviewPlot = document.getElementById("part5BOverviewChart");
  resetPlotlyHandler(overviewPlot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    const category = point ? cleanText(point.x) : "";
    if (category) applyPart5BSelectionPatch({ category, multi: eventData && eventData.event && eventData.event.shiftKey });
  });
  resetPlotlyHandler(overviewPlot, "plotly_selected", eventData => {
    if (dom.part5BStatus) dom.part5BStatus.textContent = "Part5B 內部框選已停用；請改用上方 Part5 趨勢圖、配置堆疊圖或經理人匹配圖框選。";
  });

  drawPart5BYearChart(panelRows);
  drawPart5BCaseChart("part5BRateCaseChart", part5BRateCaseRows(items, topItems), "Demo B：利率敏感 / liquidity case", "records");
  drawPart5BCaseChart("part5BEquityCaseChart", part5BEquityCaseRows(items, topItems), "Demo C：間接股票曝險 case", "records");
}

function drawPart5BYearChart(panelRows) {
  if (!window.Plotly) return;
  if (!panelRows || !panelRows.length) { Plotly.purge("part5BYearChart"); return; }
  const categories = Array.from(new Set(panelRows.map(row => row.teacher_category))).sort();
  const traces = categories.map(category => {
    const rows = panelRows.filter(row => row.teacher_category === category).sort((a, b) => a.year - b.year);
    return {
      type: "scatter",
      mode: "lines+markers",
      x: rows.map(row => row.year),
      y: rows.map(row => row.holding_record_count_proxy),
      name: category,
      line: { width: 2.4 },
      hovertemplate: "%{x}<br>record proxy：%{y:,.1f}<extra></extra>"
    };
  });
  const yieldRows = Array.from(new Map(panelRows.filter(row => Number.isFinite(row.yield10y)).map(row => [row.year, row.yield10y])).entries()).sort((a, b) => a[0] - b[0]);
  if (yieldRows.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: yieldRows.map(row => row[0]),
      y: yieldRows.map(row => row[1]),
      yaxis: "y2",
      name: "10Y Treasury yield",
      line: { color: "#d88c18", width: 2.6, dash: "dot" },
      hovertemplate: "%{x}<br>10Y：%{y:.2f}%<extra></extra>"
    });
  }
  const selectedYears = part5BSelectedYears();
  const selectedShapes = selectedYears.size ? Array.from(selectedYears).map(year => ({
    type: "rect",
    xref: "x",
    yref: "paper",
    x0: year - 0.45,
    x1: year + 0.45,
    y0: 0,
    y1: 1,
    fillcolor: "rgba(31,111,178,0.08)",
    line: { color: "rgba(31,111,178,0.35)", width: 1 }
  })) : [];
  Plotly.react("part5BYearChart", traces, {
    title: "Part5B：active-year exposure proxy vs 10Y Treasury yield（依 Part5 框選自動連動）",
    height: 380,
    margin: { l: 68, r: 64, t: 58, b: 58 },
    dragmode: "select",
    hovermode: "x unified",
    legend: { orientation: "h", y: -0.2 },
    shapes: selectedShapes,
    uirevision: "part5b-year-brush",
    xaxis: { title: "year" },
    yaxis: { title: "holding record proxy" },
    yaxis2: { title: "10Y yield", overlaying: "y", side: "right", ticksuffix: "%", showgrid: false }
  }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
  const yearPlot = document.getElementById("part5BYearChart");
  resetPlotlyHandler(yearPlot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    const year = point && Number.isFinite(Number(point.x)) ? Number(point.x) : NaN;
    const category = point && point.data ? cleanText(point.data.name) : "";
    if (Number.isFinite(year)) {
      if (category && !category.toLowerCase().includes("yield")) state.part5.part5BSelectedCategories = [category];
      applyPart5BSelectionPatch({ year, multi: eventData && eventData.event && eventData.event.shiftKey });
    }
  });
  resetPlotlyHandler(yearPlot, "plotly_selected", eventData => {
    if (dom.part5BStatus) dom.part5BStatus.textContent = "Part5B year chart 僅作為結果顯示；請用上方 Part5 趨勢圖或配置堆疊圖框選年份。";
  });
}

function drawPart5BCaseChart(chartId, rows, title, xTitle) {
  if (!window.Plotly) return;
  const ranked = (rows || []).slice(0, 14).reverse();
  if (!ranked.length) { Plotly.purge(chartId); return; }
  Plotly.react(chartId, [{
    type: "bar",
    orientation: "h",
    x: ranked.map(row => row.holding_record_count),
    y: ranked.map(row => row.item_label),
    customdata: ranked.map(row => [row.teacher_subcategory, row.unique_portfolio_count, row.first_report_dt, row.last_report_dt, row.classification_confidence, part5BSelectionKey(row)]),
    marker: {
      color: ranked.map(row => (state.part5.part5BSelectedItemKeys || []).includes(part5BSelectionKey(row)) ? "#d13f31" : (row.teacher_category.startsWith("Bond") ? "#12a59a" : "#1677c2")),
      line: { color: ranked.map(row => (state.part5.part5BSelectedItemKeys || []).includes(part5BSelectionKey(row)) ? "#7f1d1d" : "rgba(20,45,64,0.18)"), width: ranked.map(row => (state.part5.part5BSelectedItemKeys || []).includes(part5BSelectionKey(row)) ? 2 : 0.4) }
    },
    hovertemplate: "%{y}<br>%{customdata[0]}<br>records：%{x:,.0f}<br>portfolio proxy：%{customdata[1]:,.0f}<br>%{customdata[2]} - %{customdata[3]}<br>%{customdata[4]}<extra></extra>"
  }], {
    title,
    height: 400,
    margin: { l: 210, r: 24, t: 58, b: 50 },
    xaxis: { title: xTitle },
    yaxis: { automargin: true }
  }, { displaylogo: false, responsive: true, modeBarButtonsToAdd: ["select2d", "lasso2d"], modeBarButtonsToRemove: ["hoverClosestCartesian", "hoverCompareCartesian"] });
  const casePlot = document.getElementById(chartId);
  resetPlotlyHandler(casePlot, "plotly_click", eventData => {
    const point = eventData && eventData.points && eventData.points[0];
    const key = point && point.customdata ? cleanText(point.customdata[5]) : "";
    if (key) applyPart5BSelectionPatch({ itemKey: key, multi: eventData && eventData.event && eventData.event.shiftKey });
  });
  resetPlotlyHandler(casePlot, "plotly_selected", eventData => {
    if (dom.part5BStatus) dom.part5BStatus.textContent = "Part5B case chart 僅作為結果顯示；請用上方 Part5 圖表框選來連動 Demo A/B/C。";
  });
}

function part5BRateCaseRows(items, topItems) {
  const source = items && items.length ? items : topItems;
  const pattern = /(FANNIE|FNMA|FREDDIE|FHLMC|GINNIE|GNMA|TBA|TREAS|TREASURY|USD CASH|MONEY MARKET|CASH|REPO|AGG|BND|LQD|HYG|TLT|IEF|SHY|TIP)/i;
  return (source || []).filter(row => row.teacher_category.startsWith("Bond") && pattern.test(`${row.holding_ticker} ${row.holding_security_name} ${row.teacher_subcategory}`)).sort((a, b) => (b.holding_record_count || 0) - (a.holding_record_count || 0));
}

function part5BEquityCaseRows(items, topItems) {
  const source = items && items.length ? items : topItems;
  const pattern = /(IVV|EFA|IEFA|SPY|VOO|VTI|QQQ|FIDELITY|EQUITY|STOCK|ETF|ISHARES|VANGUARD|MSCI|S&P|RUSSELL)/i;
  return (source || []).filter(row => row.teacher_category.startsWith("Equity") && pattern.test(`${row.holding_ticker} ${row.holding_security_name} ${row.teacher_subcategory}`)).sort((a, b) => (b.holding_record_count || 0) - (a.holding_record_count || 0));
}

function renderPart5BTables(topItems, removedRows) {
  renderTable("part5BTopItemsTable", (topItems || []).slice(0, 120), [
    { key: "teacher_category", label: "兩大類" },
    { key: "teacher_subcategory", label: "子類" },
    { key: "holding_ticker", label: "Ticker" },
    { key: "holding_security_name", label: "名稱" },
    { key: "holding_record_count", label: "Records", format: "int" },
    { key: "unique_portfolio_count", label: "Portfolio proxy", format: "int" },
    { key: "first_report_dt", label: "First" },
    { key: "last_report_dt", label: "Last" },
    { key: "classification_confidence", label: "Confidence" }
  ], { title: "Part5B Top non-individual exposure items", expanded: false, count: (topItems || []).length });
  renderTable("part5BRemovedAuditTable", (removedRows || []).slice(0, 80), [
    { key: "holding_ticker", label: "Ticker" },
    { key: "holding_security_name", label: "名稱" },
    { key: "exclude_reason", label: "原排除原因" },
    { key: "holding_record_count", label: "Records", format: "int" },
    { key: "unique_portfolio_count", label: "Portfolio proxy", format: "int" },
    { key: "first_report_dt", label: "First" },
    { key: "last_report_dt", label: "Last" }
  ], { title: "資料治理：移除疑似 individual-stock-like audit", expanded: false, count: (removedRows || []).length });
}

function clearPart5Filters() {
  // Clear only the currently active Part5 mode.
  // Do not switch between "all" and "manager" here; those two modes keep independent memory.
  const mode = state.part5.analysisMode === "manager" ? "manager" : "all";
  const memory = ensurePart5ModeMemory();

  dom.part5PeriodSelect.value = "all";
  dom.part5AggregationSelect.value = "year";
  dom.part5FocusSelect.value = "all";
  dom.part5RankInput.value = "10";
  dom.part5LimitSelect.value = "250";
  dom.part5SearchInput.value = "";

  state.part5.activeReportKey = "";
  state.part5.activeHoldingKey = "";
  state.part5.brushedPeriodLabels = [];
  state.part5.brushedReportKeys = [];
  state.part5.brushedHoldingKeys = [];
  state.part5.part5BSelectedCategories = [];
  state.part5.part5BSelectedYears = [];
  state.part5.part5BSelectedItemKeys = [];
  if (mode === "manager") {
    state.part5.selectedManagerNames = [];
    state.part5.managerFilteredCache = null;
  }

  memory[mode] = capturePart5ModeState();
  renderPart5();
  renderPart6();
  setPart5Status(`${mode === "manager" ? "Part4 經理人模式" : "全部基金模式"} 的 Part5 篩選已清除；另一個模式的記憶不受影響。`);
}

function resetPart5Data(updateUi) {
  state.part5 = { loaded: false, loading: false, yieldRows: [], yieldMonthMap: new Map(), yieldYearMap: new Map(), stockBetaRows: [], stockBetaMap: new Map(), excludedSummaryRows: [], excludedPanelRows: [], excludedEnrichedRows: [], excludedTopRows: [], excludedRemovedRows: [], holdings: [], reports: [], reportKeys: new Set(), reportMap: new Map(), activeReportKey: "", activeHoldingKey: "", analysisMode: "all", brushedPeriodLabels: [], brushedReportKeys: [], brushedHoldingKeys: [], selectedManagerNames: [], part5BSelectedCategories: [], part5BSelectedYears: [], part5BSelectedItemKeys: [], modeMemory: defaultPart5ModeMemory(), managerMatchCache: null, managerFilteredCache: null };
  if (!updateUi || !dom.part5Status) return;
  if (dom.part5AnalysisModeSelect) dom.part5AnalysisModeSelect.value = "all";
  dom.metricP5Reports.textContent = "-"; dom.metricP5Funds.textContent = "-"; dom.metricP5Holdings.textContent = "-"; dom.metricP5Yield.textContent = "-";
  ["part5SummaryTable", "part5HoldingsTable", "part5ReportDetailTable", "part5HoldingExplainTable", "part5TopHoldingDetailTable", "part5ReportPickerTable", "part5TopHoldingPickerTable", "part5ManagerSummaryCards", "part5ManagerTable", "part5BCaseCards", "part5BTopItemsTable", "part5BRemovedAuditTable"].forEach(id => { if (dom[id]) dom[id].innerHTML = ""; });
  if (dom.part5ReportDetailStatus) dom.part5ReportDetailStatus.textContent = "尚未點選基金報告";
  if (dom.part5TopHoldingStatus) dom.part5TopHoldingStatus.textContent = "尚未鎖定持股";
  if (dom.part5BrushStatus) dom.part5BrushStatus.textContent = "尚未框選";
  if (dom.part5BStatus) dom.part5BStatus.textContent = "Part5B 尚未框選。";
  if (dom.part5ManagerStatus) dom.part5ManagerStatus.textContent = "尚未切換到經理人模式";
  if (dom.part5ManagerPanel) dom.part5ManagerPanel.classList.add("hidden");
  dom.part5FocusSelect.innerHTML = '<option value="all">全部年份</option>';
  if (dom.part5Results) dom.part5Results.classList.add("hidden");
  dom.loadPart5Btn.disabled = false;
  setPart5Status("尚未載入 Part5 資料。");
  if (window.Plotly) ["part5OverviewChart", "part5AllocationChart", "part5TopHoldingsChart", "part5ReportScatterChart", "part5ReportHoldingsChart", "part5ManagerChart", "part5BOverviewChart", "part5BYearChart", "part5BRateCaseChart", "part5BEquityCaseChart"].forEach(id => Plotly.purge(id));
  renderPart6();
}

function renderPart6() {
  if (!dom.part6Section) return;
  if (!state.rows || !state.rows.length) {
    dom.part6Section.classList.add("hidden");
    if (dom.part6Status) dom.part6Status.textContent = "請先載入 Part1 基金月資料。";
    return;
  }
  dom.part6Section.classList.remove("hidden");
  state.part6.mode = dom.part6ModeSelect ? (dom.part6ModeSelect.value || "backend") : (state.part6.mode || "backend");
  if (dom.part6WindowSelect) dom.part6WindowSelect.value = "y3";
  if (dom.part6TargetSelect && !dom.part6TargetSelect.value) dom.part6TargetSelect.value = String(state.part6.horizonMonths || 12);
  if (dom.part6FrontendContent) dom.part6FrontendContent.classList.add("hidden");
  if (dom.part6BackendContent) dom.part6BackendContent.classList.remove("hidden");
  clearPart6FrontendView();
  renderPart6BackendMode("y3", `future${state.part6.horizonMonths || 12}m`);
}

function clearPart6FrontendView() {
  if (dom.metricP6Rows) dom.metricP6Rows.textContent = "-";
  if (dom.metricP6Positive) dom.metricP6Positive.textContent = "-";
  if (dom.metricP6HighPositive) dom.metricP6HighPositive.textContent = "-";
  if (dom.metricP6AvgFuture) dom.metricP6AvgFuture.textContent = "-";
  if (dom.part6CandidateTable) dom.part6CandidateTable.innerHTML = "";
  if (dom.part6BucketTable) dom.part6BucketTable.innerHTML = "";
  if (window.Plotly) {
    ["part6BucketChart", "part6ScatterChart", "part6PortfolioChart"].forEach(id => {
      const node = document.getElementById(id);
      if (node) Plotly.purge(node);
    });
  }
}

function renderPart6BackendMode(windowKey = "y3", targetKey = "future12m") {
  const mode = state.part6.mode || "backend";
  const baseText = mode === "shap"
    ? "SHAP mode：按 Run Backend Analysis 後，這裡會顯示模型預測與每個 event 的 top positive / negative features。"
    : "Backend handoff / ML result mode：按 Run Backend Analysis 後，這裡會顯示 Python backend 解析 Part1–Part5 後的 ML 預測。";
  const statusText = state.part6.backendStatus === "running"
    ? `正在送出 Part1~Part5 JSON 並執行 3Y backend ML / SHAP。${baseText}`
    : state.part6.backendStatus === "done"
      ? `已收到後端 ML / SHAP 結果。${baseText}`
      : `Part6 固定使用 3Y trailing、future 12M target。${baseText}`;
  if (dom.part6Status) dom.part6Status.textContent = statusText;
  if (state.part6.backendResult) {
    renderPart6BackendVisuals(state.part6.backendResult);
  }
}

function currentPart6RowsForBackend() {
  return [];
}

function buildPart6PredictionRows(windowKey, targetKey) {
  const horizonConfig = HORIZONS[windowKey] || HORIZONS.y3;
  const targetConfig = PART6_TARGETS[targetKey] || PART6_TARGETS.next1;
  const useFilter = state.selectedMgmt && state.selectedMgmt.size > 0;
  const part1Membership = buildPart6Part1Membership();
  const restrictToPart1 = part1Membership.size > 0;
  const part5FundNames = activePart5FundNameSetForPart6();
  const groups = groupBy(state.rows || [], row => row.crsp_fundno);
  const portfolioSignalMap = buildPart6PortfolioSignalMap();
  const targetMonths = targetConfig.months;
  const candidates = [];

  for (const group of groups.values()) {
    group.sort((a, b) => a.caldtMs - b.caldtMs);
    for (let i = 0; i < group.length; i += 1) {
      const base = group[i];
      if (useFilter && !state.selectedMgmt.has(base.mgmt_name)) continue;
      const part1Region = part1Membership.get(base.crsp_fundno) || "";
      if (restrictToPart1 && !part1Region) continue;
      if (part5FundNames.size && !part6FundNameMatchesPart5(base, part5FundNames)) continue;
      const horizon = base.horizons && base.horizons[windowKey];
      if (!horizon || !Number.isFinite(horizon.y)) continue;
      const future = part6ForwardReturn(group, i, targetMonths);
      if (!future || !Number.isFinite(future.futureReturn)) continue;
      const portfolioSignal = findPart6PortfolioSignal(base, portfolioSignalMap);
      candidates.push({
        fund: base.fund_name,
        fund_no: base.crsp_fundno,
        mgmt_name: base.mgmt_name,
        manager: base.mgr_name,
        date: base.caldt,
        dateMs: base.caldtMs,
        window: horizonConfig.label,
        target: targetConfig.label,
        part1_region: part1Region || "All",
        trailing_return: horizon.y,
        trailing_sp500: horizon.x,
        trailing_excess: horizon.y - horizon.x,
        trailing_drawdown: horizon.max_drawdown,
        avg_net_flow: horizon.avg_net_flow,
        mtna: horizon.mtna,
        exp_ratio: horizon.exp_ratio,
        turn_ratio: horizon.turn_ratio,
        tenure: horizon.tenure,
        portfolio_change_score: portfolioSignal ? portfolioSignal.changeScore : NaN,
        delta_stock: portfolioSignal ? portfolioSignal.deltaStock : NaN,
        delta_bond: portfolioSignal ? portfolioSignal.deltaBond : NaN,
        delta_cash: portfolioSignal ? portfolioSignal.deltaCash : NaN,
        delta_portfolio_beta: portfolioSignal ? portfolioSignal.deltaPortfolioBeta : NaN,
        portfolio_weighted_beta: portfolioSignal ? portfolioSignal.portfolioWeightedBeta : NaN,
        future_return: future.futureReturn,
        future_end_date: future.futureEndDate
      });
    }
  }

  const futureReturns = candidates.map(row => row.future_return).filter(Number.isFinite);
  const highThreshold = percentile(futureReturns, 0.75);
  const zTrailing = makeZScore(candidates, "trailing_return");
  const zExcess = makeZScore(candidates, "trailing_excess");
  const zDrawdown = makeZScore(candidates.map(row => ({ value: Math.abs(row.trailing_drawdown) })), "value");
  const zFlow = makeZScore(candidates, "avg_net_flow");
  const zTurnover = makeZScore(candidates, "turn_ratio");
  const zPortfolioChange = makeZScore(candidates, "portfolio_change_score");
  const zExpense = makeZScore(candidates, "exp_ratio");

  for (let i = 0; i < candidates.length; i += 1) {
    const row = candidates[i];
    row.future_positive = row.future_return > 0 ? "Yes" : "No";
    row.future_high_positive = row.future_return > 0 && row.future_return >= highThreshold ? "Yes" : "No";
    row.signal_score =
      0.34 * zTrailing(row.trailing_return) +
      0.24 * zExcess(row.trailing_excess) -
      0.16 * zDrawdown(Math.abs(row.trailing_drawdown)) +
      0.12 * zFlow(row.avg_net_flow) +
      0.10 * zTurnover(row.turn_ratio) +
      0.16 * zPortfolioChange(row.portfolio_change_score) -
      0.08 * zExpense(row.exp_ratio);
  }

  return candidates
    .filter(row => Number.isFinite(row.signal_score))
    .sort((a, b) => b.signal_score - a.signal_score || b.future_return - a.future_return);
}

function part6ForwardReturn(group, index, targetMonths) {
  if (targetMonths <= 1) {
    const next = group[index + 1];
    if (!next || !Number.isFinite(next.raw_mret)) return null;
    return { futureReturn: next.raw_mret, futureEndDate: next.caldt };
  }
  const values = [];
  for (let offset = 1; offset <= targetMonths; offset += 1) {
    const row = group[index + offset];
    if (!row || !Number.isFinite(row.raw_mret)) continue;
    values.push(row.raw_mret);
  }
  if (values.length < Math.max(3, Math.floor(targetMonths * 0.7))) return null;
  const end = group[Math.min(group.length - 1, index + values.length)];
  return { futureReturn: safeCompoundReturn(values), futureEndDate: end ? end.caldt : "" };
}


function buildPart6Part1Membership() {
  const membership = new Map();
  if (!state.p1Applied) return membership;
  for (const row of state.rawA || []) {
    if (row && row.crsp_fundno) membership.set(row.crsp_fundno, "A");
  }
  if (state.selectionMode === "compare") {
    for (const row of state.rawB || []) {
      if (!row || !row.crsp_fundno) continue;
      const old = membership.get(row.crsp_fundno);
      membership.set(row.crsp_fundno, old && old !== "B" ? "A&B" : "B");
    }
  }
  return membership;
}

function activePart5RestrictsPart6() {
  if (!state.part5.loaded) return false;
  if (isPart5ManagerMode()) return true;
  if (dom.part5PeriodSelect && dom.part5PeriodSelect.value !== "all") return true;
  if (dom.part5FocusSelect && dom.part5FocusSelect.value !== "all") return true;
  if (dom.part5SearchInput && cleanText(dom.part5SearchInput.value)) return true;
  if ((state.part5.brushedPeriodLabels || []).length) return true;
  if ((state.part5.brushedReportKeys || []).length) return true;
  if ((state.part5.brushedHoldingKeys || []).length) return true;
  if (state.part5.activeReportKey || state.part5.activeHoldingKey) return true;
  return false;
}

function activePart5FundNameSetForPart6() {
  const names = new Set();
  if (!activePart5RestrictsPart6()) return names;
  const reports = filterPart5Reports({ useFocus: true, useBrush: true });
  for (const report of reports || []) {
    const name = normalizeLooseText(report.fund_name);
    if (name) names.add(name);
  }
  const holdings = filterPart5Holdings({ useDetailFilters: true, useFocus: true, useBrush: true });
  for (const row of holdings || []) {
    const name = normalizeLooseText(row.fund_name);
    if (name) names.add(name);
  }
  return names;
}

function part6FundNameMatchesPart5(base, nameSet) {
  if (!nameSet || !nameSet.size) return true;
  const baseName = normalizeLooseText(base.fund_name);
  if (!baseName) return false;
  for (const name of nameSet) {
    if (baseName === name || baseName.includes(name) || name.includes(baseName)) return true;
  }
  return false;
}

function buildPart6PortfolioSignalMap() {
  const map = new Map();
  if (!state.part5.loaded || !state.part5.reports || !state.part5.reports.length) return map;
  const sourceReports = activePart5RestrictsPart6() ? filterPart5Reports({ useFocus: true, useBrush: true }) : state.part5.reports;
  const byPort = groupBy(sourceReports, row => row.crsp_portno || row.fund_name);
  for (const group of byPort.values()) {
    group.sort((a, b) => a.reportDateMs - b.reportDateMs);
    let prev = null;
    for (const report of group) {
      const signal = {
        reportDateMs: report.reportDateMs,
        report_dt: report.report_dt,
        fund_name: report.fund_name,
        portfolioWeightedBeta: report.portfolio_weighted_beta,
        deltaStock: prev && Number.isFinite(report.stockPct) && Number.isFinite(prev.stockPct) ? report.stockPct - prev.stockPct : NaN,
        deltaBond: prev && Number.isFinite(report.bondPct) && Number.isFinite(prev.bondPct) ? report.bondPct - prev.bondPct : NaN,
        deltaCash: prev && Number.isFinite(report.cashPct) && Number.isFinite(prev.cashPct) ? report.cashPct - prev.cashPct : NaN,
        deltaPortfolioBeta: prev && Number.isFinite(report.portfolio_weighted_beta) && Number.isFinite(prev.portfolio_weighted_beta) ? report.portfolio_weighted_beta - prev.portfolio_weighted_beta : NaN
      };
      signal.changeScore = sum([Math.abs(signal.deltaStock), Math.abs(signal.deltaBond), Math.abs(signal.deltaCash)]) +
        (Number.isFinite(signal.deltaPortfolioBeta) ? Math.abs(signal.deltaPortfolioBeta) * 0.05 : 0);
      const nameKey = normalizeLooseText(report.fund_name);
      if (nameKey) {
        if (!map.has(nameKey)) map.set(nameKey, []);
        map.get(nameKey).push(signal);
      }
      prev = report;
    }
  }
  return map;
}

function findPart6PortfolioSignal(base, signalMap) {
  if (!signalMap || !signalMap.size) return null;
  const key = normalizeLooseText(base.fund_name);
  const rows = signalMap.get(key);
  if (!rows || !rows.length) return null;
  let best = null;
  for (const row of rows) {
    if (row.reportDateMs <= base.caldtMs + 45 * MS_PER_DAY) best = row;
    else break;
  }
  if (!best || Math.abs(base.caldtMs - best.reportDateMs) > 540 * MS_PER_DAY) return null;
  return best;
}

function makeZScore(rows, key) {
  const values = rows.map(row => row[key]).filter(Number.isFinite);
  const avg = mean(values);
  const std = sampleStd(values);
  return value => Number.isFinite(value) && Number.isFinite(avg) && Number.isFinite(std) && std > 0 ? (value - avg) / std : 0;
}

function aggregatePart6Buckets(rows) {
  const ranked = (rows || []).filter(row => Number.isFinite(row.signal_score)).slice().sort((a, b) => a.signal_score - b.signal_score);
  if (!ranked.length) return [];
  const bucketCount = 5;
  const buckets = Array.from({ length: bucketCount }, (_, index) => ({ bucket: `Q${index + 1}`, scoreValues: [], futureValues: [], positiveFlags: [], highFlags: [], portfolioChanges: [], count: 0 }));
  ranked.forEach((row, index) => {
    const bucketIndex = Math.min(bucketCount - 1, Math.floor(index / ranked.length * bucketCount));
    const bucket = buckets[bucketIndex];
    bucket.count += 1;
    bucket.scoreValues.push(row.signal_score);
    bucket.futureValues.push(row.future_return);
    bucket.positiveFlags.push(row.future_return > 0 ? 1 : 0);
    bucket.highFlags.push(row.future_high_positive === "Yes" ? 1 : 0);
    if (Number.isFinite(row.portfolio_change_score)) bucket.portfolioChanges.push(row.portfolio_change_score);
  });
  return buckets.map(bucket => ({
    bucket: bucket.bucket,
    count: bucket.count,
    avg_signal_score: mean(bucket.scoreValues),
    avg_future_return: mean(bucket.futureValues),
    positive_rate: mean(bucket.positiveFlags),
    high_positive_rate: mean(bucket.highFlags),
    avg_portfolio_change: mean(bucket.portfolioChanges)
  }));
}

function updatePart6Metrics(rows) {
  if (!dom.metricP6Rows) return;
  dom.metricP6Rows.textContent = formatInt((rows || []).length);
  dom.metricP6Positive.textContent = formatPct(mean((rows || []).map(row => row.future_return > 0 ? 1 : 0)));
  dom.metricP6HighPositive.textContent = formatPct(mean((rows || []).map(row => row.future_high_positive === "Yes" ? 1 : 0)));
  dom.metricP6AvgFuture.textContent = formatPct(mean((rows || []).map(row => row.future_return)));
}

function drawPart6Charts(rows, buckets, windowKey, targetKey) {
  if (!window.Plotly) return;
  if (!rows || !rows.length) {
    ["part6BucketChart", "part6ScatterChart", "part6PortfolioChart"].forEach(id => Plotly.purge(id));
    return;
  }
  const horizonLabel = HORIZONS[windowKey] ? HORIZONS[windowKey].label : HORIZONS.y3.label;
  const targetLabel = PART6_TARGETS[targetKey] ? PART6_TARGETS[targetKey].label : targetKey;
  Plotly.react("part6BucketChart", [
    { type: "bar", x: buckets.map(row => row.bucket), y: buckets.map(row => row.positive_rate), name: "future positive rate", marker: { color: "#12a59a" }, hovertemplate: "%{x}<br>positive：%{y:.2%}<extra></extra>" },
    { type: "bar", x: buckets.map(row => row.bucket), y: buckets.map(row => row.high_positive_rate), name: "high positive rate", marker: { color: "#1677c2" }, hovertemplate: "%{x}<br>high positive：%{y:.2%}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x: buckets.map(row => row.bucket), y: buckets.map(row => row.avg_future_return), yaxis: "y2", name: "avg future return", line: { color: "#d88c18", width: 2.4 }, hovertemplate: "%{x}<br>avg future：%{y:.2%}<extra></extra>" }
  ], {
    title: `Part6：prediction score buckets (${horizonLabel} -> ${targetLabel})`,
    height: 390,
    margin: { l: 58, r: 58, t: 58, b: 58 },
    barmode: "group",
    legend: { orientation: "h", y: -0.22 },
    xaxis: { title: "score bucket, low to high" },
    yaxis: { title: "rate", tickformat: ".0%", range: [0, 1] },
    yaxis2: { title: "avg future return", overlaying: "y", side: "right", tickformat: ".0%", showgrid: false }
  }, { displaylogo: false, responsive: true });

  const scatterRows = rows.slice(0, 8000);
  Plotly.react("part6ScatterChart", [{
    type: "scattergl",
    mode: "markers",
    x: scatterRows.map(row => row.trailing_return),
    y: scatterRows.map(row => row.future_return),
    text: scatterRows.map(row => row.fund),
    customdata: scatterRows.map(row => [row.date, row.future_end_date, row.signal_score, row.future_high_positive, row.portfolio_change_score]),
    marker: {
      size: scatterRows.map(row => Number.isFinite(row.portfolio_change_score) ? Math.max(6, Math.min(18, 6 + row.portfolio_change_score * 28)) : 7),
      color: scatterRows.map(row => row.signal_score),
      colorscale: "Viridis",
      showscale: true,
      colorbar: { title: "score", thickness: 12 },
      opacity: 0.68,
      line: { color: "rgba(20,45,64,0.28)", width: 0.4 }
    },
    hovertemplate: "%{text}<br>%{customdata[0]} -> %{customdata[1]}<br>trailing：%{x:.2%}<br>future：%{y:.2%}<br>score：%{customdata[2]:.3f}<br>high positive：%{customdata[3]}<br>portfolio change：%{customdata[4]:.4f}<extra></extra>"
  }], {
    title: "Part6：trailing return vs future return",
    height: 420,
    margin: { l: 62, r: 58, t: 58, b: 58 },
    xaxis: { title: `${horizonLabel} trailing annual return`, tickformat: ".0%", zeroline: false },
    yaxis: { title: `${targetLabel} future return`, tickformat: ".0%", zeroline: false },
    showlegend: false
  }, { displaylogo: false, responsive: true, scrollZoom: true });

  const portfolioRows = rows.filter(row => Number.isFinite(row.portfolio_change_score));
  if (!portfolioRows.length) {
    Plotly.react("part6PortfolioChart", [], {
      title: "Part6：portfolio-change signal vs future return",
      height: 380,
      margin: { l: 58, r: 28, t: 58, b: 58 },
      annotations: [{ text: "載入 Part5 後，如果基金名稱可與 Part1 對上，這裡會顯示投組變化訊號。", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false }]
    }, { displaylogo: false, responsive: true });
    return;
  }
  Plotly.react("part6PortfolioChart", [{
    type: "scattergl",
    mode: "markers",
    x: portfolioRows.map(row => row.portfolio_change_score),
    y: portfolioRows.map(row => row.future_return),
    text: portfolioRows.map(row => row.fund),
    customdata: portfolioRows.map(row => [row.delta_stock, row.delta_bond, row.delta_cash, row.delta_portfolio_beta, row.signal_score]),
    marker: {
      size: 9,
      color: portfolioRows.map(row => row.signal_score),
      colorscale: "Portland",
      showscale: true,
      colorbar: { title: "score", thickness: 12 },
      opacity: 0.72
    },
    hovertemplate: "%{text}<br>portfolio change：%{x:.4f}<br>future：%{y:.2%}<br>delta stock/bond/cash：%{customdata[0]:.2%} / %{customdata[1]:.2%} / %{customdata[2]:.2%}<br>delta beta：%{customdata[3]:.4f}<extra></extra>"
  }], {
    title: "Part6：基金投組變化 signal vs future return",
    height: 380,
    margin: { l: 58, r: 58, t: 58, b: 58 },
    xaxis: { title: "portfolio change score", zeroline: false },
    yaxis: { title: `${targetLabel} future return`, tickformat: ".0%", zeroline: false },
    showlegend: false
  }, { displaylogo: false, responsive: true, scrollZoom: true });
}

function renderPart6Tables(rows, buckets, windowKey, targetKey) {
  const horizonLabel = HORIZONS[windowKey] ? HORIZONS[windowKey].label : HORIZONS.y3.label;
  const targetLabel = PART6_TARGETS[targetKey] ? PART6_TARGETS[targetKey].label : PART6_TARGETS.next1.label;
  renderTable("part6CandidateTable", (rows || []).slice(0, 120), [
    { key: "fund", label: "基金" },
    { key: "part1_region", label: "Part1 A/B" },
    { key: "date", label: "訊號日期" },
    { key: "future_end_date", label: "未來標籤日" },
    { key: "trailing_return", label: "Trailing return", format: "pct" },
    { key: "trailing_excess", label: "Trailing excess", format: "pct" },
    { key: "future_return", label: "Future return", format: "pct" },
    { key: "future_positive", label: "正報酬" },
    { key: "future_high_positive", label: "High positive" },
    { key: "signal_score", label: "Prediction score", format: "num" },
    { key: "portfolio_change_score", label: "Portfolio change", format: "num" },
    { key: "delta_stock", label: "Delta stock", format: "pct" },
    { key: "delta_bond", label: "Delta bond", format: "pct" }
  ], { title: `Part6 Top prediction candidates（${horizonLabel} -> ${targetLabel}）`, expanded: true, count: (rows || []).length });
  renderTable("part6BucketTable", buckets || [], [
    { key: "bucket", label: "Score bucket" },
    { key: "count", label: "樣本數", format: "int" },
    { key: "avg_signal_score", label: "平均score", format: "num" },
    { key: "positive_rate", label: "正報酬率", format: "pct" },
    { key: "high_positive_rate", label: "High positive率", format: "pct" },
    { key: "avg_future_return", label: "平均未來報酬", format: "pct" },
    { key: "avg_portfolio_change", label: "平均投組變化", format: "num" }
  ], { title: "Part6 score bucket backtest summary", expanded: false, count: (buckets || []).length });
}


function uniqueValues(rows, keyFn, limit = 500) {
  const out = [];
  const seen = new Set();
  for (const row of rows || []) {
    const value = keyFn(row);
    if (value == null || value === "" || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
    if (out.length >= limit) break;
  }
  return out;
}

function compactRowsForBackend(rows, columns, limit = 200) {
  return (rows || []).slice(0, limit).map(row => {
    const out = {};
    for (const key of columns) out[key] = row[key];
    return out;
  });
}

function currentPart6RowsForBackend() {
  return [];
}


function rowsForBackend(rows, columns, limit = null) {
  const source = Array.isArray(rows) ? rows : [];
  const picked = Number.isFinite(limit) ? source.slice(0, limit) : source;
  return picked.map(row => {
    const out = {};
    for (const key of columns) out[key] = row ? row[key] : null;
    return out;
  });
}

function buildPart1PointRowsForBackend(rows, regionLabel) {
  return rowsForBackend(rows, [
    "rowId", "horizon", "horizon_label", "horizon_months", "caldt", "monthKey", "year", "yearFloat",
    "crsp_fundno", "fund_name", "mgmt_name", "mgr_name", "x_ret", "y_ret", "sp500_ret", "mret",
    "excess_ret", "period_return", "window_max_drawdown", "net_flow", "avg_net_flow", "mtna", "exp_ratio",
    "mgmt_fee", "turn_ratio", "age", "tenure", "current_mret", "current_sp500_ret", "current_excess_ret",
    "current_net_flow", "current_mtna", "current_exp_ratio", "current_mgmt_fee", "current_turn_ratio",
    "current_age", "current_tenure", "fund_trailing_return", "sp500_trailing_return", "fund_trailing_excess_return",
    "fund_trailing_period_return", "fund_trailing_max_drawdown", "trailing_avg_net_flow", "trailing_sum_net_flow",
    "trailing_avg_mtna", "trailing_avg_exp_ratio", "trailing_avg_mgmt_fee", "trailing_avg_turn_ratio",
    "trailing_avg_age", "trailing_avg_tenure", "policy", "lipper_class_name", "window_count"
  ]).map(row => ({
    ...row,
    part1_region: regionLabel,
    x_meaning: state.horizon === "monthly" ? "S&P500 monthly return" : `S&P500 ${HORIZONS[state.horizon].label} annualized trailing return`,
    y_meaning: state.horizon === "monthly" ? "Fund monthly return" : `Fund ${HORIZONS[state.horizon].label} annualized trailing return`,
    color_value: row.yearFloat,
    color_meaning: "year / observation time"
  }));
}

function buildPart2DetailPayload() {
  const compareMode = state.selectionMode === "compare" && (state.rawB || []).length > 0;
  const baseA = buildLevelTables(state.rawA || []);
  const baseB = compareMode ? buildLevelTables(state.rawB || []) : emptyLevelTables();
  const selectedRawA = (state.part3RawA && state.part3RawA.length) ? state.part3RawA : [];
  const selectedRawB = compareMode && state.part3RawB && state.part3RawB.length ? state.part3RawB : [];
  const selectedA = selectedRawA.length ? buildLevelTables(selectedRawA) : emptyLevelTables();
  const selectedB = selectedRawB.length ? buildLevelTables(selectedRawB) : emptyLevelTables();
  return {
    logic: state.part2Logic,
    horizon: state.horizon,
    horizon_title: HORIZONS[state.horizon].title,
    selection_mode: state.selectionMode,
    regions: (state.appliedP2Regions || []).map(region => ({
      level: region.level,
      feature: region.feature,
      label: region.label,
      xRange: region.xRange
    })),
    tables: {
      monthly_A: rowsForBackend(selectedA.monthly, part2BackendColumns("monthly")),
      monthly_B: rowsForBackend(selectedB.monthly, part2BackendColumns("monthly")),
      fund_level_A: rowsForBackend(selectedA.fund, part2BackendColumns("fund")),
      fund_level_B: rowsForBackend(selectedB.fund, part2BackendColumns("fund")),
      family_level_A: rowsForBackend(selectedA.family, part2BackendColumns("family")),
      family_level_B: rowsForBackend(selectedB.family, part2BackendColumns("family"))
    },
    base_tables_before_part2_filter: {
      monthly_A_count: (baseA.monthly || []).length,
      monthly_B_count: (baseB.monthly || []).length,
      fund_level_A_count: (baseA.fund || []).length,
      fund_level_B_count: (baseB.fund || []).length,
      family_level_A_count: (baseA.family || []).length,
      family_level_B_count: (baseB.family || []).length
    }
  };
}

function part2BackendColumns(levelKey) {
  const base = (PART2_TABLE_BASE_COLUMNS[levelKey] || []).map(col => col.key);
  const features = (FEATURES[levelKey] || []).map(([key]) => key);
  return Array.from(new Set(base.concat(features)));
}

function buildPart3DetailPayload() {
  const compareMode = state.selectionMode === "compare" && (state.part3RawB || []).length > 0;
  const countA = countManagers(state.part3RawA || []);
  const countB = compareMode ? countManagers(state.part3RawB || []) : new Map();
  const managerRows = mergeManagerCounts(countA, countB, compareMode);
  return {
    horizon: state.horizon,
    horizon_title: HORIZONS[state.horizon].title,
    selectedManagers: Array.from(state.pendingManagers || []),
    latestManagers: state.latestManagers || [],
    part3A_count: (state.part3RawA || []).length,
    part3B_count: (state.part3RawB || []).length,
    manager_detail: rowsForBackend(managerRows, ["manager", "countA", "countB", "total", "group", "color"])
  };
}


function buildStyleDriftRowsForBackend() {
  const managers = (state.radarRecords || []).map(record => record.manager);
  if (!managers.length) return [];
  const managerSet = new Set(managers);
  const rows = (state.activeRows || []).filter(row => managerSet.has(row.mgr_name));
  const groups = groupBy(rows, row => `${row.mgr_name}__${row.year}`);
  const out = [];
  for (const [key, group] of groups.entries()) {
    const [manager, yearText] = key.split("__");
    const returns = finiteValues(group, "mret");
    const excess = group.filter(row => Number.isFinite(row.mret) && Number.isFinite(row.sp500_ret)).map(row => row.mret - row.sp500_ret);
    out.push({
      manager,
      year: Number(yearText),
      obs: group.length,
      annual_return: mean(returns),
      avg_excess: excess.length ? mean(excess) : NaN,
      annual_volatility: sampleStd(returns),
      avg_fee: mean(finiteValues(group, "exp_ratio")),
      avg_flow: mean(finiteValues(group, "net_flow")),
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_turnover: mean(finiteValues(group, "turn_ratio"))
    });
  }
  out.sort((a, b) => (a.manager || "").localeCompare(b.manager || "") || a.year - b.year);
  return out;
}

function drawStyleDriftChart() {
  if (!window.Plotly || !dom.styleDriftChart) return;
  const rows = buildStyleDriftRowsForBackend();
  if (!rows.length) {
    Plotly.react("styleDriftChart", [], { title: "Part4 Style Drift：請先加入經理人", height: 460 }, { displaylogo: false });
    return;
  }
  const managers = Array.from(new Set(rows.map(row => row.manager))).slice(0, 20);
  const traces = managers.map(manager => {
    const g = rows.filter(row => row.manager === manager);
    return {
      type: "scatter",
      mode: "lines+markers",
      x: g.map(row => row.year),
      y: g.map(row => row.annual_return),
      name: manager,
      customdata: g.map(row => [row.avg_excess, row.annual_volatility, row.avg_fee, row.obs]),
      hovertemplate: "經理人：%{fullData.name}<br>年份：%{x}<br>年化報酬：%{y:.2%}<br>平均超額：%{customdata[0]:.2%}<br>波動：%{customdata[1]:.2%}<br>費用：%{customdata[2]:.2%}<br>觀測數：%{customdata[3]}<extra></extra>"
    };
  });
  Plotly.react("styleDriftChart", traces, {
    title: "Part4：Style Drift timeline（依年度觀察經理人風格變化）",
    height: 520,
    margin: { l: 64, r: 28, t: 64, b: 52 },
    xaxis: { title: "年份" },
    yaxis: { title: "3Y trailing annualized return", tickformat: ".1%" },
    hovermode: "closest"
  }, { displaylogo: false, responsive: true });
}

function buildPart4DetailPayload() {
  const groups = groupRadarRecords(state.radarRecords || []);
  const groupRows = [];
  groups.forEach((group, index) => {
    groupRows.push({
      group: `Group ${index + 1}`,
      count: group.length,
      managers: group.map(record => record.manager).join("、")
    });
  });
  const recordRows = [];
  groups.forEach((group, groupIndex) => {
    group.forEach(record => {
      const row = {
        group: `Group ${groupIndex + 1}`,
        manager: record.manager,
        sourceLabel: record.sourceLabel
      };
      for (const metric of RADAR_METRICS) {
        row[`score_${metric.rawKey}`] = record.scores ? record.scores[metric.label] : null;
        row[metric.rawKey] = record.raw ? record.raw[metric.rawKey] : null;
      }
      recordRows.push(row);
    });
  });
  const styleDriftRows = buildStyleDriftRowsForBackend();
  return {
    view: state.part4View,
    group_rows: groupRows,
    style_drift_rows: rowsForBackend(styleDriftRows, ["manager", "year", "obs", "annual_return", "avg_excess", "annual_volatility", "avg_fee", "avg_flow", "avg_mtna", "avg_turnover"]),
    score_and_raw_indicator_rows: recordRows,
    manager_records_raw: (state.radarRecords || []).map(record => ({
      manager: record.manager,
      sourceLabel: record.sourceLabel,
      scores: record.scores,
      raw: record.raw
    }))
  };
}

function buildPart5PeriodSummaryRowsForBackend(reports, holdings) {
  const map = new Map();
  for (const report of reports || []) {
    if (!map.has(report.periodKey)) map.set(report.periodKey, { periodKey: report.periodKey, period: report.periodLabel, years: [], funds: new Set(), reportCount: 0, holdingCount: 0, betaHoldingCount: 0, stockValues: [], bondValues: [], cashValues: [], yieldValues: [], portfolioBetaValues: [] });
    const row = map.get(report.periodKey);
    row.reportCount += 1;
    row.years.push(report.year);
    if (report.crsp_portno) row.funds.add(report.crsp_portno);
    if (Number.isFinite(report.stockPct)) row.stockValues.push(report.stockPct);
    if (Number.isFinite(report.bondPct)) row.bondValues.push(report.bondPct);
    if (Number.isFinite(report.cashPct)) row.cashValues.push(report.cashPct);
    if (Number.isFinite(report.yield10y)) row.yieldValues.push(report.yield10y);
    if (Number.isFinite(report.portfolio_weighted_beta)) row.portfolioBetaValues.push(report.portfolio_weighted_beta);
  }
  for (const holding of holdings || []) {
    const row = map.get(holding.periodKey);
    if (row) {
      row.holdingCount += 1;
      if (holding.hasStockBeta) row.betaHoldingCount += 1;
    }
  }
  return Array.from(map.values()).map(row => ({
    period: row.period,
    yearRange: row.years.length ? `${Math.min(...row.years)}-${Math.max(...row.years)}` : "-",
    reportCount: row.reportCount,
    fundCount: row.funds.size,
    holdingCount: row.holdingCount,
    betaHoldingCount: row.betaHoldingCount,
    avgYield: mean(row.yieldValues),
    avgStock: mean(row.stockValues),
    avgBond: mean(row.bondValues),
    avgCash: mean(row.cashValues),
    avgPortfolioWeightedBeta: mean(row.portfolioBetaValues)
  }));
}


function holdingActionIdentity(row) {
  const ticker = cleanText(row.holding_ticker || row.yahoo_ticker).toUpperCase();
  const name = cleanText(row.holding_security_name || row.primary_security_name);
  const fallback = cleanText(row.crsp_company_key || row.holding_permno || name).toUpperCase();
  return ticker || fallback;
}

function aggregateHoldingsByIdentity(rows) {
  const map = new Map();
  for (const row of rows || []) {
    const key = holdingActionIdentity(row);
    if (!key) continue;
    if (!map.has(key)) {
      map.set(key, {
        holding_key: key,
        holding_ticker: cleanText(row.holding_ticker || row.yahoo_ticker || key),
        holding_security_name: cleanText(row.holding_security_name || row.primary_security_name || key),
        sector: cleanText(row.sector) || "Unknown",
        industry: cleanText(row.industry) || "Unknown",
        holdingPct: 0,
        weighted_beta: 0,
        stock_beta_values: [],
        security_rank_values: [],
        rows: []
      });
    }
    const item = map.get(key);
    if (Number.isFinite(row.holdingPct)) item.holdingPct += row.holdingPct;
    if (Number.isFinite(row.weighted_beta)) item.weighted_beta += row.weighted_beta;
    if (Number.isFinite(row.stock_beta)) item.stock_beta_values.push(row.stock_beta);
    if (Number.isFinite(row.security_rank)) item.security_rank_values.push(row.security_rank);
    item.rows.push(row);
  }
  for (const item of map.values()) {
    item.stock_beta = mean(item.stock_beta_values);
    item.security_rank = item.security_rank_values.length ? Math.min(...item.security_rank_values) : NaN;
  }
  return map;
}

function buildPart5StockActionRowsForBackend(selectedReportKeys) {
  const selectedKeySet = new Set(selectedReportKeys || []);
  if (!state.part5.loaded || !selectedKeySet.size) return [];

  const reportsByPort = groupBy(state.part5.reports || [], report => String(report.crsp_portno || ""));
  const holdingsByReport = groupBy(state.part5.holdings || [], row => row.reportKey);
  const selectedHoldingKeySet = new Set((state.part5.brushedHoldingKeys && state.part5.brushedHoldingKeys.length)
    ? state.part5.brushedHoldingKeys.map(key => cleanText(key).toUpperCase())
    : (state.part5.activeHoldingKey ? [cleanText(state.part5.activeHoldingKey).toUpperCase()] : []));
  const deltaThreshold = 0.002; // 20 bps of TNA; avoid tiny noise.
  const out = [];

  for (const [portno, reports] of reportsByPort.entries()) {
    reports.sort((a, b) => a.reportDateMs - b.reportDateMs);
    for (let i = 0; i < reports.length; i += 1) {
      const currentReport = reports[i];
      if (!selectedKeySet.has(currentReport.reportKey)) continue;
      const previousReport = reports.slice(0, i).reverse().find(report => report.crsp_portno === currentReport.crsp_portno);
      if (!previousReport) continue;

      const currentMap = aggregateHoldingsByIdentity(holdingsByReport.get(currentReport.reportKey) || []);
      const previousMap = aggregateHoldingsByIdentity(holdingsByReport.get(previousReport.reportKey) || []);
      const keys = new Set([...currentMap.keys(), ...previousMap.keys()]);

      for (const key of keys) {
        const cur = currentMap.get(key) || {};
        const prev = previousMap.get(key) || {};
        const currentPct = Number.isFinite(cur.holdingPct) ? cur.holdingPct : 0;
        const previousPct = Number.isFinite(prev.holdingPct) ? prev.holdingPct : 0;
        const delta = currentPct - previousPct;
        if (Math.abs(delta) < deltaThreshold && !(previousPct === 0 && currentPct > 0)) continue;

        const direction = previousPct === 0 && currentPct > 0
          ? "new_position"
          : (delta > 0 ? "increase" : "decrease");
        const betaNow = Number.isFinite(cur.stock_beta) ? cur.stock_beta : prev.stock_beta;
        const weightedBetaDelta = Number.isFinite(betaNow) ? delta * betaNow : NaN;
        const ticker = cleanText(cur.holding_ticker || prev.holding_ticker || key).toUpperCase();
        const name = cleanText(cur.holding_security_name || prev.holding_security_name || key);

        out.push({
          reportKey: currentReport.reportKey,
          previousReportKey: previousReport.reportKey,
          crsp_portno: currentReport.crsp_portno,
          fund_ticker: currentReport.fund_ticker,
          fund_name: currentReport.fund_name,
          report_dt: currentReport.report_dt,
          previous_report_dt: previousReport.report_dt,
          year: currentReport.year,
          quarter: currentReport.quarter,
          monthKey: currentReport.monthKey,
          holding_key: key,
          holding_ticker: ticker,
          holding_security_name: name,
          sector: cleanText(cur.sector || prev.sector) || "Unknown",
          industry: cleanText(cur.industry || prev.industry) || "Unknown",
          previous_holding_pct: previousPct,
          current_holding_pct: currentPct,
          delta_holding_pct: delta,
          stock_action_direction: direction,
          stock_beta: betaNow,
          weighted_beta_delta: weightedBetaDelta,
          current_security_rank: cur.security_rank,
          previous_security_rank: prev.security_rank,
          is_user_brushed_stock: selectedHoldingKeySet.has(ticker) || selectedHoldingKeySet.has(key)
        });
      }
    }
  }

  out.sort((a, b) => Math.abs(b.delta_holding_pct) - Math.abs(a.delta_holding_pct));
  return out;
}

function buildPart5DetailPayload() {
  const reports = state.part5.loaded ? filterPart5Reports({ useFocus: true, useBrush: true }) : [];
  const holdingsForSummary = state.part5.loaded ? filterPart5Holdings({ useDetailFilters: false, useFocus: true, useBrush: true }) : [];
  const detailRows = state.part5.loaded ? filterPart5Holdings({ useDetailFilters: true, useFocus: true, useBrush: true }) : [];
  const selectedReportKeys = state.part5.brushedReportKeys && state.part5.brushedReportKeys.length
    ? [...state.part5.brushedReportKeys]
    : (state.part5.activeReportKey ? [state.part5.activeReportKey] : []);
  const selectedReports = part5FilterReportsByKeys(reports, selectedReportKeys);
  const selectedReportHoldingRows = selectedReportKeys.length
    ? state.part5.holdings.filter(row => selectedReportKeys.includes(row.reportKey))
    : [];
  const selectedHoldingKeys = state.part5.brushedHoldingKeys && state.part5.brushedHoldingKeys.length
    ? [...state.part5.brushedHoldingKeys]
    : (state.part5.activeHoldingKey ? [state.part5.activeHoldingKey] : []);
  const selectedTopHoldingRows = selectedHoldingKeys.length
    ? part5FilterHoldingsByKeys(holdingsForSummary, selectedHoldingKeys)
    : [];
  const part5BItems = filterPart5BItems(state.part5.excludedTopRows || []);
  const stockActionRows = buildPart5StockActionRowsForBackend(selectedReportKeys);
  const increasedStockActionRows = stockActionRows.filter(row => row.stock_action_direction === "increase" || row.stock_action_direction === "new_position");
  const decreasedStockActionRows = stockActionRows.filter(row => row.stock_action_direction === "decrease");
  return {
    loaded: state.part5.loaded,
    analysisMode: state.part5.analysisMode,
    period: dom.part5PeriodSelect ? dom.part5PeriodSelect.value : "all",
    aggregation: dom.part5AggregationSelect ? dom.part5AggregationSelect.value : "year",
    focus: dom.part5FocusSelect ? dom.part5FocusSelect.value : "all",
    rankMax: dom.part5RankInput ? dom.part5RankInput.value : "10",
    search: dom.part5SearchInput ? dom.part5SearchInput.value : "",
    activeReportKey: state.part5.activeReportKey,
    activeHoldingKey: state.part5.activeHoldingKey,
    brushedPeriodLabels: [...(state.part5.brushedPeriodLabels || [])],
    brushedReportKeys: [...(state.part5.brushedReportKeys || [])],
    brushedHoldingKeys: [...(state.part5.brushedHoldingKeys || [])],
    selectedManagerNames: [...(state.part5.selectedManagerNames || [])],
    reports,
    period_summary: buildPart5PeriodSummaryRowsForBackend(reports, holdingsForSummary),
    holdings_detail_all: detailRows,
    selected_fund_reports: selectedReports,
    selected_fund_report_holding_detail_all: selectedReportHoldingRows,
    top_holding_detail_all: selectedTopHoldingRows,
    brushed_top_holding_all_detail: selectedTopHoldingRows,
    stock_action_rows: stockActionRows,
    increased_stock_action_rows: increasedStockActionRows.slice(0, 300),
    decreased_stock_action_rows: decreasedStockActionRows.slice(0, 300),
    part5b_top_non_individual_exposure_items: part5BItems,
    excluded_summary: summarizePart5BItems(filterPart5BItems(state.part5.excludedEnrichedRows || [])),
    excluded_top_items: part5BItems,
    excluded_removed_audit_items: filterPart5BItems(state.part5.excludedRemovedRows || [], { ignoreUseFlag: true })
  };
}


function backendFeatureContext() {
  return {
    horizon: "y3",
    horizon_title: "Trailing 3年",
    training_window_years: 3,
    training_window_months: 36,
    targets: ["future_3m_excess_return", "future_6m_excess_return", "future_9m_excess_return", "future_12m_excess_return"],
    target_horizon_months: state.part6.horizonMonths || 12,
    current_month_feature_keys: ["current_mret", "current_sp500_ret", "current_excess_ret", "current_net_flow", "current_mtna", "current_exp_ratio", "current_mgmt_fee", "current_turn_ratio", "current_age", "current_tenure"],
    trailing_feature_keys: ["fund_trailing_return", "sp500_trailing_return", "fund_trailing_excess_return", "fund_trailing_period_return", "fund_trailing_max_drawdown", "fund_trailing_beta_vs_sp500", "trailing_avg_net_flow", "trailing_sum_net_flow", "trailing_avg_mtna", "trailing_avg_exp_ratio", "trailing_avg_mgmt_fee", "trailing_avg_turn_ratio", "trailing_avg_age", "trailing_avg_tenure"],
    action_feature_keys: ["stock_allocation", "bond_allocation", "cash_allocation", "portfolio_beta", "technology_exposure", "bond_money_exposure", "indirect_equity_exposure", "delta_stock", "delta_beta", "delta_technology", "delta_bond_money", "delta_indirect_equity", "delta_nonstock_total_exposure", "delta_sector_exposure", "rolling_style_deviation_score", "rolling_sector_deviation_score", "rolling_cross_asset_deviation_score", "rolling_action_deviation_score"]
  };
}

function buildVisualStatePayload() {
  const featureContext = backendFeatureContext();
  const part1A = buildPart1PointRowsForBackend(state.rawA || [], "A");
  const part1B = buildPart1PointRowsForBackend(state.rawB || [], "B");
  const part2Payload = buildPart2DetailPayload();
  const part3Payload = buildPart3DetailPayload();
  const part4Payload = buildPart4DetailPayload();
  const part5Payload = buildPart5DetailPayload();
  const part6Rows = currentPart6RowsForBackend();
  return {
    context: {
      horizon: "y3",
      horizonTitle: HORIZONS.y3.title,
      selectionMode: state.selectionMode,
      timestamp: new Date().toISOString(),
      backend_feature_context: featureContext
    },
    part1: {
      backend_feature_context: featureContext,
      boxA: state.p1BoxA,
      boxB: state.p1BoxB,
      rawA_count: (state.rawA || []).length,
      rawB_count: (state.rawB || []).length,
      selectedFundIdsA: uniqueValues(state.rawA, row => row.crsp_fundno, 1000000),
      selectedFundIdsB: uniqueValues(state.rawB, row => row.crsp_fundno, 1000000),
      selected_points_A: part1A,
      selected_points_B: part1B,
      point_encoding: {
        x: state.horizon === "monthly" ? "S&P500 monthly return" : `S&P500 ${HORIZONS[state.horizon].label} annualized trailing return`,
        y: state.horizon === "monthly" ? "Fund monthly return" : `Fund ${HORIZONS[state.horizon].label} annualized trailing return`,
        color: "year / observation time",
        hover_metadata: ["fund_name", "mgmt_name", "mgr_name", "caldt", "crsp_fundno"]
      }
    },
    part2: { backend_feature_context: featureContext, ...part2Payload },
    part3: { backend_feature_context: featureContext, ...part3Payload },
    part4: { backend_feature_context: featureContext, ...part4Payload },
    part5: { backend_feature_context: featureContext, ...part5Payload },
    part6: {
      backend_feature_context: featureContext,
      mode: state.part6.mode || "backend",
      window: "y3",
      target: `future${state.part6.horizonMonths || 12}m`,
      horizon_months: state.part6.horizonMonths || 12,
      date_start: state.part6.dateStart || null,
      date_end: state.part6.dateEnd || null,
      candidates: []
    }
  };
}

async function runBackendAnalysis() {
  if (!dom.backendAnalysisStatus) return;
  const payload = buildVisualStatePayload();
  state.part6.lastPayload = payload;
  console.log("Visual state payload:", payload);
  console.log("Visual state payload JSON:", JSON.stringify(payload, null, 2));
  state.part6.mode = "backend";
  state.part6.backendStatus = "running";
  if (dom.part6ModeSelect) dom.part6ModeSelect.value = "backend";
  renderPart6();
  dom.backendAnalysisStatus.className = "inline-status backend-status";
  dom.backendAnalysisStatus.textContent = "正在 POST Part1–Part5 state 到 FastAPI...";
  if (dom.runBackendAnalysisBtn) dom.runBackendAnalysisBtn.disabled = true;
  try {
    const response = await fetch("/api/ml/analyze-visual-state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    console.log("Backend analysis result:", result);
    const hasPredictions = Boolean(result && result.ml_result && Array.isArray(result.ml_result.predictions) && result.ml_result.predictions.length);
    const backendWarnings = Array.isArray(result && result.warnings) ? result.warnings.filter(Boolean) : [];
    state.part6.backendStatus = hasPredictions ? "done" : "error";
    state.part6.backendResult = result;
    dom.backendAnalysisStatus.className = hasPredictions
      ? `inline-status backend-status ${backendWarnings.length ? "warning" : "success"}`
      : "inline-status backend-status error";
    dom.backendAnalysisStatus.textContent = [
      result.message || "後端已收到並解析 visual state。",
      ...backendWarnings
    ].join(" ");
    renderBackendAnalysisSummary(result);
    renderPart6();
    part7PopulateEvents();
  } catch (error) {
    state.part6.backendStatus = "error";
    dom.backendAnalysisStatus.className = "inline-status backend-status error";
    dom.backendAnalysisStatus.textContent = `Backend analysis 失敗：${error.message || error}`;
  } finally {
    if (dom.runBackendAnalysisBtn) dom.runBackendAnalysisBtn.disabled = false;
  }
}

function renderBackendAnalysisSummary(result) {
  if (!dom.backendAnalysisSummary) return;

  // 先把 ML prediction / SHAP 用圖表畫出來，再保留原本的 raw summary table。
  renderPart6BackendVisuals(result);

  const rows = [];
  const summary = result && result.received_summary ? result.received_summary : {};
  Object.entries(summary).forEach(([key, value]) => rows.push({ section: "received_summary", field: key, value: Array.isArray(value) ? value.join(", ") : value }));
  const ml = result && result.ml_result ? result.ml_result : null;
  if (ml && Array.isArray(ml.predictions)) {
    rows.push({ section: "ml", field: "prediction_count", value: ml.predictions.length });
    rows.push({ section: "ml", field: "model_path", value: ml.model_path || "" });
    rows.push({ section: "ml", field: "feature_count", value: ml.feature_count || "" });
    ml.predictions.slice(0, 20).forEach((p, i) => {
      const h = state.part6.horizonMonths || 12;
      rows.push({ section: "prediction", field: `${i + 1}. ${p.manager || ""} ${p.report_date || ""}`, value: `prob=${formatNumber(p[`positive_probability_${h}m`], 3)} class=${p[`predicted_class_${h}m`] || ""}` });
    });
  }
  const shap = result && result.shap_result ? result.shap_result : null;
  if (shap && Array.isArray(shap.explanations)) {
    rows.push({ section: "shap", field: "explanation_count", value: shap.explanations.length });
    shap.explanations.slice(0, 10).forEach((e, i) => {
      const top = (e.features || []).slice(0, 4).map(x => `${x.feature}:${formatNumber(x.contribution, 3)}`).join("; ");
      rows.push({ section: "shap", field: `${i + 1}. ${e.event_id || "event"} ${e.horizon_months || ""}M`, value: top });
    });
  }
  if (result && result.warnings && result.warnings.length) {
    result.warnings.forEach((w, i) => rows.push({ section: "warning", field: `warning_${i + 1}`, value: w }));
  }
  renderTable("backendAnalysisSummary", rows, [
    { key: "section", label: "Section" },
    { key: "field", label: "Field" },
    { key: "value", label: "Value" }
  ], { title: "Backend：四天期模型與 TreeSHAP 回應摘要", expanded: false, count: rows.length });
}

function asPercentPoint(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function backendPredictions(result) {
  const preds = result && result.ml_result && Array.isArray(result.ml_result.predictions)
    ? result.ml_result.predictions
    : [];
  return preds
    .map((p, i) => ({
      ...p,
      rank: i + 1,
      prediction_probability: asPercentPoint(p.prediction_probability),
      event_label: `${p.manager || "Unknown manager"} | ${p.report_date || ""}`,
      short_label: `${i + 1}. ${(p.manager || "Unknown").slice(0, 18)} ${p.report_date || ""}`
    }))
    .filter(p => Number.isFinite(p.prediction_probability));
}

function backendShapRows(result) {
  const explanations = result && result.shap_result && Array.isArray(result.shap_result.explanations)
    ? result.shap_result.explanations
    : [];
  const rows = [];
  explanations.forEach((event, eventIndex) => {
    const eventLabel = `${event.manager || "Unknown manager"} | ${event.report_date || ""}`;
    (event.top_positive || []).forEach(item => rows.push({
      event_index: eventIndex,
      event_id: event.event_id,
      event_label: eventLabel,
      manager: event.manager,
      fund: event.fund,
      report_date: event.report_date,
      direction: "positive",
      feature: item.feature,
      value: item.value,
      contribution: Number(item.contribution)
    }));
    (event.top_negative || []).forEach(item => rows.push({
      event_index: eventIndex,
      event_id: event.event_id,
      event_label: eventLabel,
      manager: event.manager,
      fund: event.fund,
      report_date: event.report_date,
      direction: "negative",
      feature: item.feature,
      value: item.value,
      contribution: Number(item.contribution)
    }));
  });
  return rows.filter(row => Number.isFinite(row.contribution));
}

function aggregateBackendShap(rows, topN = 14) {
  const map = new Map();
  rows.forEach(row => {
    if (!map.has(row.feature)) {
      map.set(row.feature, { feature: row.feature, total_abs_contribution: 0, mean_contribution: 0, positive_sum: 0, negative_sum: 0, count: 0 });
    }
    const item = map.get(row.feature);
    item.total_abs_contribution += Math.abs(row.contribution);
    item.mean_contribution += row.contribution;
    if (row.contribution >= 0) item.positive_sum += row.contribution;
    else item.negative_sum += row.contribution;
    item.count += 1;
  });
  return Array.from(map.values())
    .map(row => ({ ...row, mean_contribution: row.count ? row.mean_contribution / row.count : 0 }))
    .sort((a, b) => b.total_abs_contribution - a.total_abs_contribution)
    .slice(0, topN)
    .reverse();
}

function renderPart6BackendVisuals(result) {
  const predictions = backendPredictions(result);
  const shapRows = backendShapRows(result);
  const shapAgg = aggregateBackendShap(shapRows, 14);

  const avgProb = predictions.length ? mean(predictions.map(p => p.prediction_probability)) : NaN;
  const highCount = predictions.filter(p => p.prediction_probability >= 0.6).length;
  const topProb = predictions.length ? Math.max(...predictions.map(p => p.prediction_probability)) : NaN;

  if (dom.metricBackendPredictionCount) dom.metricBackendPredictionCount.textContent = predictions.length ? formatInt(predictions.length) : "-";
  if (dom.metricBackendAvgProb) dom.metricBackendAvgProb.textContent = Number.isFinite(avgProb) ? formatPct(avgProb) : "-";
  if (dom.metricBackendHighProb) dom.metricBackendHighProb.textContent = predictions.length ? formatInt(highCount) : "-";
  if (dom.metricBackendTopProb) dom.metricBackendTopProb.textContent = Number.isFinite(topProb) ? formatPct(topProb) : "-";

  const stockActionRows = part6StockActionRowsForDisplay(predictions);

  drawPart6PredictionRankChart(predictions);
  drawPart6ProbabilityHistogram(predictions);
  drawPart6StockActionChart(stockActionRows);
  drawPart6ShapFeatureChart(shapAgg);
  drawPart6SingleEventShapChart(shapRows);
  renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows);
}

/* Part7 only: evidence-grounded critic over persisted Part1–Part6 evidence. */
async function loadPart7Status() {
  if (!dom.part7Status) return;
  try {
    const response = await fetch('/api/part7/status');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const info = await response.json();
    const modeText = info.mode === 'live' ? 'Live API mode' : 'Preview mode（不會呼叫 API）';
    dom.part7Status.textContent = `${modeText} · model=${info.default_model || '-'} · local documents=${info.local_document_count || 0} · API key=${info.api_key_configured ? 'configured' : 'not configured'} · openai package=${info.openai_package_available ? 'ready' : 'not installed'}`;
    if (dom.part7ModelInput && info.default_model) dom.part7ModelInput.value = info.default_model;
    renderPart7RuntimeCards(info);
  } catch (error) {
    dom.part7Status.textContent = `Part7 status 無法取得：${error.message}`;
  }
}

function renderPart7RuntimeCards(info) {
  if (!dom.part7RuntimeCards) return;
  const cards = [
    ['執行模式', info.mode || '-'],
    ['API Key', info.api_key_configured ? '已在 backend 設定' : '尚未設定'],
    ['Local RAG 文件', String(info.local_document_count ?? '-')],
    ['模型', info.default_model || info.model || '-']
  ];
  dom.part7RuntimeCards.innerHTML = cards.map(([label, value]) => `<div class="part7-runtime-card"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></div>`).join('');
}

function part7PopulateEvents(preferredIds = []) {
  if (!dom.part7EventSelect) return;
  const current = Array.from(dom.part7EventSelect.selectedOptions || []).map(option => option.value).filter(Boolean);
  const predictions = ((((state.part6 || {}).backendResult || {}).ml_result || {}).predictions) || [];
  const validIds = new Set(predictions.map(row => String(row.event_id || '')));
  let selected = preferredIds.length ? preferredIds.filter(id => validIds.has(id)).slice(0, 8) : current.filter(id => validIds.has(id));
  const part6Selected = preferredIds.length ? preferredIds : (typeof part6SelectedShapEventIds === 'function' ? part6SelectedShapEventIds() : []);
  if (!selected.length) selected = part6Selected.filter(id => validIds.has(id)).slice(0, 8);
  const options = ['<option value="">自動選擇最高機率事件</option>'];
  for (const row of predictions) {
    const label = `${row.report_date || '-'} | ${row.manager || '-'} | ${row.fund_ticker || row.crsp_portno || '-'} | ${row.action_type || '-'}`;
    const id = String(row.event_id || '');
    options.push(`<option value="${escapeHtml(id)}"${selected.includes(id) ? ' selected' : ''}>${escapeHtml(label)}</option>`);
  }
  dom.part7EventSelect.innerHTML = options.join('');
}

function part7EvidenceBadges(ids) {
  return (ids || []).map(id => `<span class="part7-evidence-id">${escapeHtml(id)}</span>`).join('');
}

function part7SafeUrl(value) {
  try {
    const url = new URL(String(value || ''), window.location.origin);
    return (url.protocol === 'https:' || url.protocol === 'http:') ? url.href : '';
  } catch (_) {
    return '';
  }
}

function part7RenderAssessments(title, items, className) {
  const body = (items || []).length ? items.map(item => `
    <div class="part7-item">
      <strong>${escapeHtml(item.claim || item.risk || '')}</strong> ${part7EvidenceBadges(item.evidence_ids)}
      <div>${escapeHtml(item.reasoning || item.why_it_matters || '')}</div>
      ${item.strength ? `<small>strength: ${escapeHtml(item.strength)}</small>` : ''}
      ${item.recommended_check ? `<small>建議檢查：${escapeHtml(item.recommended_check)}</small>` : ''}
    </div>`).join('') : '<p>沒有足夠證據產生此項目。</p>';
  return `<section class="part7-critic-section ${className}"><h3>${escapeHtml(title)}</h3>${body}</section>`;
}

function part7RenderEvidence(evidence) {
  const rows = (evidence || []).map(item => ({
    evidence_id: item.evidence_id,
    type: item.type,
    date: item.date,
    title: item.title,
    source: item.source,
    retrieval_score: item.retrieval_score,
    excerpt: String(item.text || '').slice(0, 420)
  }));
  return rows.length ? `<section class="part7-critic-section"><h3>RAG retrieved evidence</h3><div id="part7EvidenceTable"></div></section>` : '';
}

function part7RenderResult(result) {
  if (!dom.part7ExplanationWorkspace) return;
  const event = result.event || {};
  const events = Array.isArray(result.events) && result.events.length ? result.events : (event.event_id ? [event] : []);
  const analysis = result.analysis;
  const probability = Number(event.positive_probability);
  let html = `<div class="part7-verdict-grid">
    <div class="part7-verdict-card"><small>事件數</small><strong>${events.length || 0}</strong></div>
    <div class="part7-verdict-card"><small>經理人</small><strong>${escapeHtml([...new Set(events.map(item => item.manager).filter(Boolean))].join('；') || '-')}</strong></div>
    <div class="part7-verdict-card"><small>報告日／天期</small><strong>${escapeHtml(event.report_date || '-')} / ${escapeHtml(event.horizon_months || '-')}M</strong></div>
    <div class="part7-verdict-card"><small>Part6 positive probability</small><strong>${Number.isFinite(probability) ? formatPct(probability) : '-'}</strong></div>
  </div>`;
  if (events.length) html += `<section class="part7-critic-section"><h3>Selected Part6 events</h3>${events.map(item => `<div class="part7-item"><strong>${escapeHtml(item.manager || '-')} | ${escapeHtml(item.report_date || '-')}</strong><div>${escapeHtml(item.event_id || '')} · ${escapeHtml(item.action_type || '-')} · positive probability ${Number.isFinite(Number(item.positive_probability)) ? formatPct(Number(item.positive_probability)) : '-'}</div></div>`).join('')}</section>`;

  if (analysis) {
    const confidence = Number(analysis.confidence);
    html += `<section class="part7-critic-section"><h3>Critic conclusion</h3>
      <p><strong>${escapeHtml(analysis.verdict || 'unknown')}</strong> · confidence ${Number.isFinite(confidence) ? formatPct(confidence) : '-'}</p>
      <p>${escapeHtml(analysis.executive_summary || '')}</p><p><strong>Model claim：</strong>${escapeHtml(analysis.model_claim || '')}</p></section>`;
    html += part7RenderAssessments('支持模型解釋的證據', analysis.supporting_evidence, 'part7-support');
    html += part7RenderAssessments('反證與替代解釋', analysis.counter_evidence, 'part7-counter');
    html += part7RenderAssessments('Structural breaks', analysis.structural_breaks, 'part7-risk');
    html += part7RenderAssessments('資料限制', analysis.data_limitations, 'part7-risk');
    html += part7RenderAssessments('過度解釋風險', analysis.overinterpretation_risks, 'part7-risk');
    const questions = (analysis.questions_for_human || []).map(q => `<li>${escapeHtml(q)}</li>`).join('');
    html += `<section class="part7-critic-section"><h3>交由專家判斷的問題</h3><ol>${questions || '<li>無</li>'}</ol></section>`;
  } else {
    html += `<section class="part7-critic-section"><h3>Preview ready</h3><p>${escapeHtml(result.message || '')}</p><p>此結果沒有呼叫模型；可先檢查下方 RAG evidence 與 prompt。</p></section>`;
  }

  html += part7RenderEvidence(result.retrieved_evidence);
  const allCitations = [...((analysis || {}).citations || []), ...(result.web_citations || [])];
  if (allCitations.length) {
    html += `<section class="part7-critic-section part7-citations"><h3>Citations</h3><ul>${allCitations.map(c => {
      const title = c.title || c.evidence_id || c.url || 'source';
      const safeUrl = part7SafeUrl(c.url);
      return safeUrl ? `<li>${escapeHtml(c.evidence_id || '')} <a href="${escapeHtml(safeUrl)}" target="_blank" rel="noopener">${escapeHtml(title)}</a></li>` : `<li>${escapeHtml(c.evidence_id || '')} ${escapeHtml(title)} · ${escapeHtml(c.source || '')}</li>`;
    }).join('')}</ul></section>`;
  }
  if (result.prompt_preview) {
    html += `<details class="part7-critic-section part7-prompt-preview" open><summary><strong>實際 Prompt Preview</strong></summary><h4>Instructions</h4><pre>${escapeHtml(result.prompt_preview.instructions || '')}</pre><h4>Input + retrieved context</h4><pre>${escapeHtml(result.prompt_preview.input || '')}</pre></details>`;
  }
  dom.part7ExplanationWorkspace.innerHTML = html;
  const evidenceRows = (result.retrieved_evidence || []).map(item => ({
    evidence_id: item.evidence_id, type: item.type, date: item.date, title: item.title,
    source: item.source, retrieval_score: item.retrieval_score,
    excerpt: String(item.text || '').slice(0, 420)
  }));
  renderTable('part7EvidenceTable', evidenceRows, [
    { key: 'evidence_id', label: 'ID' }, { key: 'type', label: 'type' }, { key: 'date', label: 'date' },
    { key: 'title', label: 'title' }, { key: 'source', label: 'source' },
    { key: 'retrieval_score', label: 'score', format: 'num' }, { key: 'excerpt', label: 'excerpt' }
  ], { title: 'Part7 evidence audit trail', expanded: true, count: evidenceRows.length });
}

async function runPart7AnalysisPlaceholder() {
  part7PopulateEvents();
  if (!dom.part7RunAnalysisBtn || !dom.part7Status) return;
  dom.part7RunAnalysisBtn.disabled = true;
  dom.part7Status.textContent = 'Part7 正在建立 evidence IDs、執行 local RAG，並準備 critic request…';
  try {
    const payload = {
      event_ids: dom.part7EventSelect ? Array.from(dom.part7EventSelect.selectedOptions).map(option => option.value).filter(Boolean).slice(0, 8) : [],
      horizon_months: Number(dom.part7HorizonSelect ? dom.part7HorizonSelect.value : 12),
      model: dom.part7ModelInput ? (dom.part7ModelInput.value.trim() || null) : null,
      use_web_search: Boolean(dom.part7WebSearchInput && dom.part7WebSearchInput.checked),
      max_local_chunks: 12,
      question: dom.part7QuestionInput ? (dom.part7QuestionInput.value.trim() || null) : null
    };
    const response = await fetch('/api/part7/critic', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    dom.part7Status.textContent = result.status === 'preview'
      ? 'Preview 完成：尚未呼叫 OpenAI；已產生 RAG evidence 與 prompt。'
      : `Critic 完成 · response_id=${result.response_id || '-'}`;
    renderPart7RuntimeCards({ mode: result.status, api_key_configured: result.status === 'ok', local_document_count: (result.retrieval || {}).local_documents_available, default_model: result.model });
    part7RenderResult(result);
  } catch (error) {
    dom.part7Status.textContent = `Part7 執行失敗：${error.message}`;
  } finally {
    dom.part7RunAnalysisBtn.disabled = false;
  }
}

function renderPart6ExpertCollaboration(result) {
  const expert = result && result.expert_collaboration ? result.expert_collaboration : null;
  const recommendations = expert && Array.isArray(expert.recommendations) ? expert.recommendations : [];
  const managers = expert && Array.isArray(expert.manager_contributions) ? expert.manager_contributions : [];
  const latest = expert && expert.latest_recommendation ? expert.latest_recommendation : null;
  if (dom.part6ExpertLatestCards) {
    dom.part6ExpertLatestCards.innerHTML = latest ? `
      <div class="part6-anchor-card emphasis"><span>建議日期</span><strong>${escapeHtml(latest.report_date || '-')}</strong><p>${formatInt(latest.expert_count || 0)} 位平衡型基金專家</p></div>
      <div class="part6-anchor-card"><span>純專家共識</span><strong>股 ${escapeHtml(formatPct(latest.expert_stock))}｜債 ${escapeHtml(formatPct(latest.expert_bond))}</strong><p>現金 ${escapeHtml(formatPct(latest.expert_cash))}</p></div>
      <div class="part6-anchor-card"><span>專家 × AI 建議</span><strong>股 ${escapeHtml(formatPct(latest.human_ai_stock))}｜債 ${escapeHtml(formatPct(latest.human_ai_bond))}</strong><p>現金 ${escapeHtml(formatPct(latest.human_ai_cash))}</p></div>
      <div class="part6-anchor-card"><span>${state.part6.horizonMonths || 12}M 初步比較</span><strong>Human–AI ${Number.isFinite(Number(latest.human_ai_outcome_proxy)) ? escapeHtml(formatPct(Number(latest.human_ai_outcome_proxy))) : '-'}</strong><p>AI-only ${Number.isFinite(Number(latest.ai_outcome_proxy)) ? escapeHtml(formatPct(Number(latest.ai_outcome_proxy))) : '-'}｜${escapeHtml(latest.validation_status || '-')}</p></div>
    ` : '<div class="part6-backend-empty-window">目前選取事件不足以建立專家股債配置共識。</div>';
  }
  if (dom.part6ExpertAllocationChart && window.Plotly) {
    if (!recommendations.length) {
      Plotly.react(dom.part6ExpertAllocationChart, [], { title: '平衡型基金專家 × AI 配置（無資料）', height: 380 }, { responsive: true });
    } else {
      Plotly.react(dom.part6ExpertAllocationChart, [
        { type: 'scatter', mode: 'lines+markers', name: '純專家：股票', x: recommendations.map(r => r.report_date), y: recommendations.map(r => r.expert_stock), line: { color: '#2187d5', width: 3 } },
        { type: 'scatter', mode: 'lines+markers', name: '純 AI：股票', x: recommendations.map(r => r.report_date), y: recommendations.map(r => r.ai_stock), line: { color: '#805ad5', width: 2 } },
        { type: 'scatter', mode: 'lines+markers', name: '專家×AI：股票', x: recommendations.map(r => r.report_date), y: recommendations.map(r => r.human_ai_stock), line: { color: '#1f9d8a', width: 3 } },
        { type: 'scatter', mode: 'lines+markers', name: '等權：股票', x: recommendations.map(r => r.report_date), y: recommendations.map(r => r.equal_weight_stock), line: { color: '#7b8790', dash: 'dot' } },
        { type: 'scatter', mode: 'lines+markers', name: '專家×AI：債券', x: recommendations.map(r => r.report_date), y: recommendations.map(r => r.human_ai_bond), line: { color: '#f0a44b', width: 2 } }
      ], {
        title: `平衡型基金專家集體智慧與 Human–AI 動態配置（${expert.horizon_months || state.part6.horizonMonths}M）`,
        height: 460, margin: { l: 65, r: 30, t: 68, b: 70 },
        xaxis: { title: 'T-1 holdings report date', tickangle: -35 },
        yaxis: { title: '建議配置比例', tickformat: '.0%', range: [0, 1] },
        legend: { orientation: 'h', y: -0.28 }, hovermode: 'x unified'
      }, { displaylogo: false, responsive: true });
    }
  }
  renderTable('part6ExpertRecommendationTable', recommendations, [
    { key: 'report_date', label: 'T-1 report date' }, { key: 'expert_count', label: '專家數' },
    { key: 'expert_stock', label: '專家股票', format: 'pct' }, { key: 'expert_bond', label: '專家債券', format: 'pct' },
    { key: 'ai_stock', label: 'AI-only 股票', format: 'pct' }, { key: 'ai_bond', label: 'AI-only 債券', format: 'pct' },
    { key: 'human_ai_stock', label: '專家×AI 股票', format: 'pct' }, { key: 'human_ai_bond', label: '專家×AI 債券', format: 'pct' },
    { key: 'human_ai_cash', label: '專家×AI 現金', format: 'pct' }, { key: 'equal_weight_stock', label: '等權股票', format: 'pct' },
    { key: 'ai_outcome_proxy', label: 'AI-only outcome', format: 'pct' }, { key: 'expert_outcome_proxy', label: '專家 outcome', format: 'pct' },
    { key: 'human_ai_outcome_proxy', label: 'Human–AI outcome', format: 'pct' }, { key: 'equal_weight_outcome_proxy', label: '等權 outcome', format: 'pct' },
    { key: 'validation_status', label: '驗證狀態' }
  ], { title: '各期專家共識配置（全部期間）', expanded: true, count: recommendations.length });
  renderTable('part6ExpertManagerTable', managers, [
    { key: 'report_date', label: 'report date' }, { key: 'manager', label: '經理人' }, { key: 'fund_ticker', label: '基金' },
    { key: 'style_group', label: '同風格群組' }, { key: 'sharpe', label: 'Sharpe', format: 'num' },
    { key: 'information_ratio', label: 'Information Ratio', format: 'num' }, { key: 'alpha', label: 'Alpha', format: 'pct' },
    { key: 'performance_score', label: '專家績效分數', format: 'num' }, { key: 'relative_style_score', label: '同風格相對分數', format: 'num' },
    { key: 'expert_weight', label: '專家權重', format: 'pct' }, { key: 'ai_positive_probability', label: 'AI 正向機率', format: 'pct' }, { key: 'ai_weight', label: 'AI-only 權重', format: 'pct' },
    { key: 'human_ai_weight', label: 'Human–AI 權重', format: 'pct' }, { key: 'stock_allocation', label: '股票配置', format: 'pct' },
    { key: 'bond_allocation', label: '債券配置', format: 'pct' }, { key: 'cash_allocation', label: '現金配置', format: 'pct' }
  ], { title: '經理人專家權重與股債配置貢獻（全部列出）', expanded: true, count: managers.length });
  if (dom.part6ExpertCaveat) dom.part6ExpertCaveat.textContent = expert ? expert.research_caveat : '尚無專家協作結果。';
}

function part6AllStockActionRows(predictions) {
  const rows = part6StockActionRowsForDisplay(predictions);
  const managerMap = (((state.part6.backendResult || {}).expert_collaboration || {}).report_manager_map) || {};
  return rows.map(row => ({
    ...row,
    manager: row.manager || managerMap[part6PredictionKey(row.crsp_portno, row.report_dt)] || 'Unknown manager'
  })).sort((a, b) => String(b.report_dt).localeCompare(String(a.report_dt)) || String(a.manager).localeCompare(String(b.manager)) || (b.signed_delta_abs || 0) - (a.signed_delta_abs || 0));
}

function drawPart6AllStockActionChart(rows) {
  const node = dom.part6StockActionChart;
  if (!node || !window.Plotly) return;
  const data = (rows || []).filter(row => row.stock_action_direction).slice().reverse();
  const managerCount = new Set(data.map(row => row.manager).filter(Boolean)).size;
  const counts = { new_position: 0, increase: 0, decrease: 0 };
  data.forEach(row => { if (Object.prototype.hasOwnProperty.call(counts, row.stock_action_direction)) counts[row.stock_action_direction] += 1; });
  if (dom.part6StockActionSummary) dom.part6StockActionSummary.textContent = `共 ${data.length} 筆｜${managerCount} 位經理人｜新增 ${counts.new_position}｜加碼 ${counts.increase}｜減碼 ${counts.decrease}（全部顯示）`;
  if (!data.length) {
    Plotly.react(node, [], { title: 'Part 5 真實持股動作（尚無可對齊資料）', height: 400 }, { responsive: true });
    return;
  }
  Plotly.react(node, [{
    type: 'bar', orientation: 'h', x: data.map(row => row.delta_holding_pct),
    y: data.map(row => `${row.manager}｜${row.holding_ticker || row.holding_key}｜${row.report_dt}`),
    text: data.map(row => row.stock_action_direction === 'decrease' ? '減碼' : (row.stock_action_direction === 'new_position' ? '新增' : '加碼')),
    marker: { color: data.map(row => row.stock_action_direction === 'decrease' ? '#c94c4c' : (row.stock_action_direction === 'new_position' ? '#2187d5' : '#1f9d8a')) },
    customdata: data.map(row => [row.manager, row.fund_ticker || row.crsp_portno, row.holding_security_name, row.sector, row.previous_holding_pct, row.current_holding_pct, row.stock_action_direction, row.prediction_probability]),
    hovertemplate: '經理人：%{customdata[0]}<br>基金：%{customdata[1]}<br>標的：%{customdata[2]}<br>Sector：%{customdata[3]}<br>動作：%{customdata[6]}<br>前期權重：%{customdata[4]:.2%}<br>本期權重：%{customdata[5]:.2%}<br>變化：%{x:.2%}<br>AI 正向機率：%{customdata[7]:.1%}<extra></extra>'
  }], {
    title: `Part 6：全部真實新增／加碼／減碼（${data.length} 筆，含經理人）`,
    height: Math.max(520, 180 + data.length * 24), margin: { l: 330, r: 40, t: 72, b: 60 },
    xaxis: { title: '持股權重變化：負值＝減碼，正值＝新增／加碼', tickformat: '.1%', zeroline: true },
    yaxis: { automargin: true }, showlegend: false
  }, { displaylogo: false, responsive: true });
}

function renderPart6AllStockActionTable(rows) {
  renderTable('part6StockActionTable', rows || [], [
    { key: 'manager', label: '經理人' }, { key: 'report_dt', label: '報告日' }, { key: 'previous_report_dt', label: '前期報告日' },
    { key: 'fund_ticker', label: '基金' }, { key: 'fund_name', label: '基金名稱' }, { key: 'holding_ticker', label: 'ticker' },
    { key: 'holding_security_name', label: '持有標的' }, { key: 'sector', label: '產業' },
    { key: 'stock_action_direction', label: '真實動作' }, { key: 'previous_holding_pct', label: '前期權重', format: 'pct' },
    { key: 'current_holding_pct', label: '本期權重', format: 'pct' }, { key: 'delta_holding_pct', label: '權重變化', format: 'pct' },
    { key: 'stock_beta', label: 'Beta', format: 'num' }, { key: 'prediction_probability', label: 'AI 正向機率', format: 'pct' },
    { key: 'model_action_type', label: '模型事件類型' }, { key: 'market_regime', label: '市場狀態' }
  ], { title: 'Part 5 選取事件之全部持股動作（含經理人）', expanded: true, count: (rows || []).length });
}

/* Multi-horizon diagnostic layer. These late declarations intentionally replace
   the legacy 12M-only adapters while keeping Part 1–5 untouched. */
function syncPart6HorizonTabs() {
  const selected = Number(state.part6.horizonMonths) || 12;
  document.querySelectorAll('#part6HorizonTabs button[data-horizon]').forEach(button => {
    button.classList.toggle('active', Number(button.dataset.horizon) === selected);
  });
}

function part6DayNumber(value) {
  const time = Date.parse(value);
  return Number.isFinite(time) ? Math.floor(time / 86400000) : NaN;
}

function part6DayIso(day) {
  return new Date(Number(day) * 86400000).toISOString().slice(0, 10);
}

function configurePart6DateDomain(domain) {
  if (!domain || !domain.min || !domain.max || !dom.part6DateStartRange || !dom.part6DateEndRange) return;
  const min = part6DayNumber(domain.min), max = part6DayNumber(domain.max);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return;
  state.part6.dateDomain = { min: domain.min, max: domain.max };
  [dom.part6DateStartRange, dom.part6DateEndRange].forEach(input => { input.min = String(min); input.max = String(max); });
  dom.part6DateStartRange.value = String(state.part6.dateStart ? part6DayNumber(state.part6.dateStart) : min);
  dom.part6DateEndRange.value = String(state.part6.dateEnd ? part6DayNumber(state.part6.dateEnd) : max);
  updatePart6DateRangeFromControls();
}

function updatePart6DateRangeFromControls() {
  if (!dom.part6DateStartRange || !dom.part6DateEndRange) return;
  let start = Number(dom.part6DateStartRange.value), end = Number(dom.part6DateEndRange.value);
  if (start > end) {
    const active = document.activeElement;
    if (active === dom.part6DateStartRange) end = start; else start = end;
    dom.part6DateStartRange.value = String(start); dom.part6DateEndRange.value = String(end);
  }
  state.part6.dateStart = part6DayIso(start); state.part6.dateEnd = part6DayIso(end);
  if (dom.part6DateRangeLabel) dom.part6DateRangeLabel.textContent = `${state.part6.dateStart} ～ ${state.part6.dateEnd}`;
}

function backendPredictions(result) {
  const horizon = Number(state.part6.horizonMonths) || 12;
  const preds = result && result.ml_result && Array.isArray(result.ml_result.predictions) ? result.ml_result.predictions : [];
  return preds.map((p, i) => ({
    ...p,
    rank: i + 1,
    horizon_months: horizon,
    prediction_probability: Number(p[`positive_probability_${horizon}m`]),
    predicted_excess: Number(p[`predicted_excess_${horizon}m`]),
    predicted_class: p[`predicted_class_${horizon}m`],
    future_horizon_excess_return: p[`future_${horizon}m_excess_return`],
    future_12m_excess_return: p[`future_${horizon}m_excess_return`],
    label_positive_excess_12m: p[`outcome_5class_${horizon}m`],
    event_label: `${p.manager || 'Unknown manager'} | ${p.report_date || ''}`,
    short_label: `${i + 1}. ${(p.manager || 'Unknown').slice(0, 18)} ${p.report_date || ''}`
  })).filter(p => Number.isFinite(p.prediction_probability));
}

function backendShapRows(result) {
  const horizon = Number(state.part6.horizonMonths) || 12;
  const explanations = result && result.shap_result && Array.isArray(result.shap_result.explanations) ? result.shap_result.explanations : [];
  const rows = [];
  explanations.filter(event => Number(event.horizon_months) === horizon).forEach((event, eventIndex) => {
    const eventLabel = `${event.manager || 'Unknown manager'} | ${event.report_date || ''}`;
    (event.features || []).forEach(item => rows.push({
      event_index: eventIndex, event_id: event.event_id, event_label: eventLabel,
      manager: event.manager, fund: event.fund, report_date: event.report_date,
      horizon_months: horizon, direction: Number(item.contribution) >= 0 ? 'positive' : 'negative',
      feature: item.feature, value: item.value, contribution: Number(item.contribution)
    }));
  });
  return rows.filter(row => Number.isFinite(row.contribution));
}

function drawPart6ClusterMap(result) {
  const node = dom.part6ClusterMap;
  if (!node || !window.Plotly) return;
  const temporal = result && result.shap_result ? result.shap_result.temporal_clustering : null;
  const clusters = temporal && Array.isArray(temporal.clusters) ? temporal.clusters : [];
  const points = temporal && Array.isArray(temporal.points) ? temporal.points : [];
  if (!points.length) { Plotly.react(node, [], { title: 'TreeSHAP 決策邏輯聚類（資料不足）', height: 460 }, { responsive: true }); return; }
  const clusterMap = new Map(clusters.map(c => [Number(c.cluster), c]));
  const predictionStyleMap = new Map(((((result || {}).ml_result || {}).predictions) || []).map(p => [String(p.event_id || ''), p.manager_style_group || 'Unknown style']));
  const styleColors = {
    'Defensive / risk-control style': '#2f6b9a',
    'High-return / high-flow style': '#c94c4c',
    'Equity-tilted / growth style': '#e28a2b',
    'Flow-supported core style': '#249b8a',
    'Balanced core style': '#7655a6',
    'Unknown style': '#7b8790'
  };
  const styleOrder = Object.keys(styleColors);
  const enrichedPoints = points.map(p => ({
    ...p,
    manager_style_group: p.manager_style_group || predictionStyleMap.get(String(p.event_id || '')) || 'Unknown style'
  }));
  const observedStyles = [...new Set(enrichedPoints.map(p => p.manager_style_group))]
    .sort((a, b) => (styleOrder.indexOf(a) < 0 ? 999 : styleOrder.indexOf(a)) - (styleOrder.indexOf(b) < 0 ? 999 : styleOrder.indexOf(b)));
  const traces = observedStyles.map(style => {
    const stylePoints = enrichedPoints.filter(p => p.manager_style_group === style);
    return {
      type: 'scatter', mode: 'markers', name: style,
      x: stylePoints.map(p => p.x), y: stylePoints.map(p => p.y),
      text: stylePoints.map(p => `${p.manager || ''} | ${p.report_date || ''}`),
      customdata: stylePoints.map(p => [p.cluster, (clusterMap.get(Number(p.cluster)) || {}).name || '', p.predicted_class, p.positive_probability, style]),
      marker: {
        size: stylePoints.map(p => 12 + 30 * ((clusterMap.get(Number(p.cluster)) || {}).large_win_rate || 0)),
        color: styleColors[style] || '#7b8790', line: { color: '#fff', width: 1.2 }, opacity: 0.88
      },
      hovertemplate: '%{text}<br>經理人風格：%{customdata[4]}<br>SHAP Cluster %{customdata[0]}：%{customdata[1]}<br>預測：%{customdata[2]}<br>正向機率：%{customdata[3]:.1%}<extra></extra>'
    };
  });
  Plotly.react(node, traces, {
    title: `${temporal.horizon_months || state.part6.horizonMonths}M TreeSHAP 決策邏輯地圖（顏色＝經理人 Style）`, height: 500,
    xaxis: { title: 'PCA display axis 1', zeroline: false }, yaxis: { title: 'PCA display axis 2', zeroline: false },
    legend: { title: { text: 'Manager style' }, orientation: 'v', x: 1.01, y: 1 },
    margin: { l: 60, r: 230, t: 70, b: 60 }
  }, { displaylogo: false, responsive: true });
  resetPlotlyHandler(node, 'plotly_click', event => {
    const cluster = Number(event.points && event.points[0] && event.points[0].customdata[0]);
    if (Number.isFinite(cluster)) { state.part6.activeCluster = cluster; drawPart6Fidelity(result); }
  });
  const clusterRows = clusters.map(c => ({
    ...c,
    dominant_manager_style: c.dominant_manager_style || 'Unknown style',
    manager_style_mix: Object.entries(c.manager_style_counts || {}).map(([style, count]) => `${style}: ${count}`).join('；') || '-'
  }));
  renderTable('part6ClusterSummary', clusterRows, [
    { key: 'cluster', label: 'Cluster' }, { key: 'name', label: '資料驅動命名' },
    { key: 'dominant_manager_style', label: '主導經理人 Style' }, { key: 'manager_style_mix', label: 'Style 分布' },
    { key: 'event_count', label: '事件數' }, { key: 'large_win_rate', label: '模型預測 large-win 占比', format: 'pct' },
    { key: 'large_loss_rate', label: '模型預測 large-loss 占比', format: 'pct' }, { key: 'top_features', label: '主導特徵' }
  ], { title: 'SHAP 聚類摘要（含經理人 Style 組成）', expanded: true, count: clusters.length });
  if (!clusters.some(c => Number(c.cluster) === Number(state.part6.activeCluster))) state.part6.activeCluster = Number(clusters[0].cluster);
  drawPart6Fidelity(result);
}

function drawPart6Fidelity(result) {
  if (!window.Plotly || !dom.part6FidelityShapChart || !dom.part6FidelityRawChart) return;
  const temporal = result && result.shap_result ? result.shap_result.temporal_clustering : null;
  const clusters = temporal && Array.isArray(temporal.clusters) ? temporal.clusters : [];
  const cluster = clusters.find(c => Number(c.cluster) === Number(state.part6.activeCluster)) || clusters[0];
  if (!cluster) return;
  const rows = cluster.fidelity || [];
  if (dom.part6FidelityHint) dom.part6FidelityHint.textContent = `Cluster ${cluster.cluster}｜${cluster.name}｜${cluster.event_count} events`;
  Plotly.react(dom.part6FidelityShapChart, [{
    type: 'bar', orientation: 'h', y: rows.map(r => r.feature).reverse(), x: rows.map(r => r.mean_shap).reverse(),
    marker: { color: rows.map(r => r.mean_shap >= 0 ? '#2187d5' : '#c94c4c').reverse() },
    hovertemplate: '%{y}<br>mean TreeSHAP=%{x:.4f}<extra></extra>'
  }], { title: '左：群組平均 TreeSHAP 歸因', height: 380, margin: { l: 190, r: 30, t: 60, b: 45 } }, { displaylogo: false, responsive: true });
  const traces = rows.map((r, i) => {
    const raw = r.raw_values || [];
    const numericRaw = raw.map(Number).filter(Number.isFinite);
    const fallbackMean = numericRaw.length ? numericRaw.reduce((sum, value) => sum + value, 0) / numericRaw.length : 0;
    const fallbackStd = numericRaw.length > 1 ? Math.sqrt(numericRaw.reduce((sum, value) => sum + (value - fallbackMean) ** 2, 0) / (numericRaw.length - 1)) : 1;
    const reportedMean = Number(r.market_mean), reportedStd = Number(r.market_std);
    const mean = Number.isFinite(reportedMean) ? reportedMean : fallbackMean;
    const std = Number.isFinite(reportedStd) && reportedStd > 0 ? reportedStd : (fallbackStd > 0 ? fallbackStd : 1);
    const standardized = raw.map(value => (Number(value) - mean) / std);
    return {
      type: 'box', orientation: 'h', name: r.feature, x: standardized,
      boxpoints: 'all', jitter: 0.38, pointpos: 0, boxmean: true,
      marker: { size: 8, opacity: 0.78, color: ['#2187d5', '#1f9d8a', '#f0a44b'][i % 3], line: { color: '#fff', width: 0.8 } },
      line: { width: 2, color: ['#2187d5', '#1f9d8a', '#f0a44b'][i % 3] },
      customdata: raw.map(value => [value, mean, std]),
      hovertemplate: `${r.feature}<br>standardized=%{x:.3f}σ<br>raw=%{customdata[0]:.4f}<br>全事件均值=%{customdata[1]:.4f}<br>全事件標準差=%{customdata[2]:.4f}<extra></extra>`
    };
  });
  Plotly.react(dom.part6FidelityRawChart, traces, {
    title: '右：群組特徵標準化分布（避免不同量尺壓縮）', height: Math.max(420, 230 + rows.length * 68), showlegend: false,
    margin: { l: 190, r: 30, t: 60, b: 55 },
    xaxis: { title: 'Distance from all-event mean (standard deviations)', zeroline: true, zerolinewidth: 2, zerolinecolor: '#667784' }
  }, { displaylogo: false, responsive: true });
}

function renderPart6BackendVisuals(result) {
  configurePart6DateDomain(result && result.received_summary ? result.received_summary.date_domain : null);
  syncPart6HorizonTabs();
  const predictions = part6EnrichedPredictions(result);
  const shapRows = backendShapRows(result);
  const shapAgg = aggregateBackendShap(shapRows, 14);
  const avgProb = predictions.length ? mean(predictions.map(p => p.prediction_probability)) : NaN;
  const highCount = predictions.filter(p => p.prediction_probability >= 0.6).length;
  const topProb = predictions.length ? Math.max(...predictions.map(p => p.prediction_probability)) : NaN;
  if (dom.metricBackendPredictionCount) dom.metricBackendPredictionCount.textContent = predictions.length ? formatInt(predictions.length) : '-';
  if (dom.metricBackendAvgProb) dom.metricBackendAvgProb.textContent = Number.isFinite(avgProb) ? formatPct(avgProb) : '-';
  if (dom.metricBackendHighProb) dom.metricBackendHighProb.textContent = predictions.length ? formatInt(highCount) : '-';
  if (dom.metricBackendTopProb) dom.metricBackendTopProb.textContent = Number.isFinite(topProb) ? formatPct(topProb) : '-';
  const stockActionRows = part6StockActionRowsForDisplay(predictions);
  part6RenderAnchorCards(result, predictions);
  part6DrawStyleEventChart(predictions);
  part6RenderStyleEventTable(predictions);
  drawPart6PredictionRankChart(predictions); drawPart6ProbabilityHistogram(predictions);
  drawPart6StockActionChart(stockActionRows); drawPart6ShapFeatureChart(shapAgg); drawPart6SingleEventShapChart(shapRows);
  drawPart6ClusterMap(result);
  renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows);
}

// Final dispatcher: the file contains legacy renderers for backward compatibility;
// this assignment guarantees the multi-horizon renderer is the active one.
renderPart6BackendVisuals = function(result) {
  configurePart6DateDomain(result && result.received_summary ? result.received_summary.date_domain : null);
  syncPart6HorizonTabs();
  const predictions = part6EnrichedPredictions(result);
  const shapRows = backendShapRows(result);
  const shapAgg = aggregateBackendShap(shapRows, 14);
  const probabilities = predictions.map(p => p.prediction_probability).filter(Number.isFinite);
  if (dom.metricBackendPredictionCount) dom.metricBackendPredictionCount.textContent = formatInt(predictions.length);
  if (dom.metricBackendAvgProb) dom.metricBackendAvgProb.textContent = probabilities.length ? formatPct(mean(probabilities)) : '-';
  if (dom.metricBackendHighProb) dom.metricBackendHighProb.textContent = formatInt(probabilities.filter(p => p >= 0.6).length);
  if (dom.metricBackendTopProb) dom.metricBackendTopProb.textContent = probabilities.length ? formatPct(Math.max(...probabilities)) : '-';
  const stockActionRows = part6AllStockActionRows(predictions);
  renderPart6ExpertCollaboration(result);
  part6RenderAnchorCards(result, predictions); part6DrawStyleEventChart(predictions); part6RenderStyleEventTable(predictions);
  drawPart6PredictionRankChart(predictions); drawPart6ProbabilityHistogram(predictions); drawPart6AllStockActionChart(stockActionRows);
  drawPart6ShapFeatureChart(shapAgg); drawPart6SingleEventShapChart(shapRows); drawPart6ClusterMap(result);
  renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows);
  renderPart6AllStockActionTable(stockActionRows);
};

function drawPart6PredictionRankChart(predictions) {
  const node = dom.part6PredictionRankChart;
  if (!node || !window.Plotly) return;
  if (!predictions.length) { Plotly.purge(node); return; }
  const ranked = [...predictions].sort((a, b) => b.prediction_probability - a.prediction_probability).slice(0, 25).reverse();
  Plotly.react(node, [{
    type: "bar",
    orientation: "h",
    x: ranked.map(p => p.prediction_probability),
    y: ranked.map(p => p.short_label),
    customdata: ranked.map(p => [p.event_id || "", p.fund || "", p.action_type || "", p.future_12m_excess_return]),
    hovertemplate: "%{y}<br>positive excess probability：%{x:.1%}<br>event：%{customdata[0]}<br>fund：%{customdata[1]}<br>action：%{customdata[2]}<extra></extra>"
  }], {
    title: "Part6：Top manager-action events by predicted positive excess probability",
    height: 430,
    margin: { l: 170, r: 36, t: 58, b: 52 },
    xaxis: { title: "Predicted probability", tickformat: ".0%", range: [0, 1] },
    yaxis: { automargin: true }
  }, { displaylogo: false, responsive: true });
}

function drawPart6ProbabilityHistogram(predictions) {
  const node = dom.part6ProbabilityHistogramChart;
  if (!node || !window.Plotly) return;
  if (!predictions.length) { Plotly.purge(node); return; }
  Plotly.react(node, [{
    type: "histogram",
    x: predictions.map(p => p.prediction_probability),
    xbins: { start: 0, end: 1, size: 0.05 },
    hovertemplate: "probability bin：%{x:.1%}<br>count：%{y}<extra></extra>"
  }], {
    title: "Part6：Prediction probability distribution",
    height: 430,
    margin: { l: 58, r: 28, t: 58, b: 52 },
    xaxis: { title: "Predicted positive excess probability", tickformat: ".0%", range: [0, 1] },
    yaxis: { title: "Event count" },
    shapes: [{
      type: "line", xref: "x", yref: "paper", x0: 0.5, x1: 0.5, y0: 0, y1: 1,
      line: { dash: "dot", width: 2 }
    }]
  }, { displaylogo: false, responsive: true });
}


function part6StockActionRowsForDisplay(predictions) {
  const payloadRows = (((state.part6.lastPayload || {}).part5 || {}).increased_stock_action_rows) || [];
  const predMap = new Map();
  for (const p of predictions || []) {
    predMap.set(`${String(p.crsp_portno)}|${String(p.report_date)}`, p);
  }

  return payloadRows.map(row => {
    const pred = predMap.get(`${String(row.crsp_portno)}|${String(row.report_dt)}`) || {};
    return {
      ...row,
      manager: pred.manager || "",
      model_action_type: pred.action_type || "",
      prediction_probability: pred.prediction_probability,
      future_12m_excess_return: pred.future_12m_excess_return,
      label_positive_excess_12m: pred.label_positive_excess_12m
    };
  }).sort((a, b) => Math.abs(b.delta_holding_pct || 0) - Math.abs(a.delta_holding_pct || 0));
}

function drawPart6StockActionChart(rows) {
  const node = dom.part6StockActionChart;
  if (!node || !window.Plotly) return;
  const data = (rows || []).filter(row => row.stock_action_direction === "increase" || row.stock_action_direction === "new_position").slice(0, 25).reverse();
  if (!data.length) {
    Plotly.react(node, [], {
      title: "Part6：Actual increased stocks from Part5 JSON（目前沒有可顯示的增加持股）",
      height: 360
    }, { displaylogo: false, responsive: true });
    return;
  }

  Plotly.react(node, [{
    type: "bar",
    orientation: "h",
    x: data.map(row => row.delta_holding_pct),
    y: data.map(row => `${row.holding_ticker || row.holding_key} | ${row.report_dt}`),
    customdata: data.map(row => [row.fund_ticker, row.holding_security_name, row.sector, row.prediction_probability, row.future_12m_excess_return, row.model_action_type]),
    hovertemplate: "股票：%{y}<br>基金：%{customdata[0]}<br>名稱：%{customdata[1]}<br>Sector：%{customdata[2]}<br>增加比例：%{x:.2%}<br>ML positive excess prob：%{customdata[3]:.1%}<br>historical future excess：%{customdata[4]:.2%}<br>model action：%{customdata[5]}<extra></extra>"
  }], {
    title: "Part6：Part5 actual increased stocks linked to ML future outcome",
    height: 430,
    margin: { l: 190, r: 32, t: 58, b: 52 },
    xaxis: { title: "Increase in holding weight", tickformat: ".1%" },
    yaxis: { automargin: true }
  }, { displaylogo: false, responsive: true });
}

function drawPart6ShapFeatureChart(shapAgg) {
  const node = dom.part6ShapFeatureChart;
  if (!node || !window.Plotly) return;
  if (!shapAgg.length) { Plotly.purge(node); return; }
  Plotly.react(node, [{
    type: "bar",
    orientation: "h",
    x: shapAgg.map(row => row.total_abs_contribution),
    y: shapAgg.map(row => row.feature),
    customdata: shapAgg.map(row => [row.mean_contribution, row.positive_sum, row.negative_sum, row.count]),
    hovertemplate: "%{y}<br>total |contribution|：%{x:.4f}<br>mean contribution：%{customdata[0]:.4f}<br>positive sum：%{customdata[1]:.4f}<br>negative sum：%{customdata[2]:.4f}<br>appearances：%{customdata[3]}<extra></extra>"
  }], {
    title: "Part6 SHAP：Global top features across returned events",
    height: 430,
    margin: { l: 220, r: 32, t: 58, b: 52 },
    xaxis: { title: "Total absolute contribution" },
    yaxis: { automargin: true }
  }, { displaylogo: false, responsive: true });
}

function drawPart6SingleEventShapChart(shapRows) {
  const node = dom.part6SingleEventShapChart;
  if (!node || !window.Plotly) return;
  if (!shapRows.length) { Plotly.purge(node); return; }
  part6SyncShapEventPicker(shapRows);
  const selectedIds = part6SelectedShapEventIds();
  const selectedRows = shapRows.filter(row => selectedIds.includes(String(row.event_id || '')));
  const featureTotals = new Map();
  selectedRows.forEach(row => featureTotals.set(row.feature, (featureTotals.get(row.feature) || 0) + Math.abs(row.contribution)));
  const features = [...featureTotals.entries()].sort((a, b) => a[1] - b[1]).slice(-14).map(item => item[0]);
  const eventMap = new Map();
  selectedRows.forEach(row => {
    const key = String(row.event_id || '');
    if (!eventMap.has(key)) eventMap.set(key, { label: row.event_label, rows: new Map() });
    eventMap.get(key).rows.set(row.feature, row);
  });
  const palette = ['#2187d5', '#1f9d8a', '#f0a44b', '#7655a6', '#c94c4c', '#4f77aa', '#59a14f', '#edc948'];
  const traces = [...eventMap.entries()].map(([eventId, event], index) => ({
    type: 'bar', orientation: 'h', name: event.label,
    x: features.map(feature => (event.rows.get(feature) || {}).contribution || 0),
    y: features,
    marker: { color: palette[index % palette.length] },
    customdata: features.map(feature => {
      const row = event.rows.get(feature) || {};
      return [row.value, event.label, eventId, row.contribution == null ? 'not in top features' : 'reported'];
    }),
    hovertemplate: '%{customdata[1]}<br>%{y}<br>value：%{customdata[0]}<br>contribution：%{x:.4f}<br>%{customdata[3]}<extra></extra>'
  }));
  Plotly.react(node, traces, {
    title: `Part6 SHAP：${eventMap.size} event comparison（positive / negative contributions）`,
    height: Math.max(480, 240 + features.length * 30), barmode: 'group',
    margin: { l: 220, r: 32, t: 58, b: 52 },
    xaxis: { title: "Contribution to predicted positive excess probability", zeroline: true },
    yaxis: { automargin: true },
    legend: { orientation: 'h', y: -0.18 }
  }, { displaylogo: false, responsive: true });
}

function part6ShapEventOptions(shapRows) {
  const events = new Map();
  shapRows.forEach(row => {
    const id = String(row.event_id || '');
    if (id && !events.has(id)) events.set(id, row.event_label || id);
  });
  return [...events.entries()].map(([event_id, label]) => ({ event_id, label }));
}

function part6SelectedShapEventIds() {
  if (!dom.part6ShapEventSelect) return [];
  return Array.from(dom.part6ShapEventSelect.selectedOptions).map(option => option.value).filter(Boolean).slice(0, 8);
}

function part6SyncShapEventPicker(shapRows) {
  if (!dom.part6ShapEventSelect) return;
  const events = part6ShapEventOptions(shapRows);
  const validIds = new Set(events.map(event => event.event_id));
  let selected = part6SelectedShapEventIds().filter(id => validIds.has(id));
  if (!selected.length && events.length) selected = [events[0].event_id];
  dom.part6ShapEventSelect.innerHTML = events.map(event => `<option value="${escapeHtml(event.event_id)}"${selected.includes(event.event_id) ? ' selected' : ''}>${escapeHtml(event.label)}</option>`).join('');
  if (dom.part6ShapSelectionHint) dom.part6ShapSelectionHint.textContent = `已選 ${selected.length} 個事件；共同顯示跨事件總 |SHAP| 最高的 14 個 features。`;
}

function part6ApplyShapEventSelection() {
  if (!state.part6.backendResult) return;
  const selected = part6SelectedShapEventIds();
  Array.from(dom.part6ShapEventSelect.options).forEach(option => { option.selected = selected.includes(option.value); });
  const shapRows = backendShapRows(state.part6.backendResult);
  drawPart6SingleEventShapChart(shapRows);
  part7PopulateEvents(selected);
}

function renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows = []) {
  renderTable("part6PredictionResultTable", predictions.slice().sort((a, b) => b.prediction_probability - a.prediction_probability).slice(0, 50), [
    { key: "event_id", label: "event_id" },
    { key: "manager", label: "manager" },
    { key: "fund", label: "fund" },
    { key: "report_date", label: "report_date" },
    { key: "action_type", label: "action_type" },
    { key: "market_regime", label: "market_regime" },
    { key: "prediction_probability", label: "positive excess probability", format: "pct" },
    { key: "label_positive_excess_12m", label: "historical label" },
    { key: "future_12m_excess_return", label: "future excess", format: "pct" }
  ], { title: "Part6 backend ML prediction events", expanded: true, count: predictions.length });



  renderTable("part6StockActionTable", (stockActionRows || []).slice(0, 120), [
    { key: "report_dt", label: "report_date" },
    { key: "fund_ticker", label: "fund" },
    { key: "holding_ticker", label: "ticker" },
    { key: "holding_security_name", label: "security" },
    { key: "sector", label: "sector" },
    { key: "stock_action_direction", label: "actual stock action" },
    { key: "previous_holding_pct", label: "previous weight", format: "pct" },
    { key: "current_holding_pct", label: "current weight", format: "pct" },
    { key: "delta_holding_pct", label: "delta weight", format: "pct" },
    { key: "stock_beta", label: "stock beta", format: "num" },
    { key: "prediction_probability", label: "ML positive prob", format: "pct" },
    { key: "future_12m_excess_return", label: "future excess", format: "pct" },
    { key: "label_positive_excess_12m", label: "historical label" }
  ], { title: "Part6 actual increased/decreased stocks from Part5 JSON linked to model outcomes", expanded: true, count: (stockActionRows || []).length });

  renderTable("part6ShapResultTable", shapRows.slice().sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 100), [
    { key: "event_id", label: "event_id" },
    { key: "event_label", label: "event" },
    { key: "direction", label: "direction" },
    { key: "feature", label: "feature" },
    { key: "value", label: "value", format: "num" },
    { key: "contribution", label: "contribution", format: "num" }
  ], { title: `Part6 SHAP local explanations | global features=${shapAgg.length}`, expanded: false, count: shapRows.length });
}

function setPart5Status(text) {
  dom.part5Status.textContent = text;
}

async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return response.text();
}

function resetAll() {
  state.rows = [];
  state.activeRows = [];
  state.selectedMgmt = new Set();
  state.latestP1Box = null;
  state.p1BoxA = null;
  state.p1BoxB = null;
  state.p1Applied = false;
  state.rawA = [];
  state.rawB = [];
  state.latestP2Region = null;
  state.pendingP2Regions = [];
  state.appliedP2Regions = [];
  state.selectedFeatures = {};
  state.part2TablesA = null;
  state.part2TablesB = null;
  state.part3RawA = [];
  state.part3RawB = [];
  state.latestManagers = [];
  state.pendingManagers = new Set();
  state.radarRecords = [];
  state.part4View = "indicator";
  document.querySelectorAll("input[name='part4View']").forEach(input => {
    input.checked = input.value === state.part4View;
  });
  dom.mgmtFilter.innerHTML = "";
  dom.part2Grid.innerHTML = "";
  dom.managerTable.innerHTML = "";
  dom.groupTable.innerHTML = "";
  dom.recordTable.innerHTML = "";
  dom.part2Section.classList.add("hidden");
  dom.part3Section.classList.add("hidden");
  dom.part4Section.classList.add("hidden");
  if (dom.part6Section) dom.part6Section.classList.add("hidden");
  state.part6 = { mode: "backend", backendStatus: "idle", backendResult: null };
  if (dom.part6ModeSelect) dom.part6ModeSelect.value = "backend";
  if (dom.backendAnalysisStatus) { dom.backendAnalysisStatus.textContent = "尚未送出 Part1–Part5 state。"; dom.backendAnalysisStatus.className = "inline-status backend-status"; }
  if (dom.backendAnalysisSummary) dom.backendAnalysisSummary.innerHTML = "";
  ["part6PredictionResultTable", "part6ShapResultTable"].forEach(id => { if (dom[id]) dom[id].innerHTML = ""; });
  ["metricBackendPredictionCount", "metricBackendAvgProb", "metricBackendHighProb", "metricBackendTopProb"].forEach(id => { if (dom[id]) dom[id].textContent = "-"; });
  resetPart5Data(true);
  updatePart1Status("尚未框選");
  updatePart1Metrics();
  setProgress(0);
  if (window.Plotly) {
    Plotly.purge("scatterPlot");
    Plotly.purge("managerChart");
    Plotly.purge("indicatorChart");
    Plotly.purge("radarChart");
    if (dom.styleDriftChart) Plotly.purge("styleDriftChart");
    Plotly.purge("part5OverviewChart");
    Plotly.purge("part5AllocationChart");
    Plotly.purge("part5TopHoldingsChart");
    Plotly.purge("part5BOverviewChart");
    Plotly.purge("part5BYearChart");
    Plotly.purge("part5BRateCaseChart");
    [
      "scatterPlot", "managerChart", "indicatorChart", "radarChart", "styleDriftChart",
      "part5OverviewChart", "part5AllocationChart", "part5TopHoldingsChart",
      "part5BOverviewChart", "part5BYearChart", "part5BRateCaseChart", "part5BEquityCaseChart",
      "part6BucketChart", "part6ScatterChart", "part6PortfolioChart",
    "part6PredictionRankChart", "part6ProbabilityHistogramChart", "part6ShapFeatureChart", "part6SingleEventShapChart"
    ].forEach(id => {
      const node = document.getElementById(id);
      if (node) Plotly.purge(node);
    });
  }
}

function getCurrentSelectionModeFromDom() {
  const checked = document.querySelector("input[name='selectionMode']:checked");
  return checked ? checked.value : state.selectionMode || "single";
}

function updateControlStates() {
  state.selectionMode = getCurrentSelectionModeFromDom();

  // 不再 disable「設為 B」。
  // 因為 setRegionB() 內部已經會判斷目前是不是 A/B 比較模式。
  // 這樣可以避免 radio state 與 button disabled state 不同步。
  if (dom.setBBtn) {
    dom.setBBtn.disabled = false;
  }
}

function updateManualBenchmark() {
  const annual = parseNumber(dom.manualSp500Input.value);
  state.manualMonthlySp500 = annualToMonthly(Number.isFinite(annual) ? annual : 0.1);
}

function setStatus(text) {
  dom.loadStatus.textContent = text;
}

function setProgress(percent) {
  dom.progressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function addBarTrace(traces, centers, widths, counts, name, color) {
  if (!counts) return;
  traces.push({
    type: "bar",
    x: centers,
    y: counts,
    width: widths,
    name,
    opacity: 0.96,
    marker: { color, line: { color, width: 0.2 } },
    hoverinfo: "skip"
  });
}

function histogramCounts(rows, key, edges, cumulative) {
  if (!rows || !rows.length) return null;
  const counts = new Array(edges.length - 1).fill(0);
  let any = false;

  for (const row of rows) {
    const value = row[key];
    if (!Number.isFinite(value) || value < edges[0] || value > edges[edges.length - 1]) continue;
    let idx = binarySearchBin(edges, value);
    if (idx < 0) continue;
    if (idx >= counts.length) idx = counts.length - 1;
    counts[idx] += 1;
    any = true;
  }

  if (!any) return null;
  if (cumulative) {
    for (let i = 1; i < counts.length; i += 1) counts[i] += counts[i - 1];
  }
  return counts;
}

function sharedHistEdges(tables, key, bins) {
  const values = [];
  for (const table of tables) {
    if (!table || !table.length) continue;
    for (const row of table) {
      const value = row[key];
      if (Number.isFinite(value)) values.push(value);
    }
  }
  if (!values.length) return null;

  values.sort((a, b) => a - b);
  let lo = percentileSorted(values, 0.005);
  let hi = percentileSorted(values, 0.995);
  if (lo === hi) {
    lo = values[0];
    hi = values[values.length - 1];
  }
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * 0.1, 0.5);
    lo -= pad;
    hi += pad;
  }

  const step = (hi - lo) / bins;
  const edges = [];
  for (let i = 0; i <= bins; i += 1) edges.push(lo + step * i);
  return edges;
}

function boxFromPlotlySelection(eventData) {
  if (!eventData) return null;
  if (eventData.range && eventData.range.x && eventData.range.y) {
    const [x0, x1] = sortedPair(eventData.range.x);
    const [y0, y1] = sortedPair(eventData.range.y);
    return { x0, x1, y0, y1 };
  }
  if (!eventData.points || !eventData.points.length) return null;

  const xs = [];
  const ys = [];
  for (const point of eventData.points) {
    if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
      xs.push(point.x);
      ys.push(point.y);
    }
  }
  if (!xs.length) return null;
  return {
    x0: Math.min(...xs),
    x1: Math.max(...xs),
    y0: Math.min(...ys),
    y1: Math.max(...ys)
  };
}

function xRangeFromPlotlySelection(eventData) {
  if (!eventData) return null;
  if (eventData.range && eventData.range.x) {
    return sortedPair(eventData.range.x);
  }
  if (!eventData.points || !eventData.points.length) return null;
  const xs = eventData.points.map(point => point.x).filter(Number.isFinite);
  if (!xs.length) return null;
  return [Math.min(...xs), Math.max(...xs)];
}

function rectShape(box, color, fill) {
  const [x0, x1] = sortedPair([box.x0, box.x1]);
  const [y0, y1] = sortedPair([box.y0, box.y1]);
  return {
    type: "rect",
    xref: "x",
    yref: "y",
    x0,
    x1,
    y0,
    y1,
    line: { color, width: 3 },
    fillcolor: fill,
    layer: "above"
  };
}

function filterByBox(rows, box) {
  if (!box) return [];
  const [x0, x1] = sortedPair([box.x0, box.x1]);
  const [y0, y1] = sortedPair([box.y0, box.y1]);
  return rows.filter(row => row.x_ret >= x0 && row.x_ret <= x1 && row.y_ret >= y0 && row.y_ret <= y1);
}

function niceAxisRange(rows, key) {
  if (!rows.length) return null;
  const values = rows.map(row => row[key]).filter(Number.isFinite).sort((a, b) => a - b);
  if (!values.length) return null;
  let lo = percentileSorted(values, 0.005);
  let hi = percentileSorted(values, 0.995);
  if (lo === hi) {
    const pad = Math.max(Math.abs(lo) * 0.1, 0.01);
    return [lo - pad, hi + pad];
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function countManagers(rows) {
  const map = new Map();
  for (const row of rows) {
    const manager = row.mgr_name || "Unknown Manager";
    map.set(manager, (map.get(manager) || 0) + 1);
  }
  return map;
}

function mergeManagerCounts(countA, countB, compareMode) {
  const names = new Set([...countA.keys(), ...countB.keys()]);
  return Array.from(names).map(manager => {
    const a = countA.get(manager) || 0;
    const b = countB.get(manager) || 0;
    let group = "A";
    let color = COLORS.aDark;
    if (compareMode && a > 0 && b > 0) {
      group = "A&B";
      color = COLORS.both;
    } else if (compareMode && b > 0) {
      group = "B";
      color = COLORS.bDark;
    }
    return { manager, countA: a, countB: b, total: a + b, group, color };
  }).sort((a, b) => b.total - a.total || a.manager.localeCompare(b.manager));
}

function part2StatusText() {
  const bits = [];
  if (state.latestP2Region) bits.push(`目前：${state.latestP2Region.label}`);
  bits.push(`待套用：${state.pendingP2Regions.length}`);
  bits.push(`已套用：${state.appliedP2Regions.length}`);
  return bits.join(" / ");
}

function part3StatusText() {
  return `目前選取：${state.latestManagers.length} 位 / 待加入：${state.pendingManagers.size} 位`;
}

function regionSignature(region) {
  return `${region.level}|${region.feature}|${region.xRange.map(value => round(value, 10)).join(":")}`;
}

function regionsSignature(regions) {
  return (regions || []).map(regionSignature).join(";");
}

function resetPlotlyHandler(plot, eventName, handler) {
  if (plot.removeAllListeners) plot.removeAllListeners(eventName);
  plot.on(eventName, handler);
}

function renderTable(containerOrId, rows, columns, options = {}) {
  const container = typeof containerOrId === "string" ? document.getElementById(containerOrId) : containerOrId;
  if (!container) return;
  container.innerHTML = "";
  if (!rows || !rows.length) return;

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trHead = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column.label;
    trHead.appendChild(th);
  }
  thead.appendChild(trHead);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      td.textContent = formatValue(row[column.key], column.format);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  if (!options.title) {
    container.appendChild(table);
    return;
  }

  const details = document.createElement("details");
  details.className = "table-panel";
  details.open = options.expanded !== false;

  const summary = document.createElement("summary");
  const title = document.createElement("span");
  title.className = "table-title";
  title.textContent = options.title;
  const count = document.createElement("span");
  count.className = "table-count";
  count.textContent = `${formatInt(options.count ?? rows.length)} 筆`;
  summary.append(title, count);

  const scroll = document.createElement("div");
  scroll.className = "table-scroll";
  scroll.appendChild(table);

  details.append(summary, scroll);
  container.appendChild(details);
}


function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatValue(value, format) {
  if (format === "pct") return formatPct(value);
  if (format === "yield") return formatYield(value);
  if (format === "money") return formatMoney(value);
  if (format === "years") return Number.isFinite(value) ? `${value.toFixed(2)} 年` : "-";
  if (format === "int") return Number.isFinite(value) ? formatInt(value) : "-";
  if (format === "num") return Number.isFinite(value) ? value.toFixed(4) : "-";
  if (Number.isFinite(value)) return formatNumber(value);
  return value == null || value === "" ? "-" : String(value);
}

function cleanText(value) {
  if (value == null) return "";
  const text = String(value).trim();
  if (!text || text.toUpperCase() === "NULL" || text.toUpperCase() === "NAN") return "";
  return text;
}

function parseNumber(value) {
  if (value == null) return NaN;
  if (typeof value === "number") return Number.isFinite(value) ? value : NaN;
  const text = String(value).trim();
  if (!text || text.toUpperCase() === "NULL" || text.toUpperCase() === "NAN") return NaN;
  const normalized = text.replace(/,/g, "").replace(/%$/, "");
  const number = Number(normalized);
  if (!Number.isFinite(number)) return NaN;
  return text.endsWith("%") ? number / 100 : number;
}

function parsePercentValue(value) {
  const number = parseNumber(value);
  if (!Number.isFinite(number)) return NaN;
  return Math.abs(number) >=  1 ? number / 100 : number;
}

function zeroIfMissing(value) {
  const number = parseNumber(value);
  return Number.isFinite(number) ? number : 0;
}

function parseDateMs(value) {
  const parts = parseDateParts(value);
  if (!parts) return NaN;
  return Date.UTC(parts.year, parts.month - 1, parts.day);
}

function parseDateParts(value) {
  const text = cleanText(value);
  if (!text) return null;

  let match = text.match(/^(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?/);
  if (match) {
    return {
      year: Number(match[1]),
      month: Number(match[2]),
      day: Number(match[3] || 1)
    };
  }

  match = text.match(/^(\d{4})(\d{2})(\d{2})?$/);
  if (match) {
    return {
      year: Number(match[1]),
      month: Number(match[2]),
      day: Number(match[3] || 1)
    };
  }

  return null;
}

function monthKeyFromValue(value) {
  const parts = parseDateParts(value);
  if (!parts) return "";
  return `${parts.year}-${String(parts.month).padStart(2, "0")}`;
}

function monthKeyFromMs(ms) {
  if (!Number.isFinite(ms)) return "";
  const date = new Date(ms);
  return `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
}

function datePartsFromMs(ms) {
  const date = new Date(ms);
  return {
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate()
  };
}

function isoDateFromMs(ms) {
  const parts = datePartsFromMs(ms);
  return `${parts.year}-${String(parts.month).padStart(2, "0")}-${String(parts.day).padStart(2, "0")}`;
}

function findFirst(keys, candidates) {
  return candidates.find(candidate => keys.includes(candidate)) || null;
}

function groupBy(rows, keyFn) {
  const map = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(row);
  }
  return map;
}

function finiteValues(rows, key) {
  return rows.map(row => row[key]).filter(Number.isFinite);
}

function firstDefined(rows, key) {
  const row = rows.find(item => item[key] != null && item[key] !== "");
  return row ? row[key] : "";
}

function uniqueCount(rows, keyFn) {
  const set = new Set();
  for (const row of rows) {
    const value = keyFn(row);
    if (value != null && value !== "") set.add(value);
  }
  return set.size;
}

function hasNumericValues(rows, key) {
  return rows.some(row => Number.isFinite(row[key]));
}

function mean(values) {
  const clean = values.filter(Number.isFinite);
  if (!clean.length) return NaN;
  return sum(clean) / clean.length;
}

function sum(values) {
  return values.filter(Number.isFinite).reduce((total, value) => total + value, 0);
}

function sampleStd(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length <= 1) return NaN;
  const avg = mean(clean);
  const variance = clean.reduce((total, value) => total + (value - avg) ** 2, 0) / (clean.length - 1);
  return Math.sqrt(variance);
}

function safeLog1p(value) {
  if (!Number.isFinite(value) || value <= -0.999999) return NaN;
  return Math.log1p(value);
}

function annualizeLogSum(logSum, count) {
  if (!Number.isFinite(logSum) || !count) return NaN;
  const annualLog = logSum * (12 / count);
  if (annualLog > 700) return NaN;
  if (annualLog < -700) return -1;
  const value = Math.expm1(annualLog);
  return Number.isFinite(value) ? value : NaN;
}

function safeExpm1(logSum) {
  if (!Number.isFinite(logSum)) return NaN;
  if (logSum > 700) return NaN;
  if (logSum < -700) return -1;
  const value = Math.expm1(logSum);
  return Number.isFinite(value) ? value : NaN;
}

function safeAnnualizedReturnFromMonthlyMean(meanMonthly) {
  if (!Number.isFinite(meanMonthly) || meanMonthly <= -0.999999) return NaN;
  const annualLog = 12 * Math.log1p(meanMonthly);
  if (annualLog > 700) return NaN;
  if (annualLog < -700) return -1;
  const value = Math.expm1(annualLog);
  return Number.isFinite(value) ? value : NaN;
}

function safeCompoundReturn(values) {
  const clean = values.filter(value => Number.isFinite(value) && value > -0.999999);
  if (!clean.length) return NaN;
  const logSum = clean.reduce((total, value) => total + Math.log1p(value), 0);
  if (logSum > 700) return NaN;
  if (logSum < -700) return -1;
  const value = Math.expm1(logSum);
  return Number.isFinite(value) ? value : NaN;
}

function maxDrawdownFromMonthly(values) {
  const clean = values.filter(value => Number.isFinite(value) && value > -0.999999);
  if (!clean.length) return NaN;
  let wealthLog = 0;
  let peak = 0;
  let minDrawdown = 0;

  for (const value of clean) {
    wealthLog += Math.log1p(value);
    if (wealthLog > peak) peak = wealthLog;
    const drawdown = Math.exp(Math.max(-700, Math.min(0, wealthLog - peak))) - 1;
    if (drawdown < minDrawdown) minDrawdown = drawdown;
  }

  return minDrawdown;
}

function safeCorr(a, b) {
  const pairs = pairedFinite(a, b);
  if (pairs.length < 2) return NaN;
  const xs = pairs.map(pair => pair[0]);
  const ys = pairs.map(pair => pair[1]);
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < xs.length; i += 1) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  const denom = Math.sqrt(vx * vy);
  return denom > 0 ? num / denom : NaN;
}

function safeBeta(a, b) {
  const pairs = pairedFinite(a, b);
  if (pairs.length < 2) return NaN;
  const xs = pairs.map(pair => pair[0]);
  const ys = pairs.map(pair => pair[1]);
  const mx = mean(xs);
  const my = mean(ys);
  let cov = 0;
  let variance = 0;
  for (let i = 0; i < xs.length; i += 1) {
    cov += (xs[i] - mx) * (ys[i] - my);
    variance += (ys[i] - my) ** 2;
  }
  cov /= xs.length;
  variance /= xs.length;
  return variance > 0 ? cov / variance : NaN;
}

function pairedFinite(a, b) {
  const out = [];
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i += 1) {
    if (Number.isFinite(a[i]) && Number.isFinite(b[i])) out.push([a[i], b[i]]);
  }
  return out;
}

function captureRatio(returns, benchmark, upside) {
  const pairs = pairedFinite(returns, benchmark).filter(pair => upside ? pair[1] > 0 : pair[1] < 0);
  if (!pairs.length) return NaN;
  const avgBench = mean(pairs.map(pair => pair[1]));
  if (!avgBench) return NaN;
  return mean(pairs.map(pair => pair[0])) / avgBench;
}

function skewness(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length < 3) return NaN;
  const avg = mean(clean);
  const centered = clean.map(value => value - avg);
  const m2 = mean(centered.map(value => value ** 2));
  if (m2 <= 0) return NaN;
  const m3 = mean(centered.map(value => value ** 3));
  return m3 / (m2 ** 1.5);
}

function kurtosis(values) {
  const clean = values.filter(Number.isFinite);
  if (clean.length < 4) return NaN;
  const avg = mean(clean);
  const centered = clean.map(value => value - avg);
  const m2 = mean(centered.map(value => value ** 2));
  if (m2 <= 0) return NaN;
  const m4 = mean(centered.map(value => value ** 4));
  return m4 / (m2 ** 2) - 3;
}

function percentile(values, p) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  return percentileSorted(clean, p);
}

function percentileSorted(sorted, p) {
  if (!sorted.length) return NaN;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const weight = idx - lo;
  return sorted[lo] * (1 - weight) + sorted[hi] * weight;
}

function percentileRankAverage(values, value) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!Number.isFinite(value) || clean.length <= 1) return 0.5;

  let first = -1;
  let last = -1;
  for (let i = 0; i < clean.length; i += 1) {
    if (clean[i] === value) {
      if (first === -1) first = i;
      last = i;
    }
  }
  if (first === -1) {
    const below = clean.filter(item => item < value).length;
    return (below + 1) / clean.length;
  }
  const avgRankOneBased = ((first + 1) + (last + 1)) / 2;
  return avgRankOneBased / clean.length;
}

function mode(values) {
  const counts = new Map();
  for (const value of values.filter(Boolean)) {
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  let best = "";
  let bestCount = -1;
  for (const [value, count] of counts.entries()) {
    if (count > bestCount) {
      best = value;
      bestCount = count;
    }
  }
  return best;
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 0 ? dot / denom : 0;
}

function annualToMonthly(annual) {
  if (!Number.isFinite(annual) || annual < -1) return NaN;
  return (1 + annual) ** (1 / 12) - 1;
}

function sortedPair(values) {
  const a = Number(values[0]);
  const b = Number(values[1]);
  return a <= b ? [a, b] : [b, a];
}

function binarySearchBin(edges, value) {
  let lo = 0;
  let hi = edges.length - 1;
  if (value === edges[hi]) return hi - 1;
  while (lo < hi - 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (value < edges[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function formatInt(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : "-";
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1000000) return value.toExponential(3);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (abs >= 1) return value.toFixed(4);
  return value.toFixed(6);
}

function formatMoney(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toFixed(2);
}

function formatYield(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}%` : "-";
}

function formatPct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "-";
}


/* =========================================================
   Hotfix 20260708: robust Part4 Radar / Style Drift rendering
   and Part6 holdings-level stock action + ML outcome linkage.
   This block intentionally overrides earlier functions.
========================================================= */

function currentPart4RadioView() {
  const checked = document.querySelector("input[name='part4View']:checked");
  return checked ? checked.value : (state.part4View || "indicator");
}

function showPart4Chart(view) {
  if (dom.indicatorChart) dom.indicatorChart.classList.toggle("hidden", view !== "indicator");
  if (dom.radarChart) dom.radarChart.classList.toggle("hidden", view !== "radar");
  if (dom.styleDriftChart) dom.styleDriftChart.classList.toggle("hidden", view !== "style_drift");
}

function updatePart4View(redraw = true) {
  const view = currentPart4RadioView();
  state.part4View = view;
  showPart4Chart(view);

  if (!window.Plotly) return;
  window.setTimeout(() => {
    if (view === "indicator") {
      if (redraw && state.radarRecords && state.radarRecords.length) drawIndicatorChart(groupRadarRecords(state.radarRecords));
      if (dom.indicatorChart) Plotly.Plots.resize(dom.indicatorChart);
    } else if (view === "radar") {
      if (redraw) drawRadar(false);
      if (dom.radarChart) Plotly.Plots.resize(dom.radarChart);
    } else if (view === "style_drift") {
      if (redraw) drawStyleDriftChart();
      if (dom.styleDriftChart) Plotly.Plots.resize(dom.styleDriftChart);
    }
  }, 30);
}

function drawRadar(updateViewAfterDraw = true) {
  if (!window.Plotly) return;
  if (!state.radarRecords || !state.radarRecords.length) {
    if (dom.radarChart) {
      Plotly.react(dom.radarChart, [], { title: "Part4 Radar：請先從 Part3 加入經理人", height: 520 }, { displaylogo: false, responsive: true });
    }
    return;
  }

  const groups = groupRadarRecords(state.radarRecords);
  drawIndicatorChart(groups);

  const theta = RADAR_METRICS.map(metric => metric.label);
  const thetaClosed = theta.concat(theta[0]);
  const traces = [];
  groups.forEach((group, groupIndex) => {
    group.forEach((record, recordIndex) => {
      const values = theta.map(label => Number.isFinite(record.scores && record.scores[label]) ? record.scores[label] : 0.5);
      traces.push({
        type: "scatterpolar",
        r: values.concat(values[0]),
        theta: thetaClosed,
        fill: "toself",
        opacity: 0.42,
        name: `${record.manager} | Group ${groupIndex + 1}`,
        line: { width: 2 },
        hovertemplate: `${record.manager}<br>%{theta}<br>score：%{r:.2f}<extra></extra>`
      });
    });
  });

  if (dom.radarChart) {
    dom.radarChart.classList.toggle("hidden", (state.part4View || currentPart4RadioView()) !== "radar");
    Plotly.react(dom.radarChart, traces, {
      title: "Part4：Radar scores by manager group",
      height: Math.max(620, Math.min(1100, 520 + state.radarRecords.length * 8)),
      margin: { l: 70, r: 70, t: 80, b: 70 },
      polar: { radialaxis: { visible: true, range: [0, 1], tickformat: ".0%" } },
      legend: { orientation: "h", y: -0.12 },
      showlegend: true
    }, { displaylogo: false, responsive: true });
  }

  renderTable("groupTable", groups.map((group, index) => ({
    group: `Group ${index + 1}`,
    count: group.length,
    managers: group.map(record => record.manager).join("、")
  })), [
    { key: "group", label: "群組" },
    { key: "count", label: "經理人數", format: "int" },
    { key: "managers", label: "經理人" }
  ], { title: "Part4 相似群組", expanded: true });

  const recordRows = state.radarRecords.map(record => {
    const row = { manager: record.manager, sourceLabel: record.sourceLabel };
    for (const metric of RADAR_METRICS) {
      row[`score_${metric.rawKey}`] = record.scores ? record.scores[metric.label] : null;
      row[metric.rawKey] = record.raw ? record.raw[metric.rawKey] : null;
    }
    return row;
  });
  renderTable("recordTable", recordRows, [
    { key: "manager", label: "經理人" },
    { key: "sourceLabel", label: "來源" },
    ...RADAR_METRICS.map(metric => ({ key: `score_${metric.rawKey}`, label: `${metric.label}分數`, format: "pct" })),
    ...RADAR_METRICS.map(metric => ({ key: metric.rawKey, label: metric.label, format: metric.format }))
  ], { title: "Part4 分數與原始指標", expanded: false });

  if (updateViewAfterDraw) updatePart4View(false);
}

function buildStyleDriftRowsForBackend() {
  const managerNames = Array.from(new Set((state.radarRecords || []).map(record => String(record.manager || "")).filter(Boolean)));
  if (!managerNames.length) return [];
  const managerSet = new Set(managerNames);
  let rows = (state.activeRows || []).filter(row => managerSet.has(String(row.mgr_name || "")));

  // Fallback: if exact manager matching fails, match by loose normalized manager text.
  if (!rows.length) {
    const normManagers = new Set(managerNames.map(normalizeLooseText));
    rows = (state.activeRows || []).filter(row => normManagers.has(normalizeLooseText(row.mgr_name || "")));
  }

  const groups = groupBy(rows, row => `${row.mgr_name || "Unknown Manager"}__${row.year}`);
  const out = [];
  for (const [key, group] of groups.entries()) {
    const parts = key.split("__");
    const manager = parts.slice(0, -1).join("__") || "Unknown Manager";
    const year = Number(parts[parts.length - 1]);
    const returns = finiteValues(group, "mret");
    const excess = group.filter(row => Number.isFinite(row.mret) && Number.isFinite(row.sp500_ret)).map(row => row.mret - row.sp500_ret);
    if (!returns.length) continue;
    out.push({
      manager,
      year,
      obs: group.length,
      annual_return: mean(returns),
      avg_excess: excess.length ? mean(excess) : NaN,
      annual_volatility: sampleStd(returns),
      avg_fee: mean(finiteValues(group, "exp_ratio")),
      avg_flow: mean(finiteValues(group, "net_flow")),
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_turnover: mean(finiteValues(group, "turn_ratio"))
    });
  }
  out.sort((a, b) => (a.manager || "").localeCompare(b.manager || "") || a.year - b.year);
  return out;
}

function drawStyleDriftChart() {
  if (!window.Plotly || !dom.styleDriftChart) return;
  const rows = buildStyleDriftRowsForBackend();
  if (!rows.length) {
    Plotly.react(dom.styleDriftChart, [], {
      title: "Part4 Style Drift：目前沒有可畫的年度經理人資料，請先在 Part3 加入經理人。",
      height: 520
    }, { displaylogo: false, responsive: true });
    return;
  }
  const managers = Array.from(new Set(rows.map(row => row.manager))).slice(0, 24);
  const traces = [];
  for (const manager of managers) {
    const g = rows.filter(row => row.manager === manager).sort((a, b) => a.year - b.year);
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: g.map(row => row.year),
      y: g.map(row => row.annual_return),
      name: `${manager} return`,
      customdata: g.map(row => [row.avg_excess, row.annual_volatility, row.avg_fee, row.avg_flow, row.avg_mtna, row.obs]),
      hovertemplate: "經理人：%{fullData.name}<br>年份：%{x}<br>3Y年化報酬：%{y:.2%}<br>3Y超額：%{customdata[0]:.2%}<br>波動：%{customdata[1]:.2%}<br>費用：%{customdata[2]:.2%}<br>Flow：%{customdata[3]:,.0f}<br>MTNA：%{customdata[4]:,.0f}<br>obs：%{customdata[5]}<extra></extra>"
    });
  }
  Plotly.react(dom.styleDriftChart, traces, {
    title: "Part4：Style Drift timeline — selected managers across years",
    height: Math.max(560, Math.min(980, 420 + managers.length * 16)),
    margin: { l: 70, r: 32, t: 70, b: 70 },
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "Trailing 3Y annualized return", tickformat: ".1%" },
    hovermode: "closest",
    legend: { orientation: "h", y: -0.18 }
  }, { displaylogo: false, responsive: true });
}

function part6PredictionKey(portno, dateValue) {
  const port = String(portno || "").replace(/\.0$/, "").trim();
  const date = String(dateValue || "").slice(0, 10);
  return `${port}|${date}`;
}

function outcomeLabelText(row) {
  const label = Number(row.label_positive_excess_12m);
  if (label === 1) return "Good: future 12M positive excess";
  if (label === 0) return "Bad: future 12M negative excess";
  const v = row.future_12m_excess_return;
  if (Number.isFinite(v)) return v > 0 ? "Good: future 12M positive excess" : "Bad: future 12M negative excess";
  return "Outcome unknown";
}

function part6StockActionRowsForDisplay(predictions) {
  const predMap = new Map();
  for (const p of predictions || []) {
    predMap.set(part6PredictionKey(p.crsp_portno, p.report_date), p);
  }

  let payloadRows = (((state.part6.lastPayload || {}).part5 || {}).stock_action_rows) || [];

  // If the saved payload did not include stock actions, derive them from prediction event report keys.
  if ((!payloadRows || !payloadRows.length) && state.part5 && state.part5.loaded && predictions && predictions.length) {
    const keys = predictions.map(p => part6PredictionKey(p.crsp_portno, p.report_date));
    payloadRows = buildPart5StockActionRowsForBackend(keys);
  }

  return (payloadRows || []).map(row => {
    const pred = predMap.get(part6PredictionKey(row.crsp_portno, row.report_dt)) || {};
    const future = Number(pred.future_12m_excess_return);
    return {
      ...row,
      manager: pred.manager || "",
      model_action_type: pred.action_type || "",
      market_regime: pred.market_regime || "",
      prediction_probability: pred.prediction_probability,
      future_12m_excess_return: pred.future_12m_excess_return,
      label_positive_excess_12m: pred.label_positive_excess_12m,
      linked_event_id: pred.event_id || "",
      outcome_text: outcomeLabelText(pred),
      signed_delta_abs: Math.abs(Number(row.delta_holding_pct) || 0),
      result_bucket: Number.isFinite(future) ? (future > 0 ? "good" : "bad") : "unknown"
    };
  }).sort((a, b) => {
    const ap = Number.isFinite(a.prediction_probability) ? a.prediction_probability : -1;
    const bp = Number.isFinite(b.prediction_probability) ? b.prediction_probability : -1;
    if (bp !== ap) return bp - ap;
    return (b.signed_delta_abs || 0) - (a.signed_delta_abs || 0);
  });
}

function drawPart6StockActionChart(rows) {
  const node = dom.part6StockActionChart;
  if (!node || !window.Plotly) return;
  const data = (rows || []).filter(row => row.stock_action_direction).slice(0, 40).reverse();
  if (!data.length) {
    Plotly.react(node, [], {
      title: "Part6：Actual stock increase/decrease from Part5 JSON（尚未取得可對齊的持股變化）",
      height: 400,
      annotations: [{ text: "請先在 Part5 選取基金報告，或讓 backend 回傳與 Part5 report date 可對齊的 ML events。", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false }]
    }, { displaylogo: false, responsive: true });
    return;
  }

  Plotly.react(node, [{
    type: "bar",
    orientation: "h",
    x: data.map(row => row.delta_holding_pct),
    y: data.map(row => `${row.holding_ticker || row.holding_key} | ${row.report_dt}`),
    text: data.map(row => row.stock_action_direction === "decrease" ? "減碼" : (row.stock_action_direction === "new_position" ? "新增" : "加碼")),
    customdata: data.map(row => [
      row.fund_ticker || row.crsp_portno,
      row.holding_security_name,
      row.sector,
      row.stock_action_direction,
      row.model_action_type,
      row.prediction_probability,
      row.future_12m_excess_return,
      row.outcome_text,
      row.previous_holding_pct,
      row.current_holding_pct,
      row.market_regime
    ]),
    hovertemplate: "股票：%{y}<br>基金：%{customdata[0]}<br>名稱：%{customdata[1]}<br>Sector：%{customdata[2]}<br>實際持股動作：%{customdata[3]}<br>model action type：%{customdata[4]}<br>前期權重：%{customdata[8]:.2%}<br>本期權重：%{customdata[9]:.2%}<br>delta：%{x:.2%}<br>ML positive prob：%{customdata[5]:.1%}<br>future 12M excess：%{customdata[6]:.2%}<br>結果：%{customdata[7]}<br>market：%{customdata[10]}<extra></extra>",
    marker: {
      color: data.map(row => row.result_bucket === "good" ? "#2d9a67" : (row.result_bucket === "bad" ? "#df6b57" : "#637083")),
      line: { color: data.map(row => row.stock_action_direction === "decrease" ? "#7a1f1b" : "#174a7c"), width: 1 }
    }
  }], {
    title: "Part6：Actual stock add/reduce actions linked to ML prediction and future 12M result",
    height: Math.max(480, Math.min(900, 280 + data.length * 20)),
    margin: { l: 210, r: 44, t: 68, b: 60 },
    xaxis: { title: "Holding weight change: negative = reduce, positive = add/new", tickformat: ".1%", zeroline: true },
    yaxis: { automargin: true },
    showlegend: false
  }, { displaylogo: false, responsive: true });
}

function renderPart6BackendVisuals(result) {
  const predictions = backendPredictions(result);
  const shapRows = backendShapRows(result);
  const shapAgg = aggregateBackendShap(shapRows, 14);

  const avgProb = predictions.length ? mean(predictions.map(p => p.prediction_probability)) : NaN;
  const highCount = predictions.filter(p => p.prediction_probability >= 0.6).length;
  const topProb = predictions.length ? Math.max(...predictions.map(p => p.prediction_probability)) : NaN;

  if (dom.metricBackendPredictionCount) dom.metricBackendPredictionCount.textContent = predictions.length ? formatInt(predictions.length) : "-";
  if (dom.metricBackendAvgProb) dom.metricBackendAvgProb.textContent = Number.isFinite(avgProb) ? formatPct(avgProb) : "-";
  if (dom.metricBackendHighProb) dom.metricBackendHighProb.textContent = predictions.length ? formatInt(highCount) : "-";
  if (dom.metricBackendTopProb) dom.metricBackendTopProb.textContent = Number.isFinite(topProb) ? formatPct(topProb) : "-";

  const stockActionRows = part6StockActionRowsForDisplay(predictions);

  drawPart6PredictionRankChart(predictions);
  drawPart6ProbabilityHistogram(predictions);
  drawPart6StockActionChart(stockActionRows);
  drawPart6ShapFeatureChart(shapAgg);
  drawPart6SingleEventShapChart(shapRows);
  renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows);
}

function renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows = []) {
  renderTable("part6PredictionResultTable", predictions.slice().sort((a, b) => b.prediction_probability - a.prediction_probability).slice(0, 50), [
    { key: "event_id", label: "event_id" },
    { key: "manager", label: "manager" },
    { key: "fund", label: "fund" },
    { key: "report_date", label: "report_date" },
    { key: "action_type", label: "model action_type" },
    { key: "market_regime", label: "market_regime" },
    { key: "prediction_probability", label: "ML positive excess probability", format: "pct" },
    { key: "label_positive_excess_12m", label: "historical label" },
    { key: "future_12m_excess_return", label: "future 12M excess", format: "pct" }
  ], { title: "Part6 backend ML prediction events", expanded: true, count: predictions.length });

  renderTable("part6StockActionTable", (stockActionRows || []).slice(0, 200), [
    { key: "report_dt", label: "report_date" },
    { key: "fund_ticker", label: "fund" },
    { key: "model_action_type", label: "model action_type" },
    { key: "holding_ticker", label: "ticker" },
    { key: "holding_security_name", label: "security" },
    { key: "sector", label: "sector" },
    { key: "stock_action_direction", label: "actual stock action" },
    { key: "previous_holding_pct", label: "previous weight", format: "pct" },
    { key: "current_holding_pct", label: "current weight", format: "pct" },
    { key: "delta_holding_pct", label: "delta weight", format: "pct" },
    { key: "stock_beta", label: "stock beta", format: "num" },
    { key: "prediction_probability", label: "ML positive prob", format: "pct" },
    { key: "future_12m_excess_return", label: "future 12M excess", format: "pct" },
    { key: "label_positive_excess_12m", label: "good/bad label" },
    { key: "outcome_text", label: "interpreted outcome" }
  ], { title: "Part6 linked table: actual stock actions from Part5 JSON + model action type + one-year outcome", expanded: true, count: (stockActionRows || []).length });

  renderTable("part6ShapResultTable", shapRows.slice().sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 100), [
    { key: "event_id", label: "event_id" },
    { key: "event_label", label: "event" },
    { key: "direction", label: "direction" },
    { key: "feature", label: "feature" },
    { key: "value", label: "value", format: "num" },
    { key: "contribution", label: "contribution", format: "num" }
  ], { title: `Part6 SHAP local explanations | global features=${shapAgg.length}`, expanded: false, count: shapRows.length });
}


/* =========================================================
   FINAL HOTFIX 20260708-2: Part4 Radar / Style Drift hard redraw
   This block overrides prior Part4 functions. It avoids rendering charts
   while hidden and uses explicit display toggles to prevent blank Plotly plots.
========================================================= */

function part4SelectedViewFinal() {
  const checked = document.querySelector("input[name='part4View']:checked");
  return checked ? checked.value : (state.part4View || "indicator");
}

function part4ShowOnlyFinal(view) {
  state.part4View = view || "indicator";
  const map = {
    indicator: dom.indicatorChart || document.getElementById("indicatorChart"),
    radar: dom.radarChart || document.getElementById("radarChart"),
    style_drift: dom.styleDriftChart || document.getElementById("styleDriftChart")
  };
  Object.entries(map).forEach(([key, el]) => {
    if (!el) return;
    const show = key === state.part4View;
    el.classList.toggle("hidden", !show);
    el.style.display = show ? "block" : "none";
    el.style.visibility = show ? "visible" : "hidden";
    if (show) {
      el.style.width = "100%";
      el.style.minHeight = key === "indicator" ? "520px" : "620px";
    }
  });
}

function updatePart4View(redraw = true) {
  const view = part4SelectedViewFinal();
  part4ShowOnlyFinal(view);
  if (!window.Plotly) return;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      try {
        if (view === "indicator") {
          if (redraw && state.radarRecords && state.radarRecords.length) {
            drawIndicatorChart(groupRadarRecords(state.radarRecords));
          }
          if (dom.indicatorChart) Plotly.Plots.resize(dom.indicatorChart);
        } else if (view === "radar") {
          drawRadar(false);
          if (dom.radarChart) Plotly.Plots.resize(dom.radarChart);
        } else if (view === "style_drift") {
          drawStyleDriftChart();
          if (dom.styleDriftChart) Plotly.Plots.resize(dom.styleDriftChart);
        }
      } catch (error) {
        const el = view === "radar" ? dom.radarChart : (view === "style_drift" ? dom.styleDriftChart : dom.indicatorChart);
        if (el) {
          el.classList.remove("hidden");
          el.style.display = "block";
          el.innerHTML = `<div class="part4-error-box">Part4 ${view} rendering error: ${escapeHtml(error.message || String(error))}</div>`;
        }
        console.error("Part4 render error", error);
      }
    });
  });
}

function drawRadar(updateViewAfterDraw = true) {
  if (!window.Plotly) return;
  const radarEl = dom.radarChart || document.getElementById("radarChart");
  if (!radarEl) return;

  radarEl.classList.remove("hidden");
  radarEl.style.display = "block";
  radarEl.style.visibility = "visible";
  radarEl.style.width = "100%";
  radarEl.style.minHeight = "660px";

  const records = state.radarRecords || [];
  if (!records.length) {
    Plotly.react(radarEl, [], {
      title: "Part4 Radar：請先在 Part3 框選經理人，按『加入目前經理人』，再按『加入 Part4』。",
      height: 620,
      annotations: [{ text: "目前 Part4 沒有 manager records", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false, font: { size: 16 } }]
    }, { displaylogo: false, responsive: true });
    return;
  }

  const groups = groupRadarRecords(records);
  const theta = RADAR_METRICS.map(metric => metric.label);
  const thetaClosed = theta.concat(theta[0]);
  const traces = [];

  groups.forEach((group, groupIndex) => {
    group.forEach(record => {
      const values = theta.map(label => {
        const value = record.scores ? Number(record.scores[label]) : NaN;
        return Number.isFinite(value) ? value : 0.5;
      });
      traces.push({
        type: "scatterpolar",
        mode: "lines+markers",
        r: values.concat(values[0]),
        theta: thetaClosed,
        fill: "toself",
        opacity: 0.48,
        name: `${record.manager} | G${groupIndex + 1}`,
        hovertemplate: `${record.manager}<br>%{theta}<br>Score: %{r:.2f}<extra></extra>`
      });
    });
  });

  Plotly.react(radarEl, traces, {
    title: "Part4：Radar scores for selected managers",
    height: Math.max(660, Math.min(1200, 560 + records.length * 10)),
    margin: { l: 72, r: 72, t: 86, b: 82 },
    polar: {
      radialaxis: { visible: true, range: [0, 1], tickformat: ".0%" },
      angularaxis: { direction: "clockwise" }
    },
    legend: { orientation: "h", y: -0.14 },
    showlegend: true
  }, { displaylogo: false, responsive: true });

  renderTable("groupTable", groups.map((group, index) => ({
    group: `Group ${index + 1}`,
    count: group.length,
    managers: group.map(record => record.manager).join("、")
  })), [
    { key: "group", label: "群組" },
    { key: "count", label: "經理人數", format: "int" },
    { key: "managers", label: "經理人" }
  ], { title: "Part4 相似群組", expanded: true });

  const recordRows = records.map(record => {
    const row = { manager: record.manager, sourceLabel: record.sourceLabel };
    for (const metric of RADAR_METRICS) {
      row[`score_${metric.rawKey}`] = record.scores ? record.scores[metric.label] : null;
      row[metric.rawKey] = record.raw ? record.raw[metric.rawKey] : null;
    }
    return row;
  });
  renderTable("recordTable", recordRows, [
    { key: "manager", label: "經理人" },
    { key: "sourceLabel", label: "來源" },
    ...RADAR_METRICS.map(metric => ({ key: `score_${metric.rawKey}`, label: `${metric.label}分數`, format: "pct" })),
    ...RADAR_METRICS.map(metric => ({ key: metric.rawKey, label: metric.label, format: metric.format }))
  ], { title: "Part4 分數與原始指標", expanded: false });

  if (updateViewAfterDraw) updatePart4View(false);
}

function buildStyleDriftRowsForBackend() {
  const records = state.radarRecords || [];
  const managerNames = Array.from(new Set(records.map(record => String(record.manager || "").trim()).filter(Boolean)));
  if (!managerNames.length) return [];

  const normalizedManagerSet = new Set(managerNames.map(normalizeLooseText));
  const rows = (state.activeRows || []).filter(row => normalizedManagerSet.has(normalizeLooseText(row.mgr_name || "")));
  const groups = groupBy(rows, row => `${cleanText(row.mgr_name) || "Unknown Manager"}__${row.year}`);
  const out = [];

  for (const [key, group] of groups.entries()) {
    const parts = key.split("__");
    const manager = parts.slice(0, -1).join("__") || "Unknown Manager";
    const year = Number(parts[parts.length - 1]);
    const returns = finiteValues(group, "mret");
    const excess = group.filter(row => Number.isFinite(row.mret) && Number.isFinite(row.sp500_ret)).map(row => row.mret - row.sp500_ret);
    if (!returns.length || !Number.isFinite(year)) continue;
    out.push({
      manager,
      year,
      obs: group.length,
      annual_return: mean(returns),
      avg_excess: excess.length ? mean(excess) : NaN,
      annual_volatility: sampleStd(returns),
      avg_fee: mean(finiteValues(group, "exp_ratio")),
      avg_flow: mean(finiteValues(group, "net_flow")),
      avg_mtna: mean(finiteValues(group, "mtna")),
      avg_turnover: mean(finiteValues(group, "turn_ratio"))
    });
  }

  out.sort((a, b) => (a.manager || "").localeCompare(b.manager || "") || a.year - b.year);
  return out;
}

function drawStyleDriftChart() {
  if (!window.Plotly) return;
  const driftEl = dom.styleDriftChart || document.getElementById("styleDriftChart");
  if (!driftEl) return;

  driftEl.classList.remove("hidden");
  driftEl.style.display = "block";
  driftEl.style.visibility = "visible";
  driftEl.style.width = "100%";
  driftEl.style.minHeight = "660px";

  const rows = buildStyleDriftRowsForBackend();
  const records = state.radarRecords || [];
  if (!records.length) {
    Plotly.react(driftEl, [], {
      title: "Part4 Style Drift：請先從 Part3 加入經理人到 Part4。",
      height: 620,
      annotations: [{ text: "目前沒有 Part4 manager records", x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false, font: { size: 16 } }]
    }, { displaylogo: false, responsive: true });
    return;
  }

  if (!rows.length) {
    const names = records.map(r => r.manager).join("、");
    Plotly.react(driftEl, [], {
      title: "Part4 Style Drift：找不到可對齊的年度 manager rows",
      height: 620,
      annotations: [{
        text: `已選經理人：${names}<br>但 activeRows 中沒有可對齊的 mgr_name/year。`,
        x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false, font: { size: 14 }
      }]
    }, { displaylogo: false, responsive: true });
    return;
  }

  const managers = Array.from(new Set(rows.map(row => row.manager))).slice(0, 24);
  const traces = [];
  for (const manager of managers) {
    const g = rows.filter(row => row.manager === manager).sort((a, b) => a.year - b.year);
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: g.map(row => row.year),
      y: g.map(row => row.annual_return),
      name: manager,
      customdata: g.map(row => [row.avg_excess, row.annual_volatility, row.avg_fee, row.avg_flow, row.avg_mtna, row.avg_turnover, row.obs]),
      hovertemplate: "經理人：%{fullData.name}<br>年份：%{x}<br>3Y年化報酬：%{y:.2%}<br>3Y超額：%{customdata[0]:.2%}<br>波動：%{customdata[1]:.2%}<br>費用：%{customdata[2]:.2%}<br>Flow：%{customdata[3]:,.0f}<br>MTNA：%{customdata[4]:,.0f}<br>Turnover：%{customdata[5]:.2%}<br>obs：%{customdata[6]}<extra></extra>"
    });
  }

  Plotly.react(driftEl, traces, {
    title: "Part4：Style Drift timeline — selected managers across years",
    height: Math.max(660, Math.min(1100, 520 + managers.length * 18)),
    margin: { l: 72, r: 32, t: 82, b: 84 },
    xaxis: { title: "Year", dtick: 1 },
    yaxis: { title: "Trailing 3Y annualized return", tickformat: ".1%" },
    hovermode: "closest",
    legend: { orientation: "h", y: -0.18 }
  }, { displaylogo: false, responsive: true });
}

// Rebind Part4 radio buttons after all function overrides are loaded.
window.addEventListener("load", () => {
  document.querySelectorAll("input[name='part4View']").forEach(input => {
    input.onchange = () => {
      state.part4View = input.value;
      updatePart4View(true);
    };
  });
});

/* =========================================================
   Final override 20260708: make Radar use the same grouped
   small-multiple layout as Indicator. Do not change Part6,
   Style Drift, SHAP, or other workflows.
========================================================= */
function drawRadar(updateViewAfterDraw = true) {
  if (!window.Plotly) return;

  const radarEl = dom.radarChart || document.getElementById("radarChart");
  if (!radarEl) return;

  radarEl.classList.remove("hidden");
  radarEl.style.display = "block";
  radarEl.style.visibility = "visible";
  radarEl.style.width = "100%";

  const records = state.radarRecords || [];
  if (!records.length) {
    Plotly.react(radarEl, [], {
      title: "Part4 Radar：請先從 Part3 加入經理人到 Part4。",
      height: 620,
      annotations: [{
        text: "目前沒有 Part4 manager records",
        x: 0.5,
        y: 0.5,
        xref: "paper",
        yref: "paper",
        showarrow: false,
        font: { size: 16 }
      }]
    }, { displaylogo: false, responsive: true });
    return;
  }

  const groups = groupRadarRecords(records);
  const cols = groups.length > 1 ? 2 : 1;
  const rows = Math.ceil(groups.length / cols);
  const rowHeightPx = 500;
  const chartHeight = Math.max(660, rows * rowHeightPx);
  const colWidth = 1 / cols;
  const rowHeight = 1 / rows;
  const xGap = Math.min(0.055, colWidth * 0.13);
  const yGap = Math.min(0.07, rowHeight * 0.2);
  const theta = RADAR_METRICS.map(metric => metric.label);
  const thetaClosed = theta.concat(theta[0]);
  const traces = [];

  const layout = {
    title: "Part4：Radar scores by manager group（同 Indicator 分組）",
    height: chartHeight,
    margin: { l: 52, r: 52, t: 88, b: 82 },
    showlegend: true,
    legend: { orientation: "h", y: -0.1, font: { size: 10 } },
    annotations: []
  };

  groups.forEach((group, groupIndex) => {
    const polarName = groupIndex === 0 ? "polar" : `polar${groupIndex + 1}`;
    const col = groupIndex % cols;
    const row = Math.floor(groupIndex / cols);
    const x0 = col * colWidth + xGap;
    const x1 = (col + 1) * colWidth - xGap;
    const yTop = 1 - row * rowHeight;
    const yBottom = 1 - (row + 1) * rowHeight;
    const y0 = yBottom + yGap;
    const y1 = yTop - yGap;

    layout[polarName] = {
      domain: { x: [x0, x1], y: [y0, y1] },
      radialaxis: {
        visible: true,
        range: [0, 1],
        tickvals: [0, 0.25, 0.5, 0.75, 1],
        ticktext: ["0%", "25%", "50%", "75%", "100%"],
        gridcolor: "#dce2e8"
      },
      angularaxis: {
        direction: "clockwise",
        gridcolor: "#edf1f5"
      }
    };

    layout.annotations.push({
      text: `Group ${groupIndex + 1}（${group.length} 位）`,
      x: (x0 + x1) / 2,
      y: Math.min(1, y1 + 0.045),
      xref: "paper",
      yref: "paper",
      showarrow: false,
      font: { size: 14, color: "#334150" }
    });

    group.forEach((record, recordIndex) => {
      const values = theta.map(label => {
        const value = record.scores ? Number(record.scores[label]) : NaN;
        return Number.isFinite(value) ? value : 0.5;
      });
      const color = INDICATOR_COLORS[recordIndex % INDICATOR_COLORS.length];
      traces.push({
        type: "scatterpolar",
        mode: "lines+markers",
        subplot: polarName,
        r: values.concat(values[0]),
        theta: thetaClosed,
        fill: "toself",
        opacity: 0.48,
        name: `${record.manager} | Group ${groupIndex + 1}`,
        legendgroup: `Group ${groupIndex + 1}`,
        line: { color, width: 2 },
        marker: { color, size: 5 },
        hovertemplate: `${record.manager}<br>%{theta}<br>分數：%{r:.2f}<extra></extra>`
      });
    });
  });

  radarEl.style.minHeight = `${chartHeight}px`;
  Plotly.react(radarEl, traces, layout, { displaylogo: false, responsive: true });
  setTimeout(() => Plotly.Plots.resize(radarEl), 80);

  renderTable("groupTable", groups.map((group, index) => ({
    group: `Group ${index + 1}`,
    count: group.length,
    managers: group.map(record => record.manager).join("、")
  })), [
    { key: "group", label: "群組" },
    { key: "count", label: "經理人數", format: "int" },
    { key: "managers", label: "經理人" }
  ], { title: "Part4 相似群組", expanded: true });

  const recordRows = records.map(record => {
    const row = { manager: record.manager, sourceLabel: record.sourceLabel };
    for (const metric of RADAR_METRICS) {
      row[`score_${metric.rawKey}`] = record.scores ? record.scores[metric.label] : null;
      row[metric.rawKey] = record.raw ? record.raw[metric.rawKey] : null;
    }
    return row;
  });

  renderTable("recordTable", recordRows, [
    { key: "manager", label: "經理人" },
    { key: "sourceLabel", label: "來源" },
    ...RADAR_METRICS.map(metric => ({ key: `score_${metric.rawKey}`, label: `${metric.label}分數`, format: "pct" })),
    ...RADAR_METRICS.map(metric => ({ key: metric.rawKey, label: metric.label, format: metric.format }))
  ], { title: "Part4 分數與原始指標", expanded: false });

  if (updateViewAfterDraw) updatePart4View(false);
}


/* =========================================================
   FINAL PATCH 20260708-part6-anchor-style
   Purpose:
   1) Treat Part5 selected report_date as Part6 anchor date.
   2) Display event-time rolling style window and style-deviation context in Part6.
   3) Keep existing SHAP charts unchanged.
========================================================= */

function part6GetNode(id) {
  return (dom && dom[id]) || document.getElementById(id);
}

function part6PredictionKey(portno, reportDate) {
  return `${String(portno || '').trim()}|${String(reportDate || '').trim()}`;
}

function part6ParseDate(dateText) {
  const value = String(dateText || '').trim();
  if (!value) return null;
  const date = new Date(`${value}T00:00:00`);
  return Number.isFinite(date.getTime()) ? date : null;
}

function part6IsoDate(date) {
  if (!date || !Number.isFinite(date.getTime())) return '';
  return date.toISOString().slice(0, 10);
}

function part6AddYears(date, years) {
  if (!date) return null;
  const d = new Date(date.getTime());
  d.setFullYear(d.getFullYear() + years);
  return d;
}

function part6AddDays(date, days) {
  if (!date) return null;
  const d = new Date(date.getTime());
  d.setDate(d.getDate() + days);
  return d;
}

function part6StyleWindowText(reportDate, years) {
  const d = part6ParseDate(reportDate);
  const y = Number.isFinite(Number(years)) ? Number(years) : 3;
  if (!d) return 'trailing 3Y before report date';
  const start = part6AddYears(d, -y);
  const end = part6AddDays(d, -1);
  return `${part6IsoDate(start)} ~ ${part6IsoDate(end)}`;
}

function part6ActiveAnchorFromResult(result, predictions) {
  const parsedActive = result && result.parsed_state && result.parsed_state.part5
    ? result.parsed_state.part5.active_report_key
    : '';
  const localActive = state && state.part5 ? state.part5.activeReportKey : '';
  const activeKey = localActive || parsedActive || '';
  if (activeKey) {
    const pred = (predictions || []).find(p => part6PredictionKey(p.crsp_portno, p.report_date) === activeKey);
    const report = state && state.part5 && state.part5.reportMap ? state.part5.reportMap.get(activeKey) : null;
    return { activeKey, prediction: pred || null, report: report || null };
  }
  const sorted = (predictions || []).slice().sort((a, b) => b.prediction_probability - a.prediction_probability);
  const pred = sorted[0] || null;
  return {
    activeKey: pred ? part6PredictionKey(pred.crsp_portno, pred.report_date) : '',
    prediction: pred,
    report: null
  };
}

function part6ExplanationMap(result) {
  const explanations = result && result.shap_result && Array.isArray(result.shap_result.explanations)
    ? result.shap_result.explanations
    : [];
  const map = new Map();
  explanations.forEach(e => {
    const featureMap = new Map();
    (e.top_positive || []).concat(e.top_negative || []).forEach(item => {
      if (item && item.feature) featureMap.set(item.feature, item);
    });
    map.set(e.event_id || part6PredictionKey(e.crsp_portno, e.report_date), { event: e, features: featureMap });
  });
  return map;
}

function part6StyleFeatureValue(pred, explanationMap, featureNames) {
  for (const name of featureNames) {
    const direct = Number(pred && pred[name]);
    if (Number.isFinite(direct)) return direct;
  }
  const info = explanationMap.get(pred && pred.event_id);
  if (info && info.features) {
    for (const name of featureNames) {
      const item = info.features.get(name);
      const v = Number(item && item.value);
      if (Number.isFinite(v)) return v;
    }
  }
  return NaN;
}

function part6EnrichedPredictions(result) {
  const explanationMap = part6ExplanationMap(result);
  return backendPredictions(result).map(p => {
    const years = Number(p.training_window_years || 3);
    const styleDeviation = part6StyleFeatureValue(p, explanationMap, ['rolling_style_deviation_score', 'style_deviation_score']);
    const styleDeviationRaw = part6StyleFeatureValue(p, explanationMap, ['style_deviation_score']);
    const rollingStyleDeviation = part6StyleFeatureValue(p, explanationMap, ['rolling_style_deviation_score']);
    return {
      ...p,
      style_window_years: Number.isFinite(years) ? years : 3,
      style_window_text: part6StyleWindowText(p.report_date, Number.isFinite(years) ? years : 3),
      style_anchor_date: p.report_date || '',
      style_deviation_score: Number.isFinite(styleDeviationRaw) ? styleDeviationRaw : styleDeviation,
      rolling_style_deviation_score: Number.isFinite(rollingStyleDeviation) ? rollingStyleDeviation : styleDeviation,
      style_deviation_display: Number.isFinite(styleDeviation) ? styleDeviation : null
    };
  });
}

function part6OutcomeText(pred) {
  const v = Number(pred && pred.future_12m_excess_return);
  if (Number.isFinite(v)) return v > 0 ? 'Good: future 12M positive excess' : 'Bad: future 12M negative excess';
  return 'Outcome unknown';
}

function part6RenderAnchorCards(result, predictions) {
  const node = part6GetNode('part6AnchorContextCards');
  if (!node) return;
  const anchor = part6ActiveAnchorFromResult(result, predictions);
  const p = anchor.prediction || predictions[0] || {};
  const reportDate = p.report_date || (anchor.report && anchor.report.report_dt) || '';
  const manager = p.manager || 'No linked prediction event yet';
  const action = p.action_type || '-';
  const prob = Number(p.prediction_probability);
  const future = Number(p.future_12m_excess_return);
  const styleDev = Number(p.rolling_style_deviation_score);
  const sectorDev = Number(p.rolling_sector_deviation_score);
  const actionDev = Number(p.rolling_action_deviation_score);
  const selectedReports = result && result.parsed_state && result.parsed_state.part5 ? Number(result.parsed_state.part5.selected_report_count) : NaN;
  const activeReportKey = anchor.activeKey || '-';

  node.innerHTML = `
    <div class="part6-anchor-card emphasis">
      <span>Anchor report date from Part5</span>
      <strong>${reportDate || '-'}</strong>
      <p>目前 Part6 以 Part5 selected report_date 作為 manager-action event 的 anchor date。active key: ${activeReportKey}</p>
    </div>
    <div class="part6-anchor-card">
      <span>Event-time rolling style window</span>
      <strong>${p.style_window_start_date && p.style_window_end_date ? `${p.style_window_start_date} ~ ${p.style_window_end_date}` : (reportDate ? part6StyleWindowText(reportDate, p.style_window_years || p.training_window_years || 3) : '-')}</strong>
      <p>strict trailing 36M、排除當日事件；baseline observations: ${Number.isFinite(Number(p.style_obs_count)) ? p.style_obs_count : '-'}。</p>
    </div>
    <div class="part6-anchor-card">
      <span>Selected manager-action event</span>
      <strong>${manager}</strong>
      <p>model action type: ${action}<br>selected reports from Part5: ${Number.isFinite(selectedReports) ? selectedReports : '-'}</p>
    </div>
    <div class="part6-anchor-card">
      <span>Prediction / style deviation</span>
      <strong>${Number.isFinite(prob) ? formatPct(prob) : '-'}</strong>
      <p>future 12M excess: ${Number.isFinite(future) ? formatPct(future) : '-'}<br>style / sector / action deviation: ${Number.isFinite(styleDev) ? formatNumber(styleDev, 3) : '-'} / ${Number.isFinite(sectorDev) ? formatNumber(sectorDev, 3) : '-'} / ${Number.isFinite(actionDev) ? formatNumber(actionDev, 3) : '-'}</p>
    </div>
  `;
}

function part6BuildStyleEventRows(predictions) {
  return (predictions || []).map(p => ({
    event_id: p.event_id,
    manager: p.manager,
    fund: p.fund,
    report_date: p.report_date,
    style_window: p.style_window_text || part6StyleWindowText(p.report_date, p.training_window_years || 3),
    action_type: p.action_type,
    market_regime: p.market_regime,
    prediction_probability: p.prediction_probability,
    future_12m_excess_return: p.future_12m_excess_return,
    label_positive_excess_12m: p.label_positive_excess_12m,
    rolling_style_deviation_score: p.rolling_style_deviation_score,
    rolling_sector_deviation_score: p.rolling_sector_deviation_score,
    rolling_cross_asset_deviation_score: p.rolling_cross_asset_deviation_score,
    rolling_action_deviation_score: p.rolling_action_deviation_score,
    style_obs_count: p.style_obs_count,
    delta_stock: p.delta_stock,
    delta_beta: p.delta_beta,
    delta_technology: p.delta_technology,
    delta_nonstock_total_exposure: p.delta_nonstock_total_exposure,
    delta_sector_exposure: p.delta_sector_exposure,
    style_deviation_score: p.style_deviation_score,
    outcome_text: part6OutcomeText(p)
  }));
}

function part6DrawStyleEventChart(predictions) {
  const node = part6GetNode('part6StyleEventChart');
  if (!node || !window.Plotly) return;
  const rows = part6BuildStyleEventRows(predictions)
    .filter(row => row.report_date)
    .sort((a, b) => String(a.report_date).localeCompare(String(b.report_date)) || String(a.manager).localeCompare(String(b.manager)));
  if (!rows.length) {
    Plotly.react(node, [], {
      title: 'Part6：Report-date anchor and rolling-style context（尚未有 backend events）',
      height: 360,
      annotations: [{ text: '請先在 Part5 選 report date，然後按 Run Backend Analysis。', x: 0.5, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false }]
    }, { displaylogo: false, responsive: true });
    return;
  }
  const x = rows.map(row => row.report_date);
  const yProb = rows.map(row => row.prediction_probability);
  const yStyle = rows.map(row => Number.isFinite(Number(row.rolling_style_deviation_score)) ? Number(row.rolling_style_deviation_score) : null);
  const custom = rows.map(row => [row.manager, row.action_type, row.style_window, row.future_12m_excess_return, row.outcome_text, row.market_regime]);
  Plotly.react(node, [
    {
      type: 'scatter', mode: 'lines+markers', name: 'ML positive probability',
      x, y: yProb, yaxis: 'y', customdata: custom,
      hovertemplate: 'report date：%{x}<br>manager：%{customdata[0]}<br>action：%{customdata[1]}<br>style window：%{customdata[2]}<br>ML positive prob：%{y:.1%}<br>future 12M excess：%{customdata[3]:.2%}<br>outcome：%{customdata[4]}<br>market：%{customdata[5]}<extra></extra>'
    },
    {
      type: 'scatter', mode: 'lines+markers', name: 'Rolling style deviation',
      x, y: yStyle, yaxis: 'y2', customdata: custom,
      hovertemplate: 'report date：%{x}<br>manager：%{customdata[0]}<br>style window：%{customdata[2]}<br>rolling style deviation：%{y:.3f}<extra></extra>'
    }
  ], {
    title: 'Part6：Report-date anchor timeline — ML probability vs event-time rolling style deviation',
    height: 430,
    margin: { l: 68, r: 72, t: 62, b: 72 },
    xaxis: { title: 'Anchor report date from Part5', type: 'category', tickangle: -35 },
    yaxis: { title: 'ML positive excess probability', tickformat: '.0%', range: [0, 1] },
    yaxis2: { title: 'Rolling style deviation', overlaying: 'y', side: 'right', zeroline: false },
    legend: { orientation: 'h', y: -0.25 },
    hovermode: 'closest'
  }, { displaylogo: false, responsive: true });
}

function part6RenderStyleEventTable(predictions) {
  const rows = part6BuildStyleEventRows(predictions).sort((a, b) => String(b.report_date).localeCompare(String(a.report_date)) || b.prediction_probability - a.prediction_probability);
  renderTable('part6StyleEventTable', rows.slice(0, 120), [
    { key: 'report_date', label: 'anchor report_date' },
    { key: 'style_window', label: 'event-time style window' },
    { key: 'manager', label: 'manager' },
    { key: 'fund', label: 'fund' },
    { key: 'action_type', label: 'model action_type' },
    { key: 'market_regime', label: 'market_regime' },
    { key: 'style_obs_count', label: '36M baseline obs', format: 'int' },
    { key: 'delta_stock', label: 'Δ stock', format: 'num' },
    { key: 'delta_beta', label: 'Δ beta', format: 'num' },
    { key: 'delta_technology', label: 'Δ technology', format: 'num' },
    { key: 'delta_sector_exposure', label: 'Δ sector exposure', format: 'num' },
    { key: 'delta_nonstock_total_exposure', label: 'Δ nonstock exposure', format: 'num' },
    { key: 'rolling_style_deviation_score', label: 'rolling style deviation', format: 'num' },
    { key: 'rolling_sector_deviation_score', label: 'rolling sector deviation', format: 'num' },
    { key: 'rolling_cross_asset_deviation_score', label: 'rolling cross-asset deviation', format: 'num' },
    { key: 'rolling_action_deviation_score', label: 'rolling action deviation', format: 'num' },
    { key: 'style_deviation_score', label: 'style deviation', format: 'num' },
    { key: 'prediction_probability', label: 'ML positive prob', format: 'pct' },
    { key: 'future_12m_excess_return', label: 'future 12M excess', format: 'pct' },
    { key: 'outcome_text', label: 'interpreted outcome' }
  ], { title: 'Part6 report-date events: anchor date + rolling-style window + prediction outcome', expanded: true, count: rows.length });
}

function part6StockActionRowsForDisplay(predictions) {
  const predMap = new Map();
  for (const p of predictions || []) {
    predMap.set(part6PredictionKey(p.crsp_portno, p.report_date), p);
  }
  let payloadRows = (((state.part6.lastPayload || {}).part5 || {}).stock_action_rows) || [];
  if ((!payloadRows || !payloadRows.length) && state.part5 && state.part5.loaded && predictions && predictions.length) {
    const keys = predictions.map(p => part6PredictionKey(p.crsp_portno, p.report_date));
    if (typeof buildPart5StockActionRowsForBackend === 'function') {
      payloadRows = buildPart5StockActionRowsForBackend(keys);
    }
  }
  return (payloadRows || []).map(row => {
    const pred = predMap.get(part6PredictionKey(row.crsp_portno, row.report_dt)) || {};
    const future = Number(pred.future_12m_excess_return);
    return {
      ...row,
      manager: pred.manager || '',
      model_action_type: pred.action_type || '',
      market_regime: pred.market_regime || '',
      prediction_probability: pred.prediction_probability,
      future_12m_excess_return: pred.future_12m_excess_return,
      label_positive_excess_12m: pred.label_positive_excess_12m,
      linked_event_id: pred.event_id || '',
      style_window: pred.style_window_text || part6StyleWindowText(row.report_dt, pred.training_window_years || 3),
      rolling_style_deviation_score: pred.rolling_style_deviation_score,
      style_deviation_score: pred.style_deviation_score,
      outcome_text: Number.isFinite(future) ? (future > 0 ? 'Good: future 12M positive excess' : 'Bad: future 12M negative excess') : 'Outcome unknown',
      signed_delta_abs: Math.abs(Number(row.delta_holding_pct) || 0),
      result_bucket: Number.isFinite(future) ? (future > 0 ? 'good' : 'bad') : 'unknown'
    };
  }).sort((a, b) => {
    const ap = Number.isFinite(a.prediction_probability) ? a.prediction_probability : -1;
    const bp = Number.isFinite(b.prediction_probability) ? b.prediction_probability : -1;
    if (bp !== ap) return bp - ap;
    return (b.signed_delta_abs || 0) - (a.signed_delta_abs || 0);
  });
}

function drawPart6StockActionChart(rows) {
  const node = part6GetNode('part6StockActionChart');
  if (!node || !window.Plotly) return;
  const data = (rows || []).filter(row => row.stock_action_direction).slice(0, 40).reverse();
  if (!data.length) {
    Plotly.react(node, [], {
      title: 'Part6：Actual stock increase/decrease from Part5 JSON（尚未取得可對齊的持股變化）',
      height: 400,
      annotations: [{ text: '請先在 Part5 選取基金報告，或讓 backend 回傳與 Part5 report date 可對齊的 ML events。', x: 0.5, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false }]
    }, { displaylogo: false, responsive: true });
    return;
  }
  Plotly.react(node, [{
    type: 'bar', orientation: 'h',
    x: data.map(row => row.delta_holding_pct),
    y: data.map(row => `${row.holding_ticker || row.holding_key} | ${row.report_dt}`),
    text: data.map(row => row.stock_action_direction === 'decrease' ? '減碼' : (row.stock_action_direction === 'new_position' ? '新增' : '加碼')),
    customdata: data.map(row => [
      row.fund_ticker || row.crsp_portno,
      row.holding_security_name,
      row.sector,
      row.stock_action_direction,
      row.model_action_type,
      row.prediction_probability,
      row.future_12m_excess_return,
      row.outcome_text,
      row.previous_holding_pct,
      row.current_holding_pct,
      row.market_regime,
      row.style_window,
      row.rolling_style_deviation_score
    ]),
    hovertemplate: '股票：%{y}<br>基金：%{customdata[0]}<br>名稱：%{customdata[1]}<br>Sector：%{customdata[2]}<br>實際持股動作：%{customdata[3]}<br>model action type：%{customdata[4]}<br>前期權重：%{customdata[8]:.2%}<br>本期權重：%{customdata[9]:.2%}<br>delta：%{x:.2%}<br>Anchor / style window：%{customdata[11]}<br>rolling style deviation：%{customdata[12]:.3f}<br>ML positive prob：%{customdata[5]:.1%}<br>future 12M excess：%{customdata[6]:.2%}<br>結果：%{customdata[7]}<br>market：%{customdata[10]}<extra></extra>'
  }], {
    title: 'Part6：Actual stock add/reduce actions linked to anchor report_date, ML prediction, style deviation, and future 12M result',
    height: Math.max(480, Math.min(950, 280 + data.length * 20)),
    margin: { l: 210, r: 44, t: 72, b: 60 },
    xaxis: { title: 'Holding weight change: negative = reduce, positive = add/new', tickformat: '.1%', zeroline: true },
    yaxis: { automargin: true },
    showlegend: false
  }, { displaylogo: false, responsive: true });
}

function renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows = []) {
  const eventRows = part6BuildStyleEventRows(predictions).sort((a, b) => b.prediction_probability - a.prediction_probability);
  renderTable('part6PredictionResultTable', eventRows.slice(0, 50), [
    { key: 'event_id', label: 'event_id' },
    { key: 'manager', label: 'manager' },
    { key: 'fund', label: 'fund' },
    { key: 'report_date', label: 'anchor report_date' },
    { key: 'style_window', label: 'event-time style window' },
    { key: 'action_type', label: 'model action_type' },
    { key: 'market_regime', label: 'market_regime' },
    { key: 'rolling_style_deviation_score', label: 'rolling style deviation', format: 'num' },
    { key: 'prediction_probability', label: 'ML positive excess probability', format: 'pct' },
    { key: 'label_positive_excess_12m', label: 'historical label' },
    { key: 'future_12m_excess_return', label: 'future 12M excess', format: 'pct' }
  ], { title: 'Part6 backend ML prediction events + event-time rolling style context', expanded: true, count: predictions.length });

  renderTable('part6StockActionTable', (stockActionRows || []).slice(0, 200), [
    { key: 'report_dt', label: 'anchor report_date' },
    { key: 'style_window', label: 'event-time style window' },
    { key: 'fund_ticker', label: 'fund' },
    { key: 'model_action_type', label: 'model action_type' },
    { key: 'holding_ticker', label: 'ticker' },
    { key: 'holding_security_name', label: 'security' },
    { key: 'sector', label: 'sector' },
    { key: 'stock_action_direction', label: 'actual stock action' },
    { key: 'previous_holding_pct', label: 'previous weight', format: 'pct' },
    { key: 'current_holding_pct', label: 'current weight', format: 'pct' },
    { key: 'delta_holding_pct', label: 'delta weight', format: 'pct' },
    { key: 'rolling_style_deviation_score', label: 'rolling style deviation', format: 'num' },
    { key: 'stock_beta', label: 'stock beta', format: 'num' },
    { key: 'prediction_probability', label: 'ML positive prob', format: 'pct' },
    { key: 'future_12m_excess_return', label: 'future 12M excess', format: 'pct' },
    { key: 'label_positive_excess_12m', label: 'good/bad label' },
    { key: 'outcome_text', label: 'interpreted outcome' }
  ], { title: 'Part6 linked table: Part5 actual stock actions + anchor report_date + rolling style context + model outcome', expanded: true, count: (stockActionRows || []).length });

  renderTable('part6ShapResultTable', shapRows.slice().sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 100), [
    { key: 'event_id', label: 'event_id' },
    { key: 'event_label', label: 'event' },
    { key: 'direction', label: 'direction' },
    { key: 'feature', label: 'feature' },
    { key: 'value', label: 'value', format: 'num' },
    { key: 'contribution', label: 'contribution', format: 'num' }
  ], { title: `Part6 SHAP local explanations | global features=${shapAgg.length}`, expanded: false, count: shapRows.length });
}

function renderPart6BackendVisuals(result) {
  const predictions = part6EnrichedPredictions(result);
  const shapRows = backendShapRows(result);
  const shapAgg = aggregateBackendShap(shapRows, 14);
  const avgProb = predictions.length ? mean(predictions.map(p => p.prediction_probability)) : NaN;
  const highCount = predictions.filter(p => p.prediction_probability >= 0.6).length;
  const topProb = predictions.length ? Math.max(...predictions.map(p => p.prediction_probability)) : NaN;
  if (dom.metricBackendPredictionCount) dom.metricBackendPredictionCount.textContent = predictions.length ? formatInt(predictions.length) : '-';
  if (dom.metricBackendAvgProb) dom.metricBackendAvgProb.textContent = Number.isFinite(avgProb) ? formatPct(avgProb) : '-';
  if (dom.metricBackendHighProb) dom.metricBackendHighProb.textContent = predictions.length ? formatInt(highCount) : '-';
  if (dom.metricBackendTopProb) dom.metricBackendTopProb.textContent = Number.isFinite(topProb) ? formatPct(topProb) : '-';
  const stockActionRows = part6StockActionRowsForDisplay(predictions);
  part6RenderAnchorCards(result, predictions);
  part6DrawStyleEventChart(predictions);
  part6RenderStyleEventTable(predictions);
  drawPart6PredictionRankChart(predictions);
  drawPart6ProbabilityHistogram(predictions);
  drawPart6StockActionChart(stockActionRows);
  drawPart6ShapFeatureChart(shapAgg);
  drawPart6SingleEventShapChart(shapRows);
  renderPart6BackendTables(predictions, shapRows, shapAgg, stockActionRows);
}
