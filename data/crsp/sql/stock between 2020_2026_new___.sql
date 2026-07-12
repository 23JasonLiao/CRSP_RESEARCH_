-- 2010-2014 Balanced Funds Top Holdings
-- 目的：
-- 1. 取得 2010-2014 年 Balanced mutual funds 的 holdings
-- 2. 每個 crsp_portno 在同一個 report_dt 下只保留一個主要 fundno
-- 3. 使用 report_dt 對齊 FUND_STYLE 的 begdt / enddt，確保該時間點基金仍為 Balanced
-- 4. 串接 FUND_SUMMARY 取得股票、債券、現金配置比例
-- 5. 串接 HOLDINGS_CO_INFO 取得 ticker、security name、permno

WITH valid_holdings AS (
    SELECT
        hold.crsp_portno,
        h.crsp_fundno,
        h.nasdaq AS fund_ticker,
        h.fund_name,
        sty.lipper_obj_name,

        hold.report_dt,
        hold.security_rank,
        hold.percent_tna AS holding_percent_tna,
        hold.market_val AS holding_market_val,
        hold.crsp_company_key,

        -- 同一個 portno / report_dt / security_rank / company key
        -- 如果對應到多個 fundno，只保留一個主要 fundno
        ROW_NUMBER() OVER (
            PARTITION BY
                hold.crsp_portno,
                hold.report_dt,
                hold.security_rank,
                hold.crsp_company_key
            ORDER BY h.crsp_fundno
        ) AS rn

    FROM HOLDINGS hold

    INNER JOIN FUND_HDR h
        ON hold.crsp_portno = h.crsp_portno

    INNER JOIN FUND_STYLE sty
        ON h.crsp_fundno = sty.crsp_fundno
       AND hold.report_dt >= sty.begdt
       AND (
            hold.report_dt <= sty.enddt
            OR sty.enddt IS NULL
       )

    WHERE EXTRACT(YEAR FROM hold.report_dt) BETWEEN 2020 AND 2026
      AND hold.security_rank <= 10
      AND (
            TRIM(sty.lipper_obj_cd) = 'B'
            OR TRIM(sty.lipper_obj_name) = 'Balanced'
      )
)

SELECT
    vh.crsp_portno,
    vh.fund_ticker,
    vh.fund_name,
    vh.lipper_obj_name,

    -- 基金整體配置比例，來自 FUND_SUMMARY
    fsum.per_com AS fund_percent_common_stock,
    fsum.per_bond AS fund_percent_bond,
    fsum.per_cash AS fund_percent_cash,

    -- 持股明細，來自 HOLDINGS
    vh.report_dt,
    vh.security_rank,
    vh.holding_percent_tna,
    vh.holding_market_val,

    -- 底層資產資訊，來自 HOLDINGS_CO_INFO
    det.crsp_company_key,
    det.security_name AS holding_security_name,
    det.ticker AS holding_ticker,
    det.permno AS holding_permno

FROM valid_holdings vh

LEFT JOIN HOLDINGS_CO_INFO det
    ON vh.crsp_company_key = det.crsp_company_key

LEFT JOIN FUND_SUMMARY fsum
    ON vh.crsp_fundno = fsum.crsp_fundno
   AND EXTRACT(YEAR FROM vh.report_dt) = EXTRACT(YEAR FROM fsum.caldt)
   AND EXTRACT(MONTH FROM vh.report_dt) = EXTRACT(MONTH FROM fsum.caldt)

WHERE vh.rn = 1

ORDER BY
    vh.crsp_portno,
    vh.report_dt,
    vh.security_rank;