"""Semantic contracts shared by the Part6/Part7 research backend.

The backend treats a prediction as an auditable quantitative claim, not an
investment recommendation.  These strings are deliberately centralized so a
manager archetype, an event-level deviation, and a model explanation cannot be
silently presented as the same kind of "style".
"""

ANALYSIS_UNIT = "manager × fund/portfolio × report_date disclosed-action event"

MANAGER_STYLE_TERM = "Manager Performance–Risk–Flow Archetype"
MANAGER_STYLE_DEFINITION = (
    "A descriptive historical grouping based on observed performance, risk, "
    "fees, fund flows, size, and related manager/fund history. It is a research "
    "prior, not a traditional value/growth/size investment style, personality, "
    "causal skill estimate, or forecast guarantee."
)

EVENT_DEVIATION_DEFINITION = (
    "A report-date event's distance from the same manager's strictly ex-ante "
    "trailing history; it describes an exception relative to the manager's own "
    "past and is not a permanent manager label."
)

SHAP_PATTERN_DEFINITION = (
    "A grouping or attribution of events by the evidence used by the fitted "
    "direction model. It describes model decision logic, not manager style, "
    "manager intent, an economic mechanism, or causality."
)

DISCLOSED_ACTION_DEFINITION = (
    "A change reconstructed from reported holdings/allocation snapshots. It is "
    "not an observed trade date, execution price, real-time position, or direct "
    "evidence of manager intent."
)

PART6_CLAIM_DEFINITION = (
    "A fallible multi-horizon quantitative model claim supplied to Part7 for "
    "reliability-aware evidence audit; it is not a literal win probability, "
    "tradable expected return, or investment recommendation."
)

__all__ = [
    "ANALYSIS_UNIT",
    "MANAGER_STYLE_TERM",
    "MANAGER_STYLE_DEFINITION",
    "EVENT_DEVIATION_DEFINITION",
    "SHAP_PATTERN_DEFINITION",
    "DISCLOSED_ACTION_DEFINITION",
    "PART6_CLAIM_DEFINITION",
]
