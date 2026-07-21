"""Stable Part 6 explanation exports.

Direction TreeSHAP/runtime-fallback contributions and the two Ridge magnitude
contribution families are produced by ``shap_service.explain_events``.
"""
from __future__ import annotations

from . import SHAP_PATTERN_DEFINITION
from .shap_service import explain_events

EXPLANATION_CONTRACT = {
    "shap_pattern_definition": SHAP_PATTERN_DEFINITION,
    "causal_interpretation_allowed": False,
    "manager_intent_inference_allowed": False,
    "expected_excess_exact_shap_available": False,
}

__all__ = ["EXPLANATION_CONTRACT", "explain_events"]
