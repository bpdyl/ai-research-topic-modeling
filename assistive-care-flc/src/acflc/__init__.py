"""Fuzzy logic controller for an assistive-care flat (ST7085CEM Task 2)."""

from .controller import FuzzyController, InferenceTrace, Rule
from .membership import MF, Variable, trapmf, trimf
from .flat import build_controller, fam_tables, thermal_rules, lighting_rules

__all__ = [
    "FuzzyController", "InferenceTrace", "Rule", "MF", "Variable",
    "trimf", "trapmf", "build_controller", "fam_tables",
    "thermal_rules", "lighting_rules",
]
