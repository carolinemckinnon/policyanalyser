"""Shared constants for GPT Agent categorisation and reasoning."""
from __future__ import annotations

CATEGORY_DEFINITION = "DEFINITION"
CATEGORY_RULE = "RULE"
CATEGORY_EXCEPTION = "EXCEPTION"
CATEGORY_OTHER = "OTHER"

CATEGORY_ORDER = [CATEGORY_DEFINITION, CATEGORY_EXCEPTION, CATEGORY_RULE, CATEGORY_OTHER]

MIN_DEFINITIONS = 2
MIN_RULES = 3
MIN_EXCEPTIONS = 2  # retrieved if available

SAFETY_STATEMENT = "These policies appear to define or apply this term differently. Seek supervisor guidance."

MAX_DEFINITION_ENTRIES_PER_TERM = 5
