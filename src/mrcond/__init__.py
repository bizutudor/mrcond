"""MRCond — an open benchmark and evidence-grounded pipeline for MR Conditional implant labeling.

Research prototype — not a medical device. Never use as the sole basis for an MRI
safety decision.
"""

from .schema import (
    DISCLAIMER,
    FIELD_NAMES,
    FIELD_SPECS,
    ConditionSet,
    Evidence,
    FieldStatus,
    FieldValue,
    LabelDocument,
    MRStatus,
)

__version__ = "0.1.0"

__all__ = [
    "DISCLAIMER",
    "FIELD_NAMES",
    "FIELD_SPECS",
    "ConditionSet",
    "Evidence",
    "FieldStatus",
    "FieldValue",
    "LabelDocument",
    "MRStatus",
    "__version__",
]
