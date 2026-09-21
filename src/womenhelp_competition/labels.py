from __future__ import annotations

SUBTASK1_LABELS = ["Mild", "Medium", "High", "Severe"]
SUBTASK1_ID_TO_NAME = {
    "0": "Mild",
    "1": "Medium",
    "2": "High",
    "3": "Severe",
}
SUBTASK1_NAME_TO_ID = {value: key for key, value in SUBTASK1_ID_TO_NAME.items()}

SUBTASK2_LABELS = [
    "Economic",
    "Physical",
    "Property-related",
    "Psychological",
    "Sexual",
    "Vicarious",
    "N/A",
]
SUBTASK2_COLUMNS = [f"L{index}" for index in range(len(SUBTASK2_LABELS))]
SUBTASK2_COLUMN_TO_NAME = dict(zip(SUBTASK2_COLUMNS, SUBTASK2_LABELS))
SUBTASK2_NAME_TO_COLUMN = {value: key for key, value in SUBTASK2_COLUMN_TO_NAME.items()}

SEVERITY_ALIASES = {
    "mild": "Mild",
    "leve": "Mild",
    "low": "Mild",
    "0": "Mild",
    "medium": "Medium",
    "media": "Medium",
    "medio": "Medium",
    "1": "Medium",
    "high": "High",
    "alta": "High",
    "alto": "High",
    "2": "High",
    "severe": "Severe",
    "severa": "Severe",
    "severo": "Severe",
    "grave": "Severe",
    "3": "Severe",
}

TYPE_ALIASES = {
    "economic": "Economic",
    "economica": "Economic",
    "económica": "Economic",
    "physical": "Physical",
    "fisica": "Physical",
    "física": "Physical",
    "property-related": "Property-related",
    "property related": "Property-related",
    "property": "Property-related",
    "patrimonial": "Property-related",
    "psychological": "Psychological",
    "psicologica": "Psychological",
    "psicológica": "Psychological",
    "sexual": "Sexual",
    "vicarious": "Vicarious",
    "vicaria": "Vicarious",
    "n/a": "N/A",
    "na": "N/A",
    "none": "N/A",
    "ninguna": "N/A",
    "no aplica": "N/A",
    "sin violencia": "N/A",
}
