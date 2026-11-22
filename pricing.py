"""Shared input encoding and local XGBoost inference."""
from pathlib import Path
from urllib.parse import quote, unquote
from zipfile import ZipFile
import json

import pandas as pd
import xgboost as xgb

EXTRA = ["Cpu_Model", "Gpu_Model"]


def encode(frame, details, families=False):
    if families:
        frame = frame.copy()
        # Pool component lines that share a name across several catalog SKUs.
        frame["Cpu_Line"] = frame.Cpu_Model.str.extract(
            r"(?i)\b(Core M|Celeron|Pentium|Atom|Xeon|Ryzen|Core i[3579]|A\d+|E\d+)", expand=False
        ).fillna("Other")
        frame["Cpu_Suffix"] = frame.Cpu_Model.str.extract(
            r"\b\d{4}([A-Z]{1,3})\b", expand=False
        ).fillna("Other")
        frame["Gpu_Line"] = frame.Gpu_Model.str.extract(
            r"(?i)(Quadro|FirePro|GeForce GTX|GeForce GT|GeForce MX|GeForce|Radeon Pro|Radeon R[2579]|Radeon|Iris|UHD Graphics|HD Graphics)",
            expand=False
        ).fillna("Other")
    if not details:
        frame = frame.drop(columns=EXTRA)
    # Some catalog names contain square brackets, which XGBoost reserves.
    return pd.get_dummies(frame, dtype=float).rename(columns=quote)


def load_bundle():
    with ZipFile(Path(__file__).with_name("laptop_model.zip")) as archive:
        model = xgb.Booster(params={"nthread": 2})
        model.load_model(bytearray(archive.read("model.ubj")))
        catalog = json.loads(archive.read("catalog.json"))
    return model, catalog


GROUPS = {
    "Processor": ("Cpu_Model_", "Cpu_Tier_", "Cpu_Speed", "Cpu_Line_", "Cpu_Suffix_"),
    "Graphics": ("Gpu_Brand_", "Gpu_Model_", "Gpu_Line_"),
    "Display": ("X_res", "Y_res", "PPI", "Inches", "IPS", "Touchscreen"),
    "Storage": ("SSD_GB", "HDD_GB"),
    "Brand": ("Company_",), "Form factor": ("TypeName_",),
    "Operating system": ("OpSys_",), "Memory": ("Ram",), "Weight": ("Weight",),
}


def explain(model, catalog, row):
    inputs = encode(pd.DataFrame([row]), catalog["detailed"], catalog.get("families", False))
    aligned = inputs.reindex(columns=model.feature_names, fill_value=0)
    matrix = xgb.DMatrix(aligned, nthread=1)
    price = float(model.predict(matrix)[0])
    values = model.predict(matrix, pred_contribs=True)[0]
    impacts = {label: 0.0 for label in GROUPS}
    records = []
    for column, value in zip(aligned.columns, values[:-1], strict=True):
        label = next(label for label, prefixes in GROUPS.items() if column.startswith(prefixes))
        impacts[label] += float(value)
        field, state = decode_column(column)
        records.append((label, field, state, float(aligned[column].iat[0]), float(value)))
    detail = pd.DataFrame(records, columns=["Component", "Feature", "Indicator", "Input", "SHAP (€)"])
    return price, float(values[-1]), impacts, aligned, detail


FEATURES = {
    "Company": "Brand", "TypeName": "Form factor", "OpSys": "Operating system",
    "Cpu_Model": "CPU model", "Cpu_Tier": "CPU tier", "Cpu_Speed": "Clock speed",
    "Cpu_Line": "CPU line", "Cpu_Suffix": "CPU suffix", "Gpu_Line": "GPU line",
    "Gpu_Model": "GPU model", "Gpu_Brand": "GPU brand", "Inches": "Screen size",
    "X_res": "Horizontal pixels", "Y_res": "Vertical pixels", "PPI": "Pixel density",
    "IPS": "IPS panel", "Touchscreen": "Touchscreen", "SSD_GB": "Solid-state storage",
    "HDD_GB": "Hard-drive storage", "Ram": "Memory", "Weight": "Weight",
}


def decode_column(column):
    name = unquote(column)
    for key, label in FEATURES.items():
        if name == key or name.startswith(key + "_"):
            return label, name[len(key):].lstrip("_")
    raise ValueError(f"Unknown model input {name}")


def estimate(model, catalog, row):
    return explain(model, catalog, row)[:4]
