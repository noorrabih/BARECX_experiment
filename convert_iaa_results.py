"""
Convert iaa_results.csv to the same format as test_reasoning_with_codes.csv.

Input:  iaa_results.csv  (ID, Sentence, Readability_Level_19, Codes)
Output: iaa_results_with_codes.csv  (same columns as test_reasoning_with_codes.csv)
"""

import ast
import re
import pandas as pd
from collections import defaultdict

EXCEL_PATH = "/home/nour.rabih/Readability-morph/interpretability.xlsx"
INPUT_CSV  = "/home/nour.rabih/BARECX/X3/gemini_results_X3.csv"
OUTPUT_CSV = "/home/nour.rabih/BARECX/X3/gemini_results_X3_RC.csv"

# ---------- helpers copied from new_interpret.py ----------

CATEGORY_NAMES = {
    'O': 'Orthography',
    'W': 'word count',
    'M': 'Morphology',
    'S': 'syntactic',
    'V': 'vocab',
    'C': 'content',
}

def _load_feature_code_map(excel_path):
    df = pd.read_excel(excel_path, header=0)
    m = {}
    for _, row in df.iterrows():
        m[row['Feature']] = row['Code']
    return {k: v for k, v in m.items() if pd.notna(v)}

def _load_feature_description_map(excel_path):
    df = pd.read_excel(excel_path, header=0)
    m = {}
    for _, row in df.iterrows():
        feat = row.get("Feature")
        desc = row.get("Arabic Description") or row.get("Arabic_Description") or row.get("Description_Arabic")
        if pd.notna(feat) and pd.notna(desc):
            m[feat] = desc
    return m

def _load_code_description_map(excel_path):
    """Direct code -> Arabic description map (handles rows where Feature is NaN)."""
    df = pd.read_excel(excel_path, header=0)
    m = {}
    for _, row in df.iterrows():
        code = row.get("Code")
        desc = row.get("Arabic Description") or row.get("Arabic_Description") or row.get("Description_Arabic")
        if pd.notna(code) and pd.notna(desc):
            m[str(code).strip()] = desc
    return m

def group_codes_by_category(codes, category_names=CATEGORY_NAMES):
    grouped = defaultdict(list)
    for raw in codes:
        code = str(raw).strip()
        if not code or code.lower() == 'code':
            continue
        if code == 'O':
            grouped['Orthography'].append(code)
            continue
        first = code[0]
        if first in category_names:
            grouped[category_names[first]].append(code)
    return dict(grouped)

def get_arabic_descriptions(reasons, desc_map):
    if isinstance(reasons, list):
        return [desc_map.get(r, "") for r in reasons]
    return []

# ---------- load mappings ----------

feature_code = _load_feature_code_map(EXCEL_PATH)
feature_desc = _load_feature_description_map(EXCEL_PATH)
code_desc    = _load_code_description_map(EXCEL_PATH)

# Build reverse map: code -> first matching feature
code_to_feature = {}
for feat, code in feature_code.items():
    if code not in code_to_feature:
        code_to_feature[code] = feat

def codes_str_to_list(s):
    if pd.isna(s):
        return []
    s = str(s).strip()
    # strip ```json ... ``` or ``` ... ``` wrappers
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

def codes_to_reasons(codes):
    """Return feature name for each code; fall back to the code itself when no feature exists."""
    return [code_to_feature.get(c, c) for c in codes]

def get_arabic_descriptions_with_fallback(reasons, codes, desc_map, code_desc_map):
    """
    Get Arabic description per reason.
    If a reason is a feature name, look it up in desc_map.
    If it's a bare code (no feature name), look it up in code_desc_map.
    """
    descs = []
    for reason, code in zip(reasons, codes):
        if reason in desc_map:
            descs.append(desc_map[reason])
        elif code in code_desc_map:
            descs.append(code_desc_map[code])
        else:
            descs.append("")
    return descs

# ---------- transform ----------

df = pd.read_csv(INPUT_CSV)

rows = []
for _, row in df.iterrows():
    codes_list = codes_str_to_list(row["Codes"])
    grouped_codes = group_codes_by_category(codes_list)
    reasons = codes_to_reasons(codes_list)
    arabic_descs = get_arabic_descriptions_with_fallback(reasons, codes_list, feature_desc, code_desc)

    rows.append({
        "ID": row["ID"],
        "Label": row["Readability_Level_19"],
        "Final_Prediction": row["Readability_Level_19"],
        "Reasons": reasons,
        "Codes": grouped_codes,
        "Arabic_Descriptions": arabic_descs,
    })

out = pd.DataFrame(rows)

CATS = ['Orthography', 'word count', 'Morphology', 'syntactic', 'vocab', 'content']
for cat in CATS:
    out[cat] = out['Codes'].apply(
        lambda d: ", ".join(d.get(cat, [])) if isinstance(d, dict) else ""
    )

out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(out)} rows to {OUTPUT_CSV}")
print(out.head(5).to_string(index=False))

# ---------- save C-codes version (presence/absence per category) ----------

OUTPUT_C_CSV = OUTPUT_CSV.replace("_RC.csv", "_C.csv")

C_MAPPING = {"Orthography": "O", "word count": "W", "Morphology": "M",
             "syntactic": "S", "vocab": "V", "content": "C"}

out_c = out.copy()
for col, letter in C_MAPPING.items():
    if col in out_c.columns:
        out_c[col] = out_c[col].apply(
            lambda x: letter if (isinstance(x, str) and x.strip() != "") or
                                (not isinstance(x, str) and pd.notna(x)) else ""
        )

out_c.to_csv(OUTPUT_C_CSV, index=False)
print(f"Saved C-codes version to {OUTPUT_C_CSV}")
