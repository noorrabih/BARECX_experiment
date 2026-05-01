"""
Build a JSONL batch file for Gemini Batch API from iaa.csv.
Each line: {"key": "<ID>", "request": {"contents": [{"parts": [{"text": "..."}]}]}}
"""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from gemini import load_codes, build_prompt

df_iaa   = pd.read_csv("iaa.csv")
df_codes = load_codes("codes.xlsx")

out_path = "X3/batch.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for _, row in df_iaa.iterrows():
        sid      = row["ID"]
        sentence = row["Sentence"]
        level    = int(row["Readability_Level_19"])

        features = (
            df_codes[(df_codes["level"] == level) | (df_codes["level"] == 'all')][["code", "description"]]
            .to_dict(orient="records")
        )

        prompt = build_prompt(sentence, features)

        entry = {
            "key": str(sid),
            "request": {
                "contents": [{"parts": [{"text": prompt}]}]
            },
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Wrote {len(df_iaa)} requests to {out_path}")
