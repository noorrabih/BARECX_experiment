"""
Analyze an Arabic sentence and return BAREC codes present in it,
filtered by the given level, using the Gemini API.
"""
from __future__ import annotations

import pandas as pd
import json
import re
from google import genai


# ── Load the codes table ────────────────────────────────────────────────────
def load_codes(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = ["description", "code", "level"]
    df = df.dropna(subset=["level"])          # drop rows without a level
    df["level"] = df["level"].apply(lambda x: x if str(x).strip().lower() == "all" else int(x))
    return df


# ── Build the Gemini prompt ─────────────────────────────────────────────────
# def build_prompt(sentence: str, features: list[dict]) -> str:
#     features_block = "\n".join(
#         f'- Code: {f["code"]} | Feature: {f["description"]}'
#         for f in features
#     )
#     return f"""You are an expert Arabic linguist.
#
# Below is a list of Arabic linguistic features (each with a code):
# {features_block}
#
# Sentence to analyse:
# "{sentence}"
#
# Task:
# Identify which of the above features are present in the sentence.
# Return ONLY a valid JSON array of the matching codes, with no explanation.
# Example output: ["M3-1", "S4-2", "V2-4"]
# If none match, return an empty array: []
# """

def build_prompt(sentence: str, features: list[dict]) -> str:
    features_block = "\n".join(
        f'- Code: {f["code"]} | Feature: {f["description"]}'
        for f in features
    )

    has_syllable_features = any(f["code"].startswith("O") for f in features)
    has_word_count_features = any(f["code"] == "WC" for f in features)

    counting_instructions = ""
    if has_syllable_features:
        counting_instructions += """
- For syllable (O) features: carefully count the number of syllables in each word, ignoring final vowel diacritics (حركات الإعراب). A syllable is a consonant followed by a vowel (short or long). Only mark the O code if the sentence contains words matching the described syllable count."""
    if has_word_count_features:
        counting_instructions += """
- For word count (WC) features: carefully count the number of unique non-repeating typographic words in the sentence, strictly excluding all punctuation marks. Only mark WC if the unique word count exactly matches the number described in the feature."""

    counting_section = (
        f"\nImportant counting instructions:{counting_instructions}\n"
        if counting_instructions else ""
    )

    return f"""You are an expert Arabic linguist.

Below is a list of Arabic linguistic features (each with a code):
{features_block}
{counting_section}
Sentence to analyse:
"{sentence}"

Task:
Identify which of the above features are present in the sentence.
Return ONLY a valid JSON array of the matching codes, with no explanation.
Example output: ["M3-1", "S4-2", "V2-4"]
If none match, return an empty array: []
"""


# ── Main function ───────────────────────────────────────────────────────────
def analyze_sentence(
    sentence: str,
    level: int,
    xlsx_path: str = "codes.xlsx",
    api_key: str | None = None,
) -> list[str]:
    """
    Returns a list of codes whose features are found in `sentence`
    among the features whose BAREC Level equals `level`.

    Parameters
    ----------
    sentence  : Arabic sentence to analyse
    level     : BAREC level to filter by (integer)
    xlsx_path : path to the codes Excel file
    api_key   : Gemini API key (or set GEMINI_API_KEY env variable)
    """
    df = load_codes(xlsx_path)
    filtered = df[(df["level"] == level) | (df["level"] == "all")] # change here to remove c,v,M,s-O

    if filtered.empty:
        print(f"No features found for level {level}.")
        return []

    features = filtered[["code", "description"]].to_dict(orient="records")

    # ── Call Gemini ─────────────────────────────────────────────────────────
    client = genai.Client(api_key=api_key)   # pass key or relies on env var

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=build_prompt(sentence, features),
    )

    raw = response.text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        codes = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Could not parse Gemini response:\n{raw}")
        return []

    # Validate that returned codes actually exist in the sheet
    valid_codes = set(filtered["code"].tolist())
    codes = [c for c in codes if c in valid_codes]

    return codes


# ── CLI demo ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import os
#
#     SENTENCE = "في الأعوام السابقة، كنا نكتب عن هواية واحدة، هي هواية جمع طوابع البريد.."
#     LEVEL = 10
#
#     api_key = os.environ.get("GEMINI_API_KEY")   # or hard-code for testing
#
#     result = analyze_sentence(
#         sentence=SENTENCE,
#         level=LEVEL,
#         xlsx_path="codes.xlsx",
#         api_key=api_key,
#     )
#
#     print(f"Sentence : {SENTENCE}")
#     print(f"Level    : {LEVEL}")
#     print(f"Codes    : {result}")


# ── Run on IAA CSV ──────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import os

#     api_key = os.environ.get("GEMINI_API_KEY")

#     df_iaa = pd.read_csv("iaa.csv")

#     out_path = "iaa_results.csv"
#     total = len(df_iaa)
#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         writer = __import__("csv").DictWriter(f, fieldnames=["ID", "Sentence", "Readability_Level_19", "Codes"])
#         writer.writeheader()

#         for i, row in df_iaa.iterrows():
#             sentence = row["Sentence"]
#             level    = int(row["Readability_Level_19"])
#             sid      = row["ID"]

#             print(f"[{i+1}/{total}] ID={sid}  level={level}", flush=True)

#             codes = analyze_sentence(
#                 sentence=sentence,
#                 level=level,
#                 xlsx_path="codes.xlsx",
#                 api_key=api_key,
#             )

#             writer.writerow({"ID": sid, "Sentence": sentence, "Readability_Level_19": level, "Codes": codes})
#             f.flush()

#     print(f"\nSaved {total} rows to {out_path}")