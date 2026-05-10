"""
Build JSONL batch files for few-shot Gemini prompting (cross-validation style).

Strategy
--------
- All 1793 IAA rows are split 50/50 stratified by Readability_Level_19 →
  Fold A and Fold B.
- Shots for Fold A come from Fold B's *training* rows (one per level).
- Shots for Fold B come from Fold A's *training* rows (one per level).
- Ground truth for shots: IAA_merge annotator preferred; falls back to
  any available annotator.

Output
------
  X4/batch_fold_a.jsonl   -- requests for Fold A rows
  X4/batch_fold_b.jsonl   -- requests for Fold B rows
  X4/fold_assignments.csv -- which fold each row ended up in
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from gemini import load_codes

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
CATS = ["Orthography", "word count", "Morphology", "syntactic", "vocab", "content"]


# ── Stratified 50/50 split ─────────────────────────────────────────────────────
def stratified_half_split(df: pd.DataFrame, level_col: str, seed: int):
    """
    Split df into two halves stratified by level_col.
    Returns (fold_a_indices, fold_b_indices) — lists of integer positions
    into df.reset_index(drop=True).
    """
    rng = np.random.default_rng(seed)
    fold_a, fold_b = [], []
    for _level, group in df.groupby(level_col):
        idx = group.index.tolist()
        rng.shuffle(idx)
        mid = max(1, len(idx) // 2)   # at least 1 in fold_a even for tiny classes
        fold_a.extend(idx[:mid])
        fold_b.extend(idx[mid:])
    return fold_a, fold_b


# ── Ground-truth codes for a sentence ──────────────────────────────────────────
def get_codes_for_id(sentence_id, df_ann: pd.DataFrame) -> list[str]:
    """
    Return the list of codes for sentence_id.
    Prefers IAA_merge; falls back to the first available annotator.
    """
    rows = df_ann[df_ann["ID"].astype(str) == str(sentence_id)]
    if rows.empty:
        return []
    iaa_rows = rows[rows["annotator"] == "IAA_merge"]
    row = iaa_rows.iloc[0] if not iaa_rows.empty else rows.iloc[0]
    codes: list[str] = []
    for cat in CATS:
        val = row[cat]
        if pd.notna(val) and str(val).strip():
            for code in str(val).split(","):
                code = code.strip()
                if code:
                    codes.append(code)
    return codes


# ── Select up to n shots per level from a pool ─────────────────────────────────
def select_shots(
    pool_df: pd.DataFrame,
    df_ann: pd.DataFrame,
    df_codes: pd.DataFrame,
    n_shots: int = 1,
) -> dict[int, list[dict]]:
    """
    From pool_df (training rows), pick up to n_shots sentences per
    Readability_Level_19, preferring sentences that have annotated codes.
    Returns {level: [shot, ...]}.
    """
    train_pool = pool_df[pool_df["split"] == "train"]
    shots_by_level: dict[int, list[dict]] = {}
    for level, group in train_pool.groupby("Readability_Level_19"):
        level = int(level)
        features = (
            df_codes[(df_codes["level"] == level) | (df_codes["level"] == "all")][
                ["code", "description"]
            ].to_dict(orient="records")
        )
        chosen: list[dict] = []
        for _, row in group.iterrows():
            if len(chosen) >= n_shots:
                break
            codes = get_codes_for_id(row["ID"], df_ann)
            if codes:
                chosen.append({"sentence": row["Sentence"], "level": level, "codes": codes, "features": features})
        if len(chosen) < n_shots:
            print(f"  WARNING: level {level} has only {len(chosen)} annotated shot(s) (requested {n_shots})")
        shots_by_level[level] = chosen
    return shots_by_level


# ── Build few-shot block ────────────────────────────────────────────────────────
def build_few_shot_block(shots: list[dict]) -> str:
    separator = "\n" + "─" * 60 + "\n"
    blocks = ["Here are some annotated examples to guide you:"]
    for i, shot in enumerate(shots, 1):
        codes_str = json.dumps(shot["codes"], ensure_ascii=False)
        blocks.append(
            f"[Example {i}]\n"
            f'Sentence: "{shot["sentence"]}"\n'
            f"Answer: {codes_str}"
        )
    return separator.join(blocks)


# ── Prompt builder (few-shot version) ──────────────────────────────────────────
def build_prompt_few_shot(sentence: str, features: list[dict], few_shot_block: str) -> str:
    features_block = "\n".join(
        f'- Code: {f["code"]} | Feature: {f["description"]}' for f in features
    )

    has_syllable_features = any(f["code"].startswith("O") for f in features)
    has_word_count_features = any(f["code"] == "WC" for f in features)

    counting_instructions = ""
    if has_syllable_features:
        counting_instructions += (
            "\n- For syllable (O) features: carefully count the number of syllables in each word, "
            "ignoring final vowel diacritics (حركات الإعراب). A syllable is a consonant followed by a vowel "
            "(short or long). Only mark the O code if the sentence contains words matching the described syllable count."
        )
    if has_word_count_features:
        counting_instructions += (
            "\n- For word count (WC) features: carefully count the number of unique non-repeating typographic words "
            "in the sentence, strictly excluding all punctuation marks. Only mark WC if the unique word count "
            "exactly matches the number described in the feature."
        )

    counting_section = (
        f"\nImportant counting instructions:{counting_instructions}\n"
        if counting_instructions
        else ""
    )

    sep = "─" * 60
    return (
        f"You are an expert Arabic linguist.\n\n"
        f"You will be given a sentence and a list of linguistic features (each with a code and description).\n"
        f"Identify which features are present in the sentence.\n"
        f"Return ONLY a valid JSON array of the matching codes, with no explanation.\n"
        f"If none match, return an empty array: []\n\n"
        f"Available features:\n{features_block}\n"
        f"{counting_section}\n"
        f"{few_shot_block}\n\n"
        f"{sep}\n"
        f"Now it's your turn.\n"
        f'Sentence to analyse:\n"{sentence}"\n'
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shots", type=int, default=1, choices=range(1, 5),
                        metavar="{1-4}", help="Number of same-level shots per prompt (1–4)")
    args = parser.parse_args()
    n_shots = args.n_shots

    df_iaa = pd.read_csv("iaa.csv")
    df_codes = load_codes("codes.xlsx")
    df_ann = pd.read_csv("all_annotators_reasoning_codes_RC.csv")

    # ── Split all rows into Fold A / Fold B ────────────────────────────────────
    fold_a_idx, fold_b_idx = stratified_half_split(
        df_iaa, "Readability_Level_19", seed=SEED
    )

    df_fold_a = df_iaa.loc[fold_a_idx].reset_index(drop=True)
    df_fold_b = df_iaa.loc[fold_b_idx].reset_index(drop=True)

    print(f"Fold A: {len(df_fold_a)} rows  |  Fold B: {len(df_fold_b)} rows")
    for fold_name, fold_df in [("A", df_fold_a), ("B", df_fold_b)]:
        tr = (fold_df["split"] == "train").sum()
        print(f"  Fold {fold_name} — train: {tr}, other: {len(fold_df) - tr}")

    # ── Select shots: each fold uses the *other* fold's training rows ──────────
    shots_for_a = select_shots(df_fold_b, df_ann, df_codes, n_shots)   # shots come from fold B train
    shots_for_b = select_shots(df_fold_a, df_ann, df_codes, n_shots)   # shots come from fold A train

    print(f"\nShots for Fold A: {sum(len(v) for v in shots_for_a.values())} examples across {len(shots_for_a)} levels (up to {n_shots} per level)")
    print(f"Shots for Fold B: {sum(len(v) for v in shots_for_b.values())} examples across {len(shots_for_b)} levels (up to {n_shots} per level)")

    # ── Write batch JSONL files ────────────────────────────────────────────────
    out_dir = f"X4/{n_shots}_shots"
    os.makedirs(out_dir, exist_ok=True)

    for fold_name, fold_df, shots_by_level in [
        ("a", df_fold_a, shots_for_a),
        ("b", df_fold_b, shots_for_b),
    ]:
        out_path = f"{out_dir}/batch_fold_{fold_name}.jsonl"
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for _, row in fold_df.iterrows():
                sid = row["ID"]
                sentence = row["Sentence"]
                level = int(row["Readability_Level_19"])

                features = (
                    df_codes[
                        (df_codes["level"] == level) | (df_codes["level"] == "all")
                    ][["code", "description"]]
                    .to_dict(orient="records")
                )

                level_shots = shots_by_level.get(level, [])
                few_shot_block = build_few_shot_block(level_shots)
                prompt = build_prompt_few_shot(sentence, features, few_shot_block)

                entry = {
                    "key": str(sid),
                    "request": {"contents": [{"parts": [{"text": prompt}]}]},
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

        print(f"Wrote {written} requests to {out_path}")

    # ── Save fold assignments for later result merging ─────────────────────────
    df_fold_a["fold"] = "A"
    df_fold_b["fold"] = "B"
    df_assignments = pd.concat([df_fold_a, df_fold_b], ignore_index=True)
    df_assignments[["ID", "Sentence", "Readability_Level_19", "split", "fold"]].to_csv(
        f"{out_dir}/fold_assignments.csv", index=False
    )
    print(f"Saved fold assignments to {out_dir}/fold_assignments.csv")

    # ── Save shots for both folds to CSV ──────────────────────────────────────
    shots_rows = []
    for fold_name, shots_by_level in [("A", shots_for_a), ("B", shots_for_b)]:
        for level_shots in shots_by_level.values():
            for shot in level_shots:
                shots_rows.append({
                    "used_as_shots_for_fold": fold_name,
                    "sourced_from_fold": "B" if fold_name == "A" else "A",
                    "level": shot["level"],
                    "sentence": shot["sentence"],
                    "codes": json.dumps(shot["codes"], ensure_ascii=False),
                })
    pd.DataFrame(shots_rows).to_csv(f"{out_dir}/shots.csv", index=False)
    print(f"Saved shots to {out_dir}/shots.csv")


if __name__ == "__main__":
    main()
