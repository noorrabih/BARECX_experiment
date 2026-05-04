import pandas as pd
import numpy as np
from pathlib import Path

single_annotator = False  # Set to True if we want to evaluate a single annotator against the system
annotator1 = 'all_annotators' # all_annotators
annotator2 = 'System'
criteria = 'RC' #C


# ========= Config =========
# INPUT_CSV1  = Path(f"/l/users/nour.rabih/Readability-morph/ready_annotations/{annotator1}/{annotator1}_2_4_6_{criteria}.csv")
# INPUT_CSV2  = Path(f"/l/users/nour.rabih/Readability-morph/ready_annotations/{annotator2}/{annotator2}_2_4_6_{criteria}.csv")
INPUT_CSV1 = Path(f"/home/nour.rabih/BARECX/all_annotators_reasoning_codes_{criteria}.csv") # annotator
# INPUT_CSV2 = Path(f"/home/nour.rabih/Readability-morph/new_predictions/IAA/iaa_label_reasoning_{criteria}.csv")
INPUT_CSV2 = Path(f"/home/nour.rabih/BARECX/X3/gemini_results_X3_{criteria}.csv") # system

OUTPUT_CSV  = Path(f"/home/nour.rabih/BARECX/X3/X3_eval_{criteria}.csv")

# Base category names
BASE_CATEGORIES = [
    "Orthography",
    "word count",
    "Morphology",
    "syntactic",
    "vocab",
    "content",
]

# Pairs of (annotator_column, system_column)
CATEGORIES = [
    (f"{annotator1}-{cat}", f"{annotator2}-{cat}") for cat in BASE_CATEGORIES
]

# ========= Helpers =========
def parse_codes(cell: object) -> set:
    """
    Convert a comma-separated string to a set of trimmed codes.
    NaN or empty -> empty set.
    """
    if pd.isna(cell):
        return set()
    s = str(cell).strip()
    if not s:
        return set()
    return {code.strip() for code in s.split(",") if code.strip()}

def pr_counts(true_set: set, pred_set: set):
    """
    Return (tp, fp, fn, precision, recall, f1).
    If both sets are empty => (0,0,0, nan,nan,nan).
    """
    if not true_set and not pred_set:
        return 0, 0, 0, np.nan, np.nan, np.nan
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return tp, fp, fn, precision, recall, f1

def jaccard(a: set, b: set):
    """Jaccard index, NaN if both empty."""
    if len(a) == 0 and len(b) == 0:
        return 1
    union = a | b
    return len(a & b) / len(union) if len(union) > 0 else 0

def minimal_match(a: set, b: set):
    """
    'At least 1' overlap:
    - 1 if there is any intersection
    - 0 if no overlap and at least one non-empty
    - NaN if both empty
    """
    if len(a) == 0 and len(b) == 0:
        return 1
    return 1 if len(a & b) > 0 else 0

def exact_match(a: set, b: set):
    """
    Exact set match:
    - 1 if sets equal and non-empty
    - 0 if not equal and at least one non-empty
    - NaN if both empty
    """
    if len(a) == 0 and len(b) == 0:
        return 1
    return 1 if (a == b and len(a) > 0) else 0

# ========= Main =========
def main():
    # Read and sort
    df_annotator = pd.read_csv(INPUT_CSV1).sort_values(by="ID")
    df_system    = pd.read_csv(INPUT_CSV2).sort_values(by="ID")
    if single_annotator:
        df_annotator = df_annotator[df_annotator['annotator'] == annotator1]
        df_iaa_annotator = pd.read_csv(f"/l/users/nour.rabih/Readability-morph/ready_annotations/{annotator1}/{annotator1}_2_4_6_{criteria}.csv").sort_values(by="ID")
        # concat df_iaa_annotator to df_annotator
        df_annotator = pd.concat([df_annotator, df_iaa_annotator], ignore_index=True)
        # those whose id are in df_annotator
        df_system = df_system[df_system['ID'].isin(df_annotator['ID'])]

    print(f"Annotator rows: {len(df_annotator)}, System rows: {len(df_system)}")
    
    # Rename Sawsan's columns: e.g., "Orthography" -> "Sawsan-Orthography"
    for cat in BASE_CATEGORIES:
        df_annotator.rename(columns={cat: f"{annotator1}-{cat}"}, inplace=True)

    # Rename Samar's columns: e.g., "Orthography" -> "Samar-Orthography"
    for cat in BASE_CATEGORIES:
        df_system.rename(columns={cat: f"{annotator2}-{cat}"}, inplace=True)

    # Merge dataframes on ID column
    df = pd.merge(df_annotator, df_system, on="ID", how="inner")

    # Sanity check
    missing = [c for pair in CATEGORIES for c in pair if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # --- Per-category, per-row metrics + counts + jaccard/minimal/exact ---
    for ann_col, sys_col in CATEGORIES:
        cat = ann_col.replace(f"{annotator1}-", "")  # e.g., "Orthography"

        p_col   = f"precision_{cat}"
        r_col   = f"recall_{cat}"
        f1_col  = f"f1_{cat}"
        tp_col  = f"tp_{cat}"
        fp_col  = f"fp_{cat}"
        fn_col  = f"fn_{cat}"
        j_col   = f"jaccard_{cat}"
        mm_col  = f"minimal_match_{cat}"
        ex_col  = f"exact_match_{cat}"

        precisions, recalls, f1s = [], [], []
        tps, fps, fns = [], [], []
        jaccs, mms, exs = [], [], []

        for _, row in df.iterrows():
            true_set = parse_codes(row[ann_col])   # Sawsan
            pred_set = parse_codes(row[sys_col])   # Samar
            tp, fp, fn, p, r, f1 = pr_counts(true_set, pred_set)

            precisions.append(p); recalls.append(r); f1s.append(f1)
            tps.append(tp); fps.append(fp); fns.append(fn)

            jaccs.append(jaccard(true_set, pred_set))
            mms.append(minimal_match(true_set, pred_set))
            exs.append(exact_match(true_set, pred_set))

        df[p_col]  = precisions
        df[r_col]  = recalls
        df[f1_col] = f1s
        df[tp_col] = tps
        df[fp_col] = fps
        df[fn_col] = fns
        df[j_col]  = jaccs
        df[mm_col] = mms
        df[ex_col] = exs

    # --- Combined (all categories treated as one) ---
    annotator_cols = [a for a, _ in CATEGORIES]
    system_cols    = [s for _, s in CATEGORIES]

    prec_all, rec_all, f1_all = [], [], []
    tp_all_rows, fp_all_rows, fn_all_rows = [], [], []
    jacc_all, mm_all, ex_all = [], [], []

    # Micro counts
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for _, row in df.iterrows():
        true_all = set().union(*[parse_codes(row[c]) for c in annotator_cols])
        pred_all = set().union(*[parse_codes(row[c]) for c in system_cols])

        tp, fp, fn, p, r, f1 = pr_counts(true_all, pred_all)

        # Per-row combined metrics
        prec_all.append(p); rec_all.append(r); f1_all.append(f1)
        tp_all_rows.append(tp); fp_all_rows.append(fp); fn_all_rows.append(fn)

        # Jaccard / minimal / exact for all combined
        jacc_all.append(jaccard(true_all, pred_all))
        mm_all.append(minimal_match(true_all, pred_all))
        ex_all.append(exact_match(true_all, pred_all))

        # Micro accumulation
        total_tp += tp
        total_fp += fp
        total_fn += fn

    df["precision_all_combined"] = prec_all
    df["recall_all_combined"]    = rec_all
    df["f1_all_combined"]        = f1_all
    df["tp_all_combined"]        = tp_all_rows
    df["fp_all_combined"]        = fp_all_rows
    df["fn_all_combined"]        = fn_all_rows
    df["jaccard_all_combined"]   = jacc_all
    df["minimal_match_all_combined"] = mm_all
    df["exact_match_all_combined"]   = ex_all

    # ========== SUMMARY ROWS ==========

    # ---- Per-category MACRO (mean of per-row metrics) ----
    category_macro = {}
    for ann_col, _ in CATEGORIES:
        cat = ann_col.replace(f"{annotator1}-", "")
        category_macro[cat] = {
            "precision": float(np.nanmean(df[f"precision_{cat}"])),
            "recall":    float(np.nanmean(df[f"recall_{cat}"])),
            "f1":        float(np.nanmean(df[f"f1_{cat}"])),
            "jaccard":   float(np.nanmean(df[f"jaccard_{cat}"])),
            "minimal":   float(np.nanmean(df[f"minimal_match_{cat}"])),
            "exact":     float(np.nanmean(df[f"exact_match_{cat}"])),
        }

    # ---- Per-category MICRO (sum TP/FP/FN over rows) ----
    category_micro = {}
    for ann_col, _ in CATEGORIES:
        cat = ann_col.replace(f"{annotator1}-", "")
        tp_sum = float(np.nansum(df[f"tp_{cat}"]))
        fp_sum = float(np.nansum(df[f"fp_{cat}"]))
        fn_sum = float(np.nansum(df[f"fn_{cat}"]))

        prec_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else np.nan
        rec_micro  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else np.nan
        denom_f1   = (2 * tp_sum + fp_sum + fn_sum)
        f1_micro   = 2 * tp_sum / denom_f1 if denom_f1 > 0 else np.nan

        # micro Jaccard for this category: tp / (tp + fp + fn)
        denom_j   = (tp_sum + fp_sum + fn_sum)
        jacc_micro = tp_sum / denom_j if denom_j > 0 else np.nan

        category_micro[cat] = {
            "precision": prec_micro,
            "recall":    rec_micro,
            "f1":        f1_micro,
            "tp":        tp_sum,
            "fp":        fp_sum,
            "fn":        fn_sum,
            "jaccard":   jacc_micro,
        }


    # MICRO overall (global counts-based precision/recall/F1)
    print((total_tp + total_fp))
    overall_precision_micro = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else np.nan
    overall_recall_micro    = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else np.nan
    denom = (2 * total_tp + total_fp + total_fn)
    overall_f1_micro        = float(2 * total_tp / denom) if denom > 0 else np.nan
    denom_jaccard = (total_tp + total_fp + total_fn)
    overall_jaccard_micro = float(total_tp / denom_jaccard) if denom_jaccard > 0 else np.nan

    # ---- MACRO for all_combined (mean of per-row combined metrics) ----
    macro_all_combined = {
        "precision": float(np.nanmean(df["precision_all_combined"])),
        "recall":    float(np.nanmean(df["recall_all_combined"])),
        "f1":        float(np.nanmean(df["f1_all_combined"])),
        "jaccard":   float(np.nanmean(df["jaccard_all_combined"])),
        "minimal":   float(np.nanmean(df["minimal_match_all_combined"])),
        "exact":     float(np.nanmean(df["exact_match_all_combined"])),
    }

    # Prepare output and summary rows
    df["jaccard_all_combined_micro"] = np.nan   # global micro Jaccard column
    df_out = df.copy()
    df_out["Summary"] = ""

    # Column sets (not strictly used for ordering anymore, but kept for clarity)
    metric_cols = []
    count_cols  = []
    for ann_col, _ in CATEGORIES:
        cat = ann_col.replace(f"{annotator1}-", "")
        metric_cols += [
            f"precision_{cat}", f"recall_{cat}", f"f1_{cat}",
            f"jaccard_{cat}", f"minimal_match_{cat}", f"exact_match_{cat}"
        ]
        count_cols  += [f"tp_{cat}", f"fp_{cat}", f"fn_{cat}"]
    metric_cols += [
        "precision_all_combined", "recall_all_combined", "f1_all_combined",
        "jaccard_all_combined", "minimal_match_all_combined", "exact_match_all_combined"
    ]
    count_cols  += ["tp_all_combined", "fp_all_combined", "fn_all_combined"]

    # Build summary rows
    summary_rows = []

    # 1) Macro per category
    for cat, vals in category_macro.items():
        row = {col: np.nan for col in df_out.columns}
        row["Summary"] = f"Macro_{cat}"
        row[f"precision_{cat}"]      = vals["precision"]
        row[f"recall_{cat}"]         = vals["recall"]
        row[f"f1_{cat}"]             = vals["f1"]
        row[f"jaccard_{cat}"]        = vals["jaccard"]
        row[f"minimal_match_{cat}"]  = vals["minimal"]
        row[f"exact_match_{cat}"]    = vals["exact"]
        summary_rows.append(row)

    # 2) Micro per category
    for cat, vals in category_micro.items():
        row = {col: np.nan for col in df_out.columns}
        row["Summary"] = f"MICRO_{cat}"
        row[f"precision_{cat}"] = vals["precision"]
        row[f"recall_{cat}"]    = vals["recall"]
        row[f"f1_{cat}"]        = vals["f1"]
        row[f"tp_{cat}"]        = vals["tp"]
        row[f"fp_{cat}"]        = vals["fp"]
        row[f"fn_{cat}"]        = vals["fn"]
        row[f"jaccard_{cat}"]   = vals["jaccard"]
        summary_rows.append(row)


    # 3) MACRO_ALL_COMBINED
    macro_all_row = {col: np.nan for col in df_out.columns}
    macro_all_row["Summary"] = "MACRO_ALL_COMBINED"
    macro_all_row["precision_all_combined"]    = macro_all_combined["precision"]
    macro_all_row["recall_all_combined"]       = macro_all_combined["recall"]
    macro_all_row["f1_all_combined"]           = macro_all_combined["f1"]
    macro_all_row["jaccard_all_combined"]      = macro_all_combined["jaccard"]
    macro_all_row["minimal_match_all_combined"]= macro_all_combined["minimal"]
    macro_all_row["exact_match_all_combined"]  = macro_all_combined["exact"]
    summary_rows.append(macro_all_row)

    # 4) MICRO_ALL_COMBINED (global counts & metrics; no jaccard because that's not micro)
    micro_row = {col: np.nan for col in df_out.columns}
    micro_row["Summary"] = "MICRO_ALL_COMBINED"
    micro_row["precision_all_combined"] = overall_precision_micro
    micro_row["recall_all_combined"]    = overall_recall_micro
    micro_row["f1_all_combined"]        = overall_f1_micro
    micro_row["tp_all_combined"]        = total_tp
    micro_row["fp_all_combined"]        = total_fp
    micro_row["fn_all_combined"]        = total_fn
    micro_row["jaccard_all_combined_micro"] = overall_jaccard_micro
    summary_rows.append(micro_row)

    # Append summary rows
    df_out = pd.concat([df_out, pd.DataFrame(summary_rows)], ignore_index=True)

    # ===== FINAL COLUMN ORDER =====
    FINAL_ORDER = [
        "ID", "Sentence",

        # Orthography
        f"{annotator1}-Orthography", f"{annotator2}-Orthography",
        "precision_Orthography", "recall_Orthography", "f1_Orthography",
        "jaccard_Orthography", "minimal_match_Orthography", "exact_match_Orthography",
        "tp_Orthography", "fp_Orthography", "fn_Orthography",

        # word count
        f"{annotator1}-word count", f"{annotator2}-word count",
        "precision_word count", "recall_word count", "f1_word count",
        "jaccard_word count", "minimal_match_word count", "exact_match_word count",
        "tp_word count", "fp_word count", "fn_word count",

        # Morphology
        f"{annotator1}-Morphology", f"{annotator2}-Morphology",
        "precision_Morphology", "recall_Morphology", "f1_Morphology",
        "jaccard_Morphology", "minimal_match_Morphology", "exact_match_Morphology",
        "tp_Morphology", "fp_Morphology", "fn_Morphology",

        # syntactic
        f"{annotator1}-syntactic", f"{annotator2}-syntactic",
        "precision_syntactic", "recall_syntactic", "f1_syntactic",
        "jaccard_syntactic", "minimal_match_syntactic", "exact_match_syntactic",
        "tp_syntactic", "fp_syntactic", "fn_syntactic",

        # vocab
        f"{annotator1}-vocab", f"{annotator2}-vocab",
        "precision_vocab", "recall_vocab", "f1_vocab",
        "jaccard_vocab", "minimal_match_vocab", "exact_match_vocab",
        "tp_vocab", "fp_vocab", "fn_vocab",

        # content
        f"{annotator1}-content", f"{annotator2}-content",
        "precision_content", "recall_content", "f1_content",
        "jaccard_content", "minimal_match_content", "exact_match_content",
        "tp_content", "fp_content", "fn_content",

        # combined metrics
        "precision_all_combined", "recall_all_combined", "f1_all_combined",
        "jaccard_all_combined", "minimal_match_all_combined", "exact_match_all_combined",
        "tp_all_combined", "fp_all_combined", "fn_all_combined",
        "jaccard_all_combined_micro",     # <-- new micro Jaccard column
        "Summary",
    ]

    existing_cols = [c for c in FINAL_ORDER if c in df_out.columns]
    missing_cols  = [c for c in FINAL_ORDER if c not in df_out.columns]
    if missing_cols:
        print("⚠ WARNING: These columns were not found in df_out and will be skipped:", missing_cols)

    df_out = df_out[existing_cols]

    # # Save
    # df_out.to_csv(OUTPUT_CSV, index=False)
    from pathlib import Path

    # Save CSV as before
    df_out.to_csv(OUTPUT_CSV, index=False)

    # ===== Save a colored Excel file =====
    OUTPUT_XLSX = OUTPUT_CSV.with_suffix(".xlsx")

    # approximate colors from the 3rd row of the palette (tweak as you like)
    category_colors = {
        "Orthography": "#f4cccc",   # light red
        "word count":  "#fff2cc",   # light yellow
        "Morphology":  "#d9ead3",   # light green
        "syntactic":   "#cfe2f3",   # light blue
        "vocab":       "#d9d2e9",   # light purple
        "content":     "#ead1dc",   # light pink
        "all_combined": "#eeeeee",  # light grey for combined metrics (optional)
    }

    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        sheet_name = "Evaluation"
        df_out.to_excel(writer, index=False, sheet_name=sheet_name)

        workbook  = writer.book
        worksheet = writer.sheets[sheet_name]

        # Create a format for each category
        cat_formats = {
            cat: workbook.add_format({"bg_color": color})
            for cat, color in category_colors.items()
        }

        # Color columns by category
        for col_idx, col_name in enumerate(df_out.columns):
            fmt = None

            # Match against the base categories
            for cat in BASE_CATEGORIES:
                if (f"{cat}" in col_name) or (f"_{cat}" in col_name) or (f"-{cat}" in col_name):
                    fmt = cat_formats.get(cat)
                    break

            # Optionally color all_combined metrics separately
            if fmt is None and "all_combined" in col_name:
                fmt = cat_formats.get("all_combined")

            # Apply format to entire column if we found a category
            if fmt is not None:
                # None for width => keep default
                worksheet.set_column(col_idx, col_idx, None, fmt)

        # Optionally, make header bold
        header_fmt = workbook.add_format({"bold": True})
        worksheet.set_row(0, None, header_fmt)

        

    # Optional console preview
    print("=== MICRO (GLOBAL) SYSTEM PERFORMANCE, ALL CATEGORIES COMBINED ===")
    print("TP:", total_tp, "FP:", total_fp, "FN:", total_fn)
    print("Overall precision (micro):", round(overall_precision_micro, 3))
    print("Overall recall (micro)   :", round(overall_recall_micro, 3))
    print("Overall F1 (micro)       :", round(overall_f1_micro, 3))
    print("Overall Jaccard (micro)   :", round(overall_jaccard_micro, 3))
    print(f"\nSaved with TP/FP/FN + Jaccard/minimal/exact and micro/macro summaries: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
