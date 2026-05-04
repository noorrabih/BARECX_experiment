"""
Submit both fold JSONL files to Gemini Batch API, poll until done, merge results.

Usage
-----
  # Step 1 — upload & submit (run once):
  python run_batch.py --submit

  # Step 2 — poll & download (run after jobs complete, or re-run to check):
  python run_batch.py --collect --fold-a-job batches/XXXX --fold-b-job batches/YYYY

  # Or do everything in one blocking call:
  python run_batch.py --submit --wait
"""
from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"
POLL_INTERVAL = 30  # seconds
COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def submit_fold(client, fold: str) -> str:
    """Upload JSONL and create a batch job for one fold. Returns job name."""
    jsonl_path = f"X4/batch_fold_{fold}.jsonl"
    print(f"[Fold {fold.upper()}] Uploading {jsonl_path} ...")
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name=f"barec-X4-fold-{fold}",
            mime_type="application/jsonl",
        ),
    )
    print(f"[Fold {fold.upper()}] Uploaded: {uploaded.name}")

    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": f"barec-X4-fold-{fold}"},
    )
    print(f"[Fold {fold.upper()}] Job created: {job.name}  state={job.state.name}")
    return job.name


def wait_for_job(client, job_name: str, fold: str):
    """Poll until the job reaches a terminal state. Returns the final job object."""
    job = client.batches.get(name=job_name)
    while job.state.name not in COMPLETED_STATES:
        print(f"[Fold {fold.upper()}] state={job.state.name} — waiting {POLL_INTERVAL}s ...")
        time.sleep(POLL_INTERVAL)
        job = client.batches.get(name=job_name)
    print(f"[Fold {fold.upper()}] Final state: {job.state.name}")
    return job


def download_results(client, job, fold: str) -> list[dict]:
    """Download and parse batch results for one fold."""
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Fold {fold.upper()} job did not succeed: {job.state.name}")

    result_file_name = job.dest.file_name
    raw = client.files.download(file=result_file_name).decode("utf-8")

    rows = []
    errors = 0
    for line in raw.strip().splitlines():
        if not line:
            continue
        entry = json.loads(line)
        sid = entry.get("key")
        if "error" in entry:
            print(f"  [Fold {fold.upper()}] Error for key {sid}: {entry['error'].get('message', '')}")
            errors += 1
            continue
        response = entry.get("response", {})
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            text = ""
        rows.append({"ID": sid, "Codes": text})

    print(f"[Fold {fold.upper()}] Parsed {len(rows)} results ({errors} errors).")
    return rows


def save_results(rows_a: list[dict], rows_b: list[dict]):
    """Merge both folds with iaa.csv metadata and save to CSV."""
    df_iaa = pd.read_csv("iaa.csv")[["ID", "Sentence", "Readability_Level_19", "split"]]
    df_iaa["ID"] = df_iaa["ID"].astype(str)

    df_results = pd.DataFrame(rows_a + rows_b)
    df_results["ID"] = df_results["ID"].astype(str)

    # Add fold column from fold_assignments
    df_folds = pd.read_csv("X4/fold_assignments.csv")[["ID", "fold"]]
    df_folds["ID"] = df_folds["ID"].astype(str)

    df_out = df_iaa.merge(df_results, on="ID", how="left")
    df_out = df_out.merge(df_folds, on="ID", how="left")

    out_path = "X4/gemini_results_X4.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="Upload and submit batch jobs")
    parser.add_argument("--wait", action="store_true", help="Block until both jobs complete (requires --submit)")
    parser.add_argument("--collect", action="store_true", help="Download results and save CSV")
    parser.add_argument("--fold-a-job", default=None, help="Fold A job name (for --collect)")
    parser.add_argument("--fold-b-job", default=None, help="Fold B job name (for --collect)")
    args = parser.parse_args()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    fold_a_job_name = args.fold_a_job
    fold_b_job_name = args.fold_b_job

    if args.submit:
        fold_a_job_name = submit_fold(client, "a")
        fold_b_job_name = submit_fold(client, "b")
        print(f"\nJob names to pass to --collect:")
        print(f"  --fold-a-job {fold_a_job_name}")
        print(f"  --fold-b-job {fold_b_job_name}")

        if args.wait:
            job_a = wait_for_job(client, fold_a_job_name, "a")
            job_b = wait_for_job(client, fold_b_job_name, "b")
            rows_a = download_results(client, job_a, "a")
            rows_b = download_results(client, job_b, "b")
            save_results(rows_a, rows_b)

    elif args.collect:
        if not fold_a_job_name or not fold_b_job_name:
            parser.error("--collect requires --fold-a-job and --fold-b-job")
        job_a = wait_for_job(client, fold_a_job_name, "a")
        job_b = wait_for_job(client, fold_b_job_name, "b")
        rows_a = download_results(client, job_a, "a")
        rows_b = download_results(client, job_b, "b")
        save_results(rows_a, rows_b)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
