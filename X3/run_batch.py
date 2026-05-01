"""
Submit iaa_batch.jsonl to Gemini Batch API, poll for completion, save results to CSV.
"""
from __future__ import annotations

import os, time, json
import pandas as pd
from google import genai
from google.genai import types

MODEL = "gemini-3-flash-preview"

completed_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# # ── 1. Upload JSONL via Files API ────────────────────────────────────────────
# print("Uploading iaa_batch.jsonl ...")
# uploaded_file = client.files.upload(
#     file="batch.jsonl",
#     config=types.UploadFileConfig(display_name="barec-batch-requests-X3", mime_type="jsonl"),
# )
# print(f"Uploaded: {uploaded_file.name}")

# # ── 2. Create batch job ──────────────────────────────────────────────────────
# batch_job = client.batches.create(
#     model=MODEL,
#     src=uploaded_file.name,
#     config={"display_name": "barec-reasoning-job-X3"},
# )
# print(f"Created batch job: {batch_job.name}  state={batch_job.state.name}")

# # ── 3. Poll until done ───────────────────────────────────────────────────────
# while batch_job.state.name not in completed_states:
#     time.sleep(30)
#     batch_job = client.batches.get(name=batch_job.name)
#     print(f"  state={batch_job.state.name}", flush=True)


job_name = "batches/k4iu2624s4vzkrxlkce41265czjx6sy6jhwm"
batch_job = client.batches.get(name=job_name)

print(f"Finished: {batch_job.state.name}")
if batch_job.state.name != "JOB_STATE_SUCCEEDED":
    raise RuntimeError(f"Batch job did not succeed: {batch_job.state.name}")

# ── 4. Download & parse results ──────────────────────────────────────────────
result_file_name = batch_job.dest.file_name
raw = client.files.download(file=result_file_name).decode("utf-8")

rows = []
for line in raw.strip().splitlines():
    if not line:
        continue
    entry = json.loads(line)
    sid = entry.get("key")
    response = entry.get("response", {})
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        text = ""
    rows.append({"ID": sid, "Codes": text})

# ── 5. Save to CSV ───────────────────────────────────────────────────────────
df_results = pd.DataFrame(rows)
df_iaa = pd.read_csv("../iaa.csv")[["ID", "Sentence", "Readability_Level_19"]]
df_iaa["ID"] = df_iaa["ID"].astype(str)
df_out = df_iaa.merge(df_results, on="ID", how="left")
df_out.to_csv("gemini_results_X3.csv", index=False)
print(f"Saved {len(df_out)} rows to gemini_results_X3.csv")


# -c "
# import os
# from google import genai
# client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
# batch = client.batches.get(name='batches/6ivfq7qsy8xdhvt3qipgdyq9zzb3gqsipgy9')
# print(batch)
# "

# -c "
# from google import genai
# import os
# client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
# client.batches.cancel(name='batches/0u8y9h61157iwl2x4eqhhsus5ee4umha2109')
# "