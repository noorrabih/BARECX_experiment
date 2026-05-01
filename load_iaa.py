import pandas as pd
import sys
import os

# Login using e.g. `huggingface-cli login` to access this dataset
splits = {'train': 'data/train-00000-of-00001.parquet', 'dev': 'data/dev-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
base_url = "hf://datasets/CAMeL-Lab/BAREC-Corpus-v1.0/"


dfs = []
for split_name, fname in splits.items():
    df_split = pd.read_parquet(base_url + fname)
    df_split["split"] = split_name
    dfs.append(df_split)

df_all = pd.concat(dfs, ignore_index=True)
df_iaa = df_all[df_all["Annotator"] == "IAA"].reset_index(drop=True)

print(f"IAA rows — train/dev/test combined: {len(df_iaa)}")
df_iaa.to_csv("iaa.csv", index=False)