"""Upload card.md + figures + temperature.json to HF repo (from local Mac).

Small files go direct from local — faster than proxying through kunshan.
Usage: python release/upload_hf_card.py
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
CARD_DIR = ROOT / "models" / "metask-jev-4b"
REPO = "wayfind/metask-jev-4b-policy-mix"


def main():
    api = HfApi()  # uses local HF_TOKEN / cached login
    print("identity:", api.whoami()["name"])

    files = [("card.md", "README.md"), ("temperature.json", "temperature.json")]
    figs = sorted((CARD_DIR / "eval" / "figs").glob("*.png"))
    for f in figs:
        files.append((f, f"eval/figs/{f.name}"))

    for src, dst in files:
        p = CARD_DIR / src if not str(src).startswith("/") else Path(src)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst,
                        repo_id=REPO, repo_type="model")
        print(f"  {dst}")
    print("CARD+FIGS DONE:", f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()