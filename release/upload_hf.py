"""Upload a released metask-jev model to HuggingFace (run from local Mac).

Reads models/registry.yaml for repo id and card; weights are pulled from the
machine recorded in `artifacts` (rsync) if not present locally.

Usage: python release/upload_hf.py --name metask-jev-4b
"""
import argparse
import subprocess
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="model name in registry.yaml")
    ap.add_argument("--work-dir", default=str(ROOT.parent / "metask-jev-lab-work"))
    args = ap.parse_args()

    registry = yaml.safe_load((ROOT / "models" / "registry.yaml").read_text())
    entry = next(m for m in registry if m["name"] == args.name)
    repo_id = entry["remote"]["hf"]
    assert repo_id, f"{args.name} has no hf id in registry"

    # 1. local weight copy (rsync with resume)
    local_dir = Path(args.work_dir) / args.name / "merged"
    if not (local_dir / "model.safetensors").exists():
        src = entry["artifacts"].get("kunshan")
        if not src:
            raise SystemExit("no local weights and no kunshan path in registry")
        print(f"[rsync] pulling weights from kunshan: {src}")
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-avP", "--partial",
                        f"kunshan:{src}/", str(local_dir) + "/"], check=True)

    # 2. card from models/<name>/card.md (single source of truth)
    card = ROOT / "models" / args.name / "card.md"
    (local_dir / "README.md").write_text(card.read_text())

    # 3. upload
    from huggingface_hub import HfApi
    api = HfApi()
    print("identity:", api.whoami()["name"])
    api.create_repo(repo_id, private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(local_dir),
                      commit_message=f"{args.name}: release")
    print("HF UPLOAD DONE:", f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()