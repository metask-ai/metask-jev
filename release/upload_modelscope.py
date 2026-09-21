"""Upload a released nimble model to ModelScope (魔搭), mirroring the HF release.

Requires: pip install modelscope; MSCODESCOPE_TOKEN or logged-in via
`modelscope login`. Reads the same registry entry and card as the HF uploader.

Usage: python release/upload_modelscope.py --name nimble-4b
"""
import argparse
import subprocess
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--work-dir", default=str(ROOT.parent / "nimble-lab-work"))
    args = ap.parse_args()

    registry = yaml.safe_load((ROOT / "models" / "registry.yaml").read_text())
    entry = next(m for m in registry if m["name"] == args.name)
    ms_id = entry["remote"].get("modelscope")
    assert ms_id, f"{args.name} has no modelscope id in registry"

    local_dir = Path(args.work_dir) / args.name / "merged"
    if not (local_dir / "model.safetensors").exists():
        src = entry["artifacts"].get("kunshan")
        assert src, "no local weights and no kunshan path in registry"
        subprocess.run(["rsync", "-avP", "--partial",
                        f"kunshan:{src}/", str(local_dir) + "/"], check=True)

    card = ROOT / "models" / args.name / "card.md"
    (local_dir / "README.md").write_text(card.read_text())

    # modelscope SDK upload
    from modelscope.hub.api import HubApi
    api = HubApi()
    # api.login(os.environ["MODELSCOPE_TOKEN"])  # or already logged in
    api.push_model(model_id=ms_id, model_dir=str(local_dir))
    print("MODELSCOPE UPLOAD DONE:", f"https://modelscope.cn/models/{ms_id}")


if __name__ == "__main__":
    main()