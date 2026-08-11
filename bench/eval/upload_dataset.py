"""Builds a LangSmith dataset from data/filled/*.{png,jpg,pdf,...} paired
with matching data/ground_truth/<same-stem>.json files.

Usage:
    python -m bench.eval.upload_dataset [--dataset-name NAME]
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from bench.form.field_schema import FIELD_NAMES

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FILLED_DIR = REPO_ROOT / "data" / "filled"
GROUND_TRUTH_DIR = REPO_ROOT / "data" / "ground_truth"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".heic", ".webp"}


def build_examples() -> list[dict]:
    examples = []
    for image_path in sorted(FILLED_DIR.iterdir()):
        if image_path.suffix.lower() not in VALID_SUFFIXES:
            continue

        gt_path = GROUND_TRUTH_DIR / f"{image_path.stem}.json"
        if not gt_path.exists():
            print(f"skipping {image_path.name}: no ground truth at {gt_path.relative_to(REPO_ROOT)}")
            continue

        ground_truth = json.loads(gt_path.read_text())
        missing = [f for f in FIELD_NAMES if f not in ground_truth]
        if missing:
            print(f"skipping {image_path.name}: ground truth missing fields {missing}")
            continue

        examples.append({
            "inputs": {"image_path": str(image_path)},
            "outputs": ground_truth,
        })
    return examples


def upload(dataset_name: str) -> None:
    examples = build_examples()
    if not examples:
        raise SystemExit(
            "No valid (image, ground_truth) pairs found. Add filled forms to data/filled/ "
            "and matching JSON files to data/ground_truth/ — see README."
        )

    client = Client()
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists ({dataset.id}) — adding examples")
    else:
        dataset = client.create_dataset(
            dataset_name,
            description="Handwritten and typed employee-onboarding-form field extraction benchmark.",
        )
        print(f"Created dataset '{dataset_name}' ({dataset.id})")

    client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(examples)} examples to '{dataset_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="form-extraction-bench")
    args = parser.parse_args()
    upload(args.dataset_name)


if __name__ == "__main__":
    main()
