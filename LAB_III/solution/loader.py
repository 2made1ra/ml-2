"""Download the full WikiArt dataset for local notebook usage.

This script keeps all Hugging Face artifacts inside ``raw_data`` and also
stores a ready-to-load Arrow copy with ``Dataset.save_to_disk``.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "raw_data"
HF_HOME_DIR = RAW_DATA_DIR / "hf_home"
HF_HUB_CACHE_DIR = HF_HOME_DIR / "hub"
HF_DATASETS_CACHE_DIR = HF_HOME_DIR / "datasets"
LOCAL_DATASET_DIR = RAW_DATA_DIR / "wikiart_dataset"

DATASET_NAME = "huggan/wikiart"


def configure_huggingface_cache() -> None:
    """Point Hugging Face caches to LAB_III/solution/raw_data."""

    os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download huggan/wikiart into LAB_III/solution/raw_data and save "
            "a local Arrow dataset for offline loading."
        )
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing local Arrow dataset before downloading again.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DATASET_NAME,
        help=f"Hugging Face dataset id to download. Default: {DATASET_NAME}",
    )
    return parser.parse_args()


def download_dataset(dataset_name: str, force: bool = False) -> Path:
    """Download the dataset and save it to LOCAL_DATASET_DIR."""

    configure_huggingface_cache()

    from datasets import load_dataset

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    HF_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_DATASET_DIR.exists():
        if not force:
            print(f"Local dataset already exists: {LOCAL_DATASET_DIR}")
            print("Use --force to rebuild it.")
            return LOCAL_DATASET_DIR
        shutil.rmtree(LOCAL_DATASET_DIR)

    print(f"Downloading dataset: {dataset_name}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
    print(f"HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")

    dataset = load_dataset(
        dataset_name,
        cache_dir=str(HF_DATASETS_CACHE_DIR),
    )

    print(f"Saving local Arrow dataset to: {LOCAL_DATASET_DIR}")
    dataset.save_to_disk(str(LOCAL_DATASET_DIR))

    print("Done.")
    print(f"Use this path in the notebook: {LOCAL_DATASET_DIR}")
    return LOCAL_DATASET_DIR


def main() -> None:
    args = parse_args()
    download_dataset(dataset_name=args.dataset_name, force=args.force)


if __name__ == "__main__":
    main()
