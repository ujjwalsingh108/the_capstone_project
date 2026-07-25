from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a fine-tuning scaffold job")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Prepared fine-tuning run for model={args.model_name} dataset={args.dataset_path} output={output_dir}"
    )


if __name__ == "__main__":
    main()
