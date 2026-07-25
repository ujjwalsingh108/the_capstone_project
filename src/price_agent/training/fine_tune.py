from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FineTuningJob:
    model_name: str
    dataset_path: str
    output_dir: str

    def command(self) -> list[str]:
        return [
            "python",
            "-m",
            "price_agent.training.run",
            "--model-name",
            self.model_name,
            "--dataset-path",
            self.dataset_path,
            "--output-dir",
            self.output_dir,
        ]
