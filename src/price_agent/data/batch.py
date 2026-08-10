from __future__ import annotations

import os
import json
import pickle
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
from typing import Any
from tqdm.notebook import tqdm

load_dotenv(override=True)
groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def _resolve_project_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()


MODEL = os.environ.get("GROQ_MODEL")
PROJECT_ROOT = _resolve_project_root(Path(__file__).resolve())
PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "02-preprocessed"
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = PREPROCESSED_ROOT / "batches.pkl"

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Batch:
    BATCH_SIZE = 1_000

    batches: list[Batch] = []

    def __init__(self, items: list[Any], start: int, end: int, lite: bool):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id: str | None = None
        self.batch_id: str | None = None
        self.output_file_id: str | None = None
        self.done = False
        folder = "lite" if lite else "full"
        self.batches_dir = PREPROCESSED_ROOT / BATCHES_FOLDER / folder
        self.output_dir = PREPROCESSED_ROOT / OUTPUT_FOLDER / folder
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state.parent.mkdir(parents=True, exist_ok=True)

    def make_jsonl(self, item: Any) -> str:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item.full},
            ],
            "reasoning_effort": "low",
        }
        line = {
            "custom_id": str(item.id),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        return json.dumps(line)

    def make_file(self) -> None:
        batch_file = self.batches_dir / self.filename
        with batch_file.open("w", encoding="utf-8") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self) -> None:
        batch_file = self.batches_dir / self.filename
        with batch_file.open("rb") as f:
            response = groq.files.create(file=f, purpose="batch")
        self.file_id = response.id

    def submit_batch(self) -> None:
        if self.file_id is None:
            raise ValueError("file_id is missing. Run send_file() before submit_batch().")
        response = groq.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=self.file_id,
        )
        self.batch_id = response.id

    def is_ready(self) -> bool:
        if self.batch_id is None:
            raise ValueError("batch_id is missing. Run submit_batch() before is_ready().")
        response = groq.batches.retrieve(self.batch_id)
        status = response.status
        if status == "completed":
            self.output_file_id = response.output_file_id
        return status == "completed"

    def fetch_output(self) -> None:
        if self.output_file_id is None:
            raise ValueError("output_file_id is missing. Wait for batch completion before fetch_output().")
        output_file = str(self.output_dir / self.filename)
        response = groq.files.content(self.output_file_id)
        response.write_to_file(output_file)

    def apply_output(self) -> None:
        output_file = str(self.output_dir / self.filename)
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                json_line = json.loads(line)
                item_id = int(json_line["custom_id"])
                summary = json_line["response"]["body"]["choices"][0]["message"]["content"]
                self.items[item_id].summary = summary
        self.done = True

    @classmethod
    def create(cls, items: list[Any], lite: bool) -> None:
        cls.batches = []
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls) -> None:
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls) -> None:
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls) -> None:
        if not cls.batches:
            raise ValueError("No batches available to save.")
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items: list[Any]) -> None:
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
