from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_dataset
from concurrent.futures import ProcessPoolExecutor
from price_agent.data.parser import parse
import os

CHUNK_SIZE = 1000

cpu_count = os.cpu_count() or 1
WORKERS = max(cpu_count - 1, 1)


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


class ItemLoader:
    def __init__(self, category):
        self.category = category
        self.dataset: Dataset | None = None

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        return parse(datapoint, self.category)

    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = [self.from_datapoint(datapoint) for datapoint in chunk]
        return [item for item in batch if item is not None]

    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        assert self.dataset is not None
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """
        assert self.dataset is not None
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                results.extend(batch)
        return results

    def load(self, workers=WORKERS, data_folder: str | None = None):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"Loading dataset {self.category}", flush=True)

        if data_folder is None:
            project_root = _resolve_project_root()
            data_folder = str(project_root / "data/01-raw/Amazon-Reviews-2023" / f"raw_meta_{self.category}")

        data_path = Path(data_folder)
        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in '{data_path}'.")

        data_files = [str(path) for path in parquet_files]
        loaded_dataset = load_dataset("parquet", data_files=data_files, split="train")
        if not isinstance(loaded_dataset, Dataset):
            raise TypeError("Expected a map-style Dataset when loading local parquet files.")
        self.dataset = loaded_dataset
        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(
            f"Completed {self.category} with {len(results):,} datapoints in {(finish - start).total_seconds() / 60:.1f} mins",
            flush=True,
        )
        return results
