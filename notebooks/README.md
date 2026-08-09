# Notebooks

Use this folder for exploratory data analysis, baseline modeling, retrieval experiments, and prompt evaluation.

Suggested notebooks:

- `EDA.ipynb`
- `Baseline.ipynb`
- `RAG_Experiments.ipynb`
- `Fine_Tuning.ipynb`

## Data Pipeline Reference

The fine-tuning workflow in this repository relies on four closely related files:

- `src/price_agent/data/items.py`
- `src/price_agent/data/parser.py`
- `src/price_agent/data/loaders.py`
- `notebooks/fine_tuning_frontier_modal.ipynb`

### `items.py`

`src/price_agent/data/items.py` defines the core data model used by the rest of the workflow. Its `Item` class is a Pydantic model representing one cleaned product example after preprocessing, with fields such as `title`, `category`, `price`, `full`, `weight`, and `parent_asin`.

This file also includes helper methods for downstream training workflows:

- `make_prompt()` builds prompt text for supervised fine-tuning.
- `test_prompt()` returns the prompt prefix without the answer.
- `push_to_hub()` converts lists of `Item` objects into Hugging Face datasets and uploads them.
- `from_hub()` reconstructs `Item` objects from a Hugging Face dataset.

This file answers one main question: what does a valid, cleaned training example look like?

### `parser.py`

`src/price_agent/data/parser.py` is the record-level transformation layer. It takes one raw Amazon metadata row and decides whether that row should become an `Item`.

Its responsibilities include:

- parsing and validating the price
- decoding and cleaning the `details` field
- choosing a category when one is not explicitly passed in
- building a cleaned combined text block with `scrub()`
- extracting product weight when available
- filtering out unusable rows

The main entry point is `parse(datapoint, category=None) -> Item | None`.

If a record has no valid price, too little usable text, or malformed required fields, `parse()` returns `None`. Otherwise it returns a cleaned `Item`.

This file answers: given one raw parquet row, should we keep it, and if yes, how do we normalize it?

### `loaders.py`

`src/price_agent/data/loaders.py` is the batch-loading and orchestration layer. It loads local parquet shards, splits the dataset into chunks, and processes those chunks in parallel.

Its `ItemLoader` class is responsible for:

- locating the correct `raw_meta_<category>` folder
- loading parquet files with Hugging Face `load_dataset`
- chunking the dataset for parallel work
- calling `parse()` on each datapoint
- collecting all successfully parsed `Item` objects

`loaders.py` does not define the cleaning rules itself. It delegates record cleaning to `parser.py` and produces `Item` objects defined in `items.py`.

This file answers: how do we load many raw rows from disk and convert them into usable training examples efficiently?

### `fine_tuning_frontier_modal.ipynb`

`notebooks/fine_tuning_frontier_modal.ipynb` is the interactive exploration layer built on top of the three Python modules above.

The notebook does two related things:

- it explores raw data directly by loading parquet files and manually inspecting rows
- it validates the reusable pipeline by calling `parse()` and later `ItemLoader`

In practice, the notebook is where you:

- inspect sample raw datapoints
- measure dataset size
- analyze price and text-length distributions
- load one or more local categories
- deduplicate the resulting items
- prepare for later fine-tuning steps

This file answers: how do we inspect, validate, and experiment with the data-loading pipeline interactively?

## How These Files Fit Together

The end-to-end flow is:

1. Raw product metadata is stored in local parquet files under `data/01-raw/Amazon-Reviews-2023/raw_meta_*`.
2. A raw row is read either directly in the notebook or through `ItemLoader`.
3. `parser.py` cleans that row and decides whether it is usable.
4. If it is usable, `items.py` defines the `Item` object that represents the cleaned example.
5. `loaders.py` scales that conversion across many parquet files and many rows.
6. The notebook uses the resulting `Item` objects for analysis, deduplication, and fine-tuning preparation.

A simple mental model is:

- `items.py`: schema
- `parser.py`: single-record cleaning logic
- `loaders.py`: batch loading logic
- `fine_tuning_frontier_modal.ipynb`: exploration and validation

One useful design detail is that the notebook shows both paths:

- a manual path: `dataset -> parse(...) -> items`
- a reusable path: `ItemLoader -> load() -> items`

That separation keeps the normalization logic in one place while allowing both experimentation and reusable pipeline execution.
