# Standard
from typing import Optional
import json

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset

# First Party
from instructlab.sdg.logger_config import setup_logger

ALLOWED_COLS = ["id", "messages", "metadata"]
logger = setup_logger(__name__)


def adjust_train_sample_size(ds: Dataset, num_samples: int):
    logger.info(f"Rebalancing dataset to have {num_samples} samples ...")
    df = ds.to_pandas()
    df = df.sample(n=num_samples, random_state=42, replace=True).reset_index(drop=True)
    return Dataset.from_pandas(df)


def load_ds(path, sampling_size, num_proc):
    logger.info(f"Loading dataset from {path} ...")
    dataset = load_dataset("json", data_files=path, split="train")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    if sampling_size != 1.0:
        if isinstance(sampling_size, int):
            num_samples = sampling_size
        else:
            num_samples = int(len(dataset) * sampling_size)
        dataset = adjust_train_sample_size(dataset, num_samples)

    # move any column that is not in ALLOWED_COLS to metadata
    def move_unallowed_cols_to_metadata(example):
        metadata = example.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        for col in dataset.column_names:
            if col not in ALLOWED_COLS:
                metadata[col] = example[col]
                example.pop(col)
        example["metadata"] = json.dumps(metadata)
        return example

    dataset = dataset.map(move_unallowed_cols_to_metadata, num_proc=num_proc)

    # check if metadata column is string if not convert it using json.dumps
    if not isinstance(dataset["metadata"][0], str):
        dataset = dataset.map(
            lambda x: {"metadata": json.dumps(x["metadata"])}, num_proc=num_proc
        )

    return dataset


def add_system_message(sample: dict, sys_prompt: str) -> dict:
    # check if the messages have role system
    has_system = False
    for msg in sample["messages"]:
        if msg["role"] == "system":
            has_system = True
            msg["content"] = sys_prompt

    if not has_system:
        sample["messages"].insert(0, {"role": "system", "content": sys_prompt})

    return sample


class Recipe:
    def __init__(
        self, initial_datasets: Optional[list] = None, sys_prompt: Optional[str] = ""
    ):
        self.recipe = {
            "datasets": initial_datasets or [],
            "sys_prompt": sys_prompt,
        }
        self.sys_prompt = self.recipe.get("sys_prompt", "")
        self.dataset_added = False

    def _create_mixed_dataset(self, num_proc):
        if not self.dataset_added:
            logger.error("No dataset added to the recipe")

        mixed_ds = [
            load_ds(dataset["path"], dataset["sampling_size"], num_proc)
            for dataset in self.recipe["datasets"]
        ]

        mixed_ds = concatenate_datasets(mixed_ds)
        mixed_ds = mixed_ds.map(
            add_system_message,
            fn_kwargs={"sys_prompt": self.sys_prompt},
            num_proc=num_proc,
        )

        # assert that the dataset only has the allowed columns
        assert set(mixed_ds.column_names) == set(
            ALLOWED_COLS
        ), "Dataset has invalid columns"
        return mixed_ds

    def add_dataset(self, path, sampling_size):
        self.dataset_added = True
        self.recipe["datasets"].append({"path": path, "sampling_size": sampling_size})

    def save_mixed_dataset(self, output_path, num_proc):
        mixed_ds = self._create_mixed_dataset(num_proc)
        mixed_ds.to_json(output_path, orient="records", lines=True)
        logger.info(f"Mixed Dataset saved to {output_path}")
