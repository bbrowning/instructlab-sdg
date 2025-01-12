# SPDX-License-Identifier: Apache-2.0
# Third Party
from datasets import Dataset

# Local
from .pipeline import Pipeline


# This is part of the public API.
class SDG:
    def __init__(self, pipelines: list[Pipeline]) -> None:
        self.pipelines = pipelines

    def generate(self, dataset: Dataset):
        """
        Generate the dataset by running the chained pipeline steps.
        dataset: the input dataset
        """
        for pipeline in self.pipelines:
            dataset = pipeline.generate(dataset)
        return dataset
