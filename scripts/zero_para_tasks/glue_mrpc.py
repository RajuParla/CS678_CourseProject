import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset


class Glue_MRPC(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "glue-mrpc"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "not_equivalent",
            1: "equivalent",
        }

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append(
                (
                    "sentence 1: "
                    + datapoint["sentence1"]
                    + " [SEP] sentence 2: "
                    + datapoint["sentence2"],
                    self.label[datapoint["label"]],
                )
            )
        return lines

    def load_dataset(self):
        return datasets.load_dataset("glue", "mrpc")


def main():
    dataset = Glue_MRPC()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(seed=seed, path="../data/")


if __name__ == "__main__":
    main()
