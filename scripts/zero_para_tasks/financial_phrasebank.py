import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymClassificationDataset


class FinancialPhrasebank(FewshotGymClassificationDataset):
    def __init__(self):
        self.hf_identifier = "financial_phrasebank"

        self.task_type = "classification"

        # for classification tasks, specify the meaning of each label
        self.label = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    def get_train_test_lines(self, dataset):
        # only train set, manually split 20% data as test

        lines = self.map_hf_dataset_to_list(dataset, "train")

        np.random.seed(42)
        np.random.shuffle(lines)

        n = len(lines)

        train_lines = lines[: int(0.8 * n)]
        test_lines = lines[int(0.8 * n) :]

        return train_lines, test_lines

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            lines.append((datapoint["sentence"], self.label[datapoint["label"]]))
        return lines

    def load_dataset(self):
        return datasets.load_dataset("financial_phrasebank", "sentences_allagree")


def main():
    dataset = FinancialPhrasebank()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(seed=seed, path="../data/")


if __name__ == "__main__":
    main()
