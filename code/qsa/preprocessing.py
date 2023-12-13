import keras_nlp
import numpy as np
import pandas as pd
from pytreebank import load_sst

from .utils import Utils


class Preprocessing(Utils):
    def __init__(self):
        super().__init__()
        if self._data_created():
            return
        self._process()
        self._compress_data()

    def _data_created(self):
        if (self.path["data"] / "labels.csv").exists():
            return True
        return False

    def _process(self):
        df = load_sst(self.path["data"])

        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            "bert_base_en_uncased", sequence_length=180
        )
        model = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")

        for dataset_type in self._dataset_types:
            print(f"Processing {dataset_type}")
            path_output = self.path["data"] / dataset_type
            if not path_output.exists():
                path_output.mkdir(parents=True)

            dataset = df[dataset_type]

            for i, line in enumerate(dataset):
                if (path_output / f"{i}.npy").exists():
                    continue
                print(f"Remaining: {len(dataset) - i}")
                original_label, sentence = line.to_labeled_lines()[0]
                if original_label == 2:
                    label = -1
                elif original_label < 2:
                    label = 0
                else:
                    label = 1

                sentence = sentence.lower()
                tokens = preprocessor([sentence])
                output = model(tokens)["pooled_output"].numpy()
                self.save_np(path_output / f"{i}.npy", output)
                with open(path_output / "dataset_type_labels.txt", "a+") as f:
                    f.write(f"{i},{label},{original_label}")
                    f.write("\n")

    def _compress_data(self):
        labels = []
        for dataset_type in self._dataset_types:
            path_output = self.path["data"] / dataset_type
            dataset_labels = pd.read_csv(
                path_output / "dataset_type_labels.txt",
                header=None,
                names=["id", "label", "original_label"],
            )
            dataset_labels["type"] = dataset_type
            labels.append(dataset_labels)

            outputs = []
            for _, line in dataset_labels.iterrows():
                sentence = line["id"]
                output = self.load_np(path_output / f"{sentence}.npy")
                outputs.append(output)
            outputs = np.concatenate(outputs, axis=0)
            self.save_np(self.path["data"] / f"{dataset_type}.npy", outputs)
        labels = pd.concat(labels, axis=0)
        labels.to_csv(self.path["data"] / "labels.csv", index=False)

    def load_dataset(self):
        labels = pd.read_csv(self.path["data"] / "labels.csv")
        train = self.load_np(self.path["data"] / "train.npy")
        dev = self.load_np(self.path["data"] / "dev.npy")
        test = self.load_np(self.path["data"] / "test.npy")
        if self._IS_BINARY_PROBLEM:
            labels = labels[labels["original_label"] != 2]

            train = train[labels.loc[labels["type"] == "train", "id"].values]
            dev = dev[labels.loc[labels["type"] == "dev", "id"].values]
            test = test[labels.loc[labels["type"] == "test", "id"].values]
        df = {
            "train": {
                "data": train,
                "labels": labels[labels["type"] == "train"]["label"].values,
            },
            "dev": {
                "data": dev,
                "labels": labels[labels["type"] == "dev"]["label"].values,
            },
            "test": {
                "data": test,
                "labels": labels[labels["type"] == "test"]["label"].values,
            },
        }
        return df
