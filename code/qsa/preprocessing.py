import keras_nlp
import numpy as np
import pandas as pd
from pytreebank import load_sst

from .utils import Utils


class Preprocessing(Utils):
    def __init__(self):
        super().__init__()
        self._df = None
        if self._data_created():
            return
        self._process()
        self._compress_data()

    @staticmethod
    def save(path, array):
        with open(path, "wb+") as fh:
            fh.write(
                "{0:} {1:} {2:}\n".format(
                    array.dtype, array.shape[0], array.shape[1]
                ).encode("ascii")
            )
            fh.write(array.data)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            header = fh.readline()
            data = fh.read()
        dtype, w, h = header.decode("ascii").strip().split()
        return np.frombuffer(data, dtype=dtype).reshape((int(w), int(h)))

    def _data_created(self):
        if (self._path["data"] / "labels.csv").exists():
            return True
        return False

    def _process(self):
        self._df = load_sst(self._path["data"])

        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            "bert_base_en_uncased", sequence_length=180
        )
        model = keras_nlp.models.BertBackbone.from_preset("bert_base_en_uncased")

        for dataset_type in self._dataset_types:
            print(f"Processing {dataset_type}")
            path_output = self._path["data"] / dataset_type
            if not path_output.exists():
                path_output.mkdir(parents=True)

            df = self._df[dataset_type]

            for i, line in enumerate(df):
                if (path_output / f"{i}.npy").exists():
                    continue
                print(f"Remaining: {len(df) - i}")
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
                self.save(path_output / f"{i}.npy", output)
                with open(path_output / "dataset_type_labels.txt", "a+") as f:
                    f.write(f"{i},{label},{original_label}")
                    f.write("\n")

    def _compress_data(self):
        labels = []
        for dataset_type in self._dataset_types:
            path_output = self._path["data"] / dataset_type
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
                output = self.load(path_output / f"{sentence}.npy")
                outputs.append(output)
            outputs = np.concatenate(outputs, axis=0)
            self.save(self._path["data"] / f"{dataset_type}.npy", outputs)
        labels = pd.concat(labels, axis=0)
        labels.to_csv(self._path["data"] / "labels.csv", index=False)
