import keras_nlp
import numpy as np
import tensorflow as tf
from pytreebank import load_sst

from .utils import Utils


class Preprocessing(Utils):
    def __init__(self):
        super().__init__()
        self._df = None
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.process()

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

    def process(self):
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
                print(f"Remaining: {len(df) - i}")
                if (path_output / f"{i}.npy").exists():
                    continue
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
