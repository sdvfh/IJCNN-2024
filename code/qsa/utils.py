from pathlib import Path

import numpy as np
import tensorflow as tf


class Utils:
    _dataset_types = ["train", "dev", "test"]

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self._model = None
        self.seed = None
        self._path = {}
        self.df = {}

        self._n_rep = 30
        self._binary_problem = True

        self.get_path()

    def get_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.absolute()}
        self._path["data"] = self._path["root"] / "data"

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

    def _save_results(self):
        return
