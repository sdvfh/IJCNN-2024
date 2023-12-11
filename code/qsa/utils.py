from pathlib import Path

import tensorflow as tf


class Utils:
    _dataset_types = ["train", "dev", "test"]

    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self._path = {}
        self.get_path()

    def get_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.absolute()}
        self._path["data"] = self._path["root"] / "data"
