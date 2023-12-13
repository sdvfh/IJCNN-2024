from pathlib import Path

import numpy as np
import tensorflow as tf
from scikitplot.helpers import binary_ks_curve
from scipy.stats import wilcoxon
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

    def save_results(self):
        # metric = self._compute_metrics(
        #     self.df["test"]["labels"], self._model.y_test_pred
        # )
        return

    @staticmethod
    def _compute_metrics(y_true, y_pred):
        y_pred_round = y_pred.copy().round()
        metric = {
            "accuracy": balanced_accuracy_score(y_true, y_pred_round),
            "precision": precision_score(y_true, y_pred_round),
            "recall": recall_score(y_true, y_pred_round),
            "f1": f1_score(y_true, y_pred_round),
            "kappa": cohen_kappa_score(y_true, y_pred_round),
            "roc_auc": roc_auc_score(y_true, y_pred),
        }
        _, _, _, metric["ks"], metric["threshold"], _ = binary_ks_curve(y_true, y_pred)
        return metric
