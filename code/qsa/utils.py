import json
from pathlib import Path

import tensorflow as tf
from pennylane import numpy as np
from scikitplot.helpers import binary_ks_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Utils:
    _dataset_types = ["train", "dev", "test"]
    _IS_BINARY_PROBLEM = True
    _n_rep = 30

    def __init__(self):
        self._model = None
        self.path = None
        self.df = None
        self.seed = 42
        self._get_path()
        self._set_seeds()

    def _get_path(self):
        self.path = {"root": Path(__file__).parent.parent.parent.absolute()}
        self.path["data"] = self.path["root"] / "data"

    def _set_seeds(self):
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()

    @staticmethod
    def save_np(path, array):
        with open(path, "wb+") as fh:
            fh.write(
                "{0:} {1:} {2:}\n".format(
                    array.dtype, array.shape[0], array.shape[1]
                ).encode("ascii")
            )
            fh.write(array.data)

    @staticmethod
    def load_np(path):
        with open(path, "rb") as fh:
            header = fh.readline()
            data = fh.read()
        dtype, w, h = header.decode("ascii").strip().split()
        return np.frombuffer(data, dtype=dtype).reshape((int(w), int(h)))

    @staticmethod
    def _compute_metrics(y_true, y_pred):
        y_pred_round = y_pred.copy().round()
        metric = {
            "accuracy": balanced_accuracy_score(y_true, y_pred_round),
            "precision": precision_score(y_true, y_pred_round),
            "recall": recall_score(y_true, y_pred_round),
            "f1": f1_score(y_true, y_pred_round),
            "kappa": cohen_kappa_score(y_true, y_pred_round),
            # "roc_auc": roc_auc_score(y_true, y_pred),
        }
        # _, _, _, metric["ks"], metric["threshold"], _ = binary_ks_curve(y_true, y_pred)
        return metric

    def save_results(self, test=False):
        if test:
            y_true = self.df["test"]["labels"]
            flag = "test"
        else:
            y_true = self.df["dev"]["labels"]
            flag = "dev"
        metrics = self._compute_metrics(y_true, self._model.y_pred)
        params_to_save = self._model.get_savable_params()
        json_to_save = {
            "metrics": metrics,
            "model_params": params_to_save,
            "training_type": flag,
            "time_training": self._model.time_training,
        }

        model_result_path = (
            self.path["data"]
            / "results"
            / self._model.name
            / flag
            / str(self._model.combination_idx)
        )
        if not model_result_path.exists():
            model_result_path.mkdir(parents=True)

        with open(model_result_path / f"{self.seed}.json", "w+") as file:
            json.dump(json_to_save, file)

    def results_exists(self, test=False):
        if test:
            flag = "test"
        else:
            flag = "dev"
        model_result_path = (
            self.path["data"]
            / "results"
            / self._model.name
            / flag
            / str(self._model.combination_idx)
        )
        if not model_result_path.exists():
            return False
        return (model_result_path / f"{self.seed}.json").exists()
