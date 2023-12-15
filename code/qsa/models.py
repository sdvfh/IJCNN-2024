import json
from itertools import product

import keras_nlp
import numpy as np
import pandas as pd
import xgboost as xgb
from cuml.ensemble import RandomForestClassifier as cuRFC
from sklearn.svm import SVC as skSVC


class Model:
    name = None
    _hyperparameters = {}
    _default_params = {}

    def __init__(self, qsa):
        self._qsa = qsa
        self._actual_params = {}
        self.combination_idx = None
        self._ml = None
        self.y_pred = None

    def get_combinations(self):
        # Get all combinations of hyperparameters
        param_combinations = list(product(*self._hyperparameters.values()))

        # Create a list of dictionaries for each combination
        param_dicts = [
            dict(zip(self._hyperparameters.keys(), combination, strict=False))
            for combination in param_combinations
        ]
        return param_dicts

    def get_best_combination(self):
        model_result_path = (self._qsa.path["data"] / "results" / self.name).rglob(
            "*.json"
        )
        results = []
        for file in model_result_path:
            combination_idx = int(file.parent.name)
            with open(file, "r") as f:
                result = json.load(f)

            model_params, columns = self.fix_savable_params(result["model_params"])

            results.append(
                {
                    "combination_idx": combination_idx,
                    "training_type": result["training_type"],
                    **result["metrics"],
                    **model_params,
                }
            )
        results = pd.DataFrame(results)
        results = results[results["training_type"] == "dev"].drop(
            columns=["training_type"]
        )
        results = results.groupby(["combination_idx"]).mean().reset_index()
        best_combination_idx = results.loc[results["f1"].idxmax(), "combination_idx"]
        other_params = results.loc[results["f1"].idxmax(), columns].to_dict()
        combinations = self.get_combinations()
        combination = combinations[best_combination_idx]
        combination.update(other_params)
        return best_combination_idx, combination

    def set_combination(self, combination_idx, combination):
        self._actual_params = {
            **self._default_params,
            **combination,
        }
        self.combination_idx = combination_idx

    def set_seed(self):
        self._actual_params["seed"] = self._qsa.seed

    def train(self, test=False):
        raise NotImplementedError

    def predict(self, test=False):
        raise NotImplementedError

    def get_savable_params(self):
        other_params = self._get_other_savable_params()
        return {"hyperparameters": self._actual_params, **other_params}

    def _get_other_savable_params(self):
        return {}

    def fix_savable_params(self, params):
        return {}, []


class ClassicalProcessing:
    def process_data(self, test=False):
        pass


class QuantumProcessing:
    def process_data(self, test=False):
        pass


class ClassicalModel(Model):
    _model_template = None

    def set_seed(self):
        self._actual_params["random_state"] = self._qsa.seed

    def train(self, test=False):
        if test:
            train = np.concatenate(
                [self._qsa.df["train"]["data"], self._qsa.df["dev"]["data"]],
                axis=0,
            )
            labels = np.concatenate(
                [self._qsa.df["train"]["labels"], self._qsa.df["dev"]["labels"]],
                axis=0,
            )
        else:
            train = self._qsa.df["train"]["data"]
            labels = self._qsa.df["train"]["labels"]
        self._ml = self._model_template(**self._actual_params)
        self._ml.fit(train, labels)

    def predict(self, test=False):
        if test:
            data = self._qsa.df["test"]["data"]
        else:
            data = self._qsa.df["dev"]["data"]
        self.y_pred = self._ml.predict_proba(data)[:, 1]


class QuantumModel(Model):
    pass


class XGBoost(Model, ClassicalProcessing):
    name = "XGBoost"
    _hyperparameters = {
        "max_depth": [6, 12, 24],
        "learning_rate": [0.1, 0.01, 0.05],
        "gamma": [0, 0.25, 1.0],
        "reg_lambda": [0, 1.0, 10.0],
        "scale_pos_weight": [1, 3, 5],
    }

    _default_params = {
        "objective": "binary:logistic",
        "seed_per_iteration": True,
        "device": "gpu",
    }

    def train(self, test=False):
        if test:
            data = np.concatenate(
                [self._qsa.df["train"]["data"], self._qsa.df["dev"]["data"]],
                axis=0,
            )
            labels = np.concatenate(
                [self._qsa.df["train"]["labels"], self._qsa.df["dev"]["labels"]],
                axis=0,
            )
            dtrain = xgb.DMatrix(data, label=labels)
            dtest = xgb.DMatrix(self._qsa.df["test"]["data"])
        else:
            dtrain = xgb.DMatrix(
                self._qsa.df["train"]["data"],
                label=self._qsa.df["train"]["labels"],
            )
            dtest = xgb.DMatrix(
                self._qsa.df["dev"]["data"], label=self._qsa.df["dev"]["labels"]
            )

        if test:
            actual_params = self._actual_params.copy()
            num_boost_round = int(actual_params.pop("best_iteration"))
            self._ml = xgb.train(
                actual_params,
                dtrain,
                num_boost_round,
                evals=[(dtrain, "train")],
            )
            self._ml.best_iteration = num_boost_round
        else:
            es = xgb.callback.EarlyStopping(
                rounds=20,
                min_delta=1e-4,
                save_best=True,
                data_name="test",
                metric_name="logloss",
            )
            self._ml = xgb.train(
                self._actual_params,
                dtrain,
                10_000_000,
                evals=[(dtrain, "train"), (dtest, "test")],
                callbacks=[es],
            )

    def predict(self, test=False):
        if test:
            dtest = xgb.DMatrix(self._qsa.df["test"]["data"])
        else:
            dtest = xgb.DMatrix(self._qsa.df["dev"]["data"])
        self.y_pred = self._ml.predict(
            dtest, iteration_range=(0, self._ml.best_iteration)
        )

    def _get_other_savable_params(self):
        return {
            "best_iteration": self._ml.best_iteration,
        }

    def fix_savable_params(self, params):
        params_normalized = {
            "best_iteration": params["best_iteration"],
            # **params["hyperparameters"]
        }
        return params_normalized, ["best_iteration"]


class RandomForest(ClassicalModel, ClassicalProcessing):
    name = "RandomForest"
    _hyperparameters = {
        "n_estimators": [100, 500, 100],
        "split_criterion": ["gini", "entropy"],
        "max_depth": [6, 12, 24],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4],
    }
    _default_params = {
        "n_streams": 1,
    }
    _model_template = cuRFC


class SVC(ClassicalModel, ClassicalProcessing):
    name = "SVC"
    _hyperparameters = {
        "C": [0.1, 1, 5, 10],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }
    _default_params = {
        "probability": True,
    }

    _model_template = skSVC


class BERT(Model, ClassicalProcessing):
    name = "BERT"
    _hyperparameters = {
        "epochs": [1],
        "batch_size": [2],
    }

    def train(self, test=False):
        self._ml = keras_nlp.models.BertClassifier.from_preset(
            "bert_tiny_en_uncased_sst2", num_classes=2, activation="softmax"
        )

    def predict(self, test=False):
        if test:
            data = self._qsa.df["test"]["sentences"]
        else:
            data = self._qsa.df["dev"]["sentences"]
        self.y_pred = self._ml.predict(data)[:, 1]


models_template = [BERT]
