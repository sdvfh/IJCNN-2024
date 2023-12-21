import json
from itertools import product

import keras_nlp
import pandas as pd
import pennylane as qml
import xgboost as xgb
from cuml.ensemble import RandomForestClassifier as cuRFC
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC as skSVC


class Model:
    name = None
    _hyperparameters = {}
    _default_params = {}

    def __init__(self, qsa):
        self._qsa = qsa
        self._actual_params = {}
        self.time_training = None
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


class Processing:
    def __init__(self):
        self._qsa = None

    def process_data(self, test=False):
        raise NotImplementedError


class ClassicalProcessing(Processing):
    def process_data(self, test=False):
        pass


class QuantumProcessing(Processing):
    def process_data(self, test=False):
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        self._qsa.df["train"]["data_transformed"] = scaler.fit_transform(
            self._qsa.df["train"]["data"]
        )
        if test:
            self._qsa.df["test"]["data_transformed"] = scaler.transform(
                self._qsa.df["test"]["data"]
            )
        else:
            self._qsa.df["dev"]["data_transformed"] = scaler.transform(
                self._qsa.df["dev"]["data"]
            )


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
    _hyperparameters = {
        "n_layers": [1, 4, 8, 16],
    }

    def __init__(self, qsa):
        super().__init__(qsa)
        self._num_qubits = 10

    def set_seed(self):
        self._actual_params["seed"] = self._qsa.seed

    def train(self, test=False):
        dev = qml.device("lightning.gpu", wires=self._num_qubits)

        def log_loss(labels, predictions):
            epsilon = (
                1e-15  # Small value to prevent logarithm from approaching infinity
            )

            loss = 0
            for label, prediction in zip(labels, predictions, strict=True):
                prediction = np.clip(
                    prediction, epsilon, 1 - epsilon
                )  # Clip probabilities to avoid log(0) or log(1)
                loss = loss - (
                    label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
                )

            loss = loss / len(labels)
            return loss

        def cost(weights, bias, X, Y):
            predictions = [variational_classifier(weights, bias, x) for x in X]
            return log_loss(Y, predictions)

        def accuracy(labels, predictions):
            loss = 0
            for label, prediction in zip(labels, predictions, strict=True):
                if abs(label - prediction) < 1e-5:
                    loss = loss + 1
            loss = loss / len(labels)

            return loss

        @qml.qnode(dev)
        def circuit(weights, features):
            qml.AmplitudeEmbedding(
                features=features,
                wires=range(self._num_qubits),
                normalize=True,
                pad_with=0,
            )
            # n_selected_features = 2 ** self._num_qubits
            # for i in range(0, len(features), n_selected_features):
            #     selected_features = features[i: i + n_selected_features]
            #     selected_features = qml.AmplitudeEmbedding._preprocess(
            #         features=selected_features,
            #         wires=range(self._num_qubits),
            #         normalize=True,
            #         pad_with=0
            #     )
            #     qml.MottonenStatePreparation(state_vector=selected_features, wires=range(self._num_qubits))

            self._layer(weights, wires=range(self._num_qubits))

            return qml.expval(qml.PauliZ(0))

        def variational_classifier(weights, bias, feature):
            return circuit(weights, feature) + bias

        def transform_y(old_y):
            new_y = np.array(old_y) * 2 - np.ones(len(old_y))
            new_y = np.array(new_y, requires_grad=False)
            return new_y

        opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)
        weights_init = self._get_initial_weights()
        bias_init = np.array(0.0, requires_grad=True)

        best_acc_dev = 0.0
        patience = 20
        no_improvement_count = 0

        weights = weights_init.copy()
        bias = bias_init.copy()
        # best_weights = weights.copy()
        # best_bias = bias.copy()

        x_train = np.array(
            self._qsa.df["train"]["data_transformed"], requires_grad=False
        )
        y_train = transform_y(self._qsa.df["train"]["labels"])
        if test:
            x_dev = np.array(
                self._qsa.df["test"]["data_transformed"], requires_grad=False
            )
            y_dev = transform_y(self._qsa.df["test"]["labels"])

            x_train = np.concatenate(
                [x_train, self._qsa.df["dev"]["data_transformed"]], axis=0
            )
            y_train = np.concatenate(
                [y_train, transform_y(self._qsa.df["dev"]["labels"])], axis=0
            )
        else:
            x_dev = np.array(
                self._qsa.df["dev"]["data_transformed"], requires_grad=False
            )
            y_dev = transform_y(self._qsa.df["dev"]["labels"])

        # x_train = x_train[:10]
        # y_train = y_train[:10]
        # x_dev = x_dev[:10]
        # y_dev = y_dev[:10]
        batch_size = len(x_train)

        for it in range(1000):
            batch_index = np.random.randint(0, len(x_train), (batch_size,))
            feats_train_batch = x_train[batch_index]
            Y_train_batch = y_train[batch_index]
            weights, bias, _, _ = opt.step(
                cost, weights, bias, feats_train_batch, Y_train_batch
            )

            predictions_train = [
                np.sign(variational_classifier(weights, bias, f)) for f in x_train
            ]
            predictions_dev = [
                np.sign(variational_classifier(weights, bias, f)) for f in x_dev
            ]

            acc_train = accuracy(y_train, predictions_train)
            acc_dev = accuracy(y_dev, predictions_dev)
            # acc_test = accuracy(y_test, predictions_test)
            final_cost = cost(weights, bias, x_train, y_train)
            print(
                "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc dev: {:0.7f} "
                "".format(it + 1, final_cost, acc_train, acc_dev)
            )

            if acc_dev > best_acc_dev:
                best_acc_dev = acc_dev
                # best_weights = weights.copy()
                # best_bias = bias.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

                # Check early stopping condition
            if no_improvement_count >= patience:
                print(
                    "Early stopping! No improvement for {} consecutive iterations.".format(
                        patience
                    )
                )
                break

        self.y_pred = [np.sign(variational_classifier(weights, bias, x)) for x in x_dev]
        self.y_pred = np.array(self.y_pred).copy()
        self.y_pred[self.y_pred == -1] = 0

    def _layer(self, weights, wires):
        pass

    def _get_initial_weights(self):
        pass

    def predict(self, test=False):
        pass


class QuantumAnsatz14(QuantumModel, QuantumProcessing):
    name = "QuantumAnsatz14"

    def _layer(self, weights, wires):
        for layer in range(self._actual_params["n_layers"]):
            qml.RY(weights[layer][0], wires=wires[0])
            qml.RY(weights[layer][1], wires=wires[1])
            qml.RY(weights[layer][2], wires=wires[2])
            qml.RY(weights[layer][3], wires=wires[3])
            qml.ctrl(qml.RX, (3,), control_values=True)(weights[layer][4], wires=0)
            qml.ctrl(qml.RX, (2,), control_values=True)(weights[layer][5], wires=3)
            qml.ctrl(qml.RX, (1,), control_values=True)(weights[layer][6], wires=2)
            qml.ctrl(qml.RX, (0,), control_values=True)(weights[layer][7], wires=1)
            qml.RY(weights[layer][8], wires=wires[0])
            qml.RY(weights[layer][9], wires=wires[1])
            qml.RY(weights[layer][10], wires=wires[2])
            qml.RY(weights[layer][11], wires=wires[3])
            qml.ctrl(qml.RX, (3,), control_values=True)(weights[layer][12], wires=2)
            qml.ctrl(qml.RX, (0,), control_values=True)(weights[layer][13], wires=3)
            qml.ctrl(qml.RX, (1,), control_values=True)(weights[layer][14], wires=0)
            qml.ctrl(qml.RX, (2,), control_values=True)(weights[layer][15], wires=1)

    def _get_initial_weights(self):
        return (
            np.pi
            / 2
            * np.random.randn(self._actual_params["n_layers"], 16, requires_grad=True)
        )


class StrongAnsatz(QuantumModel, QuantumProcessing):
    name = "StrongAnsatz"

    def _layer(self, weights, wires):
        qml.StronglyEntanglingLayers(weights, wires=wires)

    def _get_initial_weights(self):
        return (
            np.pi
            / 2
            * np.random.randn(
                self._actual_params["n_layers"], self._num_qubits, 3, requires_grad=True
            )
        )


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


models_template = [StrongAnsatz]
