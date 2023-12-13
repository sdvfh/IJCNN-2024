from itertools import product

import xgboost as xgb


class Model:
    name = None

    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
        self.y_test_pred = None

    def run(self):
        self._process()
        self._train()
        pass

    def _process(self):
        pass

    def _train(self):
        pass


class ClassicModel(Model):
    _hyperparameters = {}

    def _train(self):
        all_combinations = self._make_all_combinations()
        for combination in all_combinations:
            self._train_model(combination)
            self._orchestrator.save_results()

    def _make_all_combinations(self):
        return []
        # Get all combinations of hyperparameters
        # param_combinations = list(product(*self._hyperparameters.values()))

        # Create a list of dictionaries for each combination
        # param_dicts = [
        #     {
        #         param_name: param_value
        #         for param_name, param_value in zip(
        #             self._hyperparameters.keys(), combination
        #         )
        #     }
        #     for combination in param_combinations
        # ]
        # return param_dicts

    def _train_model(self, combination):
        pass


class XGBoost(ClassicModel):
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

    def _train_model(self, combination):
        actual_params = {
            **self._default_params,
            **combination,
            "seed": self._orchestrator.seed,
        }
        dtrain = xgb.DMatrix(
            self._orchestrator.df["train"]["data"],
            label=self._orchestrator.df["train"]["labels"],
        )
        ddev = xgb.DMatrix(
            self._orchestrator.df["dev"]["data"],
            label=self._orchestrator.df["dev"]["labels"],
        )

        es = xgb.callback.EarlyStopping(
            rounds=20,
            min_delta=1e-4,
            save_best=True,
            data_name="eval",
            metric_name="logloss",
        )
        ml = xgb.train(
            actual_params,
            dtrain,
            1_000_000,
            evals=[(dtrain, "train"), (ddev, "eval")],
            callbacks=[es],
        )
        self.y_test_pred = ml.predict(
            xgb.DMatrix(self._orchestrator.df["test"]["data"])
        )


class RandomForest(ClassicModel):
    name = "RandomForest"


class SVM(ClassicModel):
    name = "SVM"


class QuantumModel(Model):
    pass


class QuantumAnsatz14(QuantumModel):
    name = "QuantumAnsatz14"


class QuantumAnsatz6Original(QuantumModel):
    name = "QuantumAnsatz6Original"


class QuantumAnsatz6Modified(QuantumModel):
    name = "QuantumAnsatz6Modified"


class QuantumAnsatzStrong(QuantumModel):
    name = "QuantumAnsatzStrong"


models = {
    XGBoost.name: XGBoost,
    RandomForest.name: RandomForest,
    SVM.name: SVM,
    QuantumAnsatz14.name: QuantumAnsatz14,
    QuantumAnsatz6Original.name: QuantumAnsatz6Original,
    QuantumAnsatz6Modified.name: QuantumAnsatz6Modified,
    QuantumAnsatzStrong.name: QuantumAnsatzStrong,
}
