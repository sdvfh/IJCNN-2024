import os
from itertools import product, repeat
from time import time

from joblib import Parallel, delayed

from .models import models_template_all, models_template_cpu, models_template_gpu
from .preprocessing import Preprocessing
from .utils import Utils


class QSA(Utils):
    def __init__(self, use_gpu):
        super().__init__()
        self.df = Preprocessing().load_dataset()
        self.use_gpu = use_gpu

    def run(self):
        if self.use_gpu:
            self._run_models(models_template_gpu, n_jobs=2)
        else:
            self._run_models(models_template_cpu, n_jobs=5)

    def _run_models(self, models_template, n_jobs):
        all_possible_combinations = []
        for model_template in models_template.values():
            self._model = model_template(self)
            combinations = self._model.get_combinations()
            for combination_idx, combination in enumerate(combinations):
                all_possible_combinations.append(
                    (self._model.name, combination_idx, combination)
                )
        all_possible_combinations = product(
            all_possible_combinations, range(self._n_rep)
        )
        Parallel(n_jobs=n_jobs)(
            delayed(self._run)(
                model_name, combination_idx, combination, seed, test=False
            )
            for (
                model_name,
                combination_idx,
                combination,
            ), seed in all_possible_combinations
        )

        for model_name, model_template in models_template.items():
            self._model = model_template(self)
            self._model = models_template_gpu[model_name](self)
            best_combination_idx, best_combination = self._model.get_best_combination()
            print("Testing best combination")
            all_possible_combinations = zip(
                repeat(
                    (model_name, best_combination_idx, best_combination), self._n_rep
                ),
                range(self._n_rep),
                strict=True,
            )

            Parallel(n_jobs=n_jobs)(
                delayed(self._run)(
                    model_name, combination_idx, combination, seed, test=True
                )
                for (
                    model_name,
                    combination_idx,
                    combination,
                ), seed in all_possible_combinations
            )

    def _run(self, model_name, combination_idx, combination, seed, test):
        self._model = models_template_all[model_name](self)
        self._model.set_combination(combination_idx, combination)
        self.seed = seed

        if self.results_exists(test=test):
            return

        print(f"---------- Running {self._model.name:>22} ----------")
        print(f"{f'----- Combination {combination_idx:03d} -----':^52}")
        print(f"{f'----- Rep {seed:02d} -----':^52}")

        self._set_seeds()
        self._model.set_seed()
        self._model.process_data(test=test)

        self._model.time_training = time()
        self._model.train(test=test)
        self._model.time_training = time() - self._model.time_training

        print(f"Time training: {self._model.time_training:.2f} seconds")

        self._model.predict(test=test)
        self.save_results(test=test)

    def metrics(self):
        pass
