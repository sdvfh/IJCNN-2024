from time import time

from .models import models_template
from .preprocessing import Preprocessing
from .utils import Utils


class QSA(Utils):
    def __init__(self, df):
        super().__init__()
        self.df = df

    def run(self):
        for model_template in models_template:
            self._model = model_template(self)

            print(f"---------- Running {self._model.name:>22} ----------")
            combinations = self._model.get_combinations()

            for combination_idx, combination in enumerate(combinations):
                self._run(combination_idx, combination, test=False)

            best_combination_idx, best_combination = self._model.get_best_combination()
            print("Testing best combination")

            self._run(best_combination_idx, best_combination, test=True)

    def _run(self, best_combination_idx, best_combination, test):
        print(f"{f'----- Combination {best_combination_idx:03d} -----':^52}")
        self._model.set_combination(best_combination_idx, best_combination)
        for seed in range(self._n_rep):
            print(f"{f'----- Rep {seed:02d} -----':^52}")
            self.seed = seed
            if self.results_exists(test=test):
                continue
            self._model_pipeline(test=test)

    def _model_pipeline(self, test=False):
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
