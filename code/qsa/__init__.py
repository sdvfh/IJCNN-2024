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
                print(f"{f'----- Combination {combination_idx:03d} -----':^52}")
                self._model.set_combination(combination_idx, combination)
                for seed in range(self._n_rep):
                    print(f"{f'----- Rep {seed:02d} -----':^52}")
                    self.seed = seed
                    if self.results_exists(test=False):
                        continue
                    self._model_pipeline()

            best_combination_idx, best_combination = self._model.get_best_combination()
            print("Testing best combination")
            print(f"{f'----- Combination {best_combination_idx:03d} -----':^52}")
            self._model.set_combination(best_combination_idx, best_combination)
            for seed in range(self._n_rep):
                print(f"{f'----- Rep {seed:02d} -----':^52}")
                self.seed = seed
                if self.results_exists(test=True):
                    continue
                self._model_pipeline(test=True)

    def _model_pipeline(self, test=False):
        self._model.set_seed()
        self._model.process_data(test=test)
        self._model.train(test=test)
        self._model.predict(test=test)
        self.save_results(test=test)

    def metrics(self):
        pass
