from .models import models
from .preprocessing import Preprocessing


class Orchestrator(Preprocessing):
    def __init__(self):
        super().__init__()

    def run(self):
        self._load_dataset()
        for model_name, model_class in models.items():
            print(f"---------- Running {model_name:>22} ----------")
            for i in range(self._n_rep):
                print(f"{f'----- Rep {i:02d} -----':>22}")
                self.seed = i
                self._model = model_class(self)
                self._model.run()
                self._save_results()
