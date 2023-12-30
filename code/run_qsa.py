import os

from qsa import QSA

use_gpu = os.environ.get("USE_GPU", "0") == "1"

model = QSA(use_gpu)
model.run()
model.metrics()
