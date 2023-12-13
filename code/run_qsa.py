from qsa import QSA
from qsa.preprocessing import Preprocessing

preprocessing = Preprocessing()
df = preprocessing.load_dataset()

model = QSA(df=df)
model.run()
model.metrics()
