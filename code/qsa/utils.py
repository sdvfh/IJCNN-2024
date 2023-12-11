from pathlib import Path


class Utils:
    _dataset_types = ["train", "dev", "test"]

    def __init__(self):
        self._path = {}
        self.get_path()

    def get_path(self):
        self._path = {"root": Path(__file__).parent.parent.parent.absolute()}
        self._path["data"] = self._path["root"] / "data"
