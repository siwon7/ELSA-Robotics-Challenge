import gc
import gzip
import pickle
import sys
import time

import numpy as np


class CompatObservation:
    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatDataContainer:
    def __init__(self):
        self.data = None

    def __setstate__(self, state):
        self.__dict__.update(state)


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "colosseum.rlbench.datacontainer" and name == "DataContainer":
            return CompatDataContainer
        if module == "rlbench.backend.observation" and name == "Observation":
            return CompatObservation
        return super().find_class(module, name)


def load_pickled_data(path, max_retries=3, retry_sleep_sec=0.5):
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with gzip.open(path, "rb") as file_obj:
                container = CompatUnpickler(file_obj).load()
            return container.data if hasattr(container, "data") else container
        except (EOFError, OSError) as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            print(
                f"Retrying dataset load for {path} after {type(exc).__name__} "
                f"(attempt {attempt}/{max_retries})"
            )
            gc.collect()
            time.sleep(retry_sleep_sec)

    raise last_error
