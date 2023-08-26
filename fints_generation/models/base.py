import pandas as pd
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

class Generator(ABC):
    @abstractmethod
    def __init__(self):
        self.columns = []
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        self.columns = data.columns
        """
        parameter data:
            index: datetime
            columns: assets
        """
        pass

    @abstractmethod
    def sample(self, index: pd.DatetimeIndex, n_samples: int) -> list:
        """
        parameter index: datetime,
        parameter n_samples: number of samples

        returns:
            list of DataFrames
        """
        pass
