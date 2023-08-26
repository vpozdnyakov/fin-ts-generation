import numpy as np
import pandas as pd
import pytest
from inspect import getmembers, isclass, isabstract
from fints_generation import models

models = [f[1] for f in getmembers(models, isclass) if not isabstract(f[1])]

class TestOnSyntheticData:
    def setup_class(self):
        np.random.seed(0)
        X = np.random.randn(1000, 3)
        data = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
        data.index = pd.date_range('2020-01-01', periods=1000, freq='D')
        self.data = data

    @pytest.mark.parametrize("model", models)
    def test_base(self, model):
        gen = model()
        gen.fit(self.data)
        fakes = gen.sample(self.data.index, n_samples=10)

        assert len(fakes) == 10
        assert fakes[0].shape == self.data.shape
