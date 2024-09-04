import numpy as np
from src.selection.lasso import LassoExtractor


def test_lasso():
    dummy_data = np.random.rand(10, 10, 10, 10)
    dummy_label = np.random.randint(low=0, high=10, size=(10, 10, 10))
    extractor = LassoExtractor()
    extractor.fit(dummy_data, dummy_label, num_batches=10)
    extractor.transform(dummy_data)
