import numpy as np
from src.decomposition.pca import PCAFeatureExtractor


def test_pca():
    """
    PCAFeatureExtractor Test
    method: fit, transform
    """
    dummy_data = np.random.rand(10, 10, 10, 10)
    extractor = PCAFeatureExtractor(10)
    extractor.fit(dummy_data)
    extractor.transform(dummy_data)
