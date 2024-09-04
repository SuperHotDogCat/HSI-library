import numpy as np
from src.selection.vif import VIFExtractor, VIFProcessor
import matplotlib.pyplot as plt


def test_vifprocessor():
    dummy_data = np.random.randn(10, 200, 30, 30)
    processor = VIFProcessor()
    distributions = processor.calculate_interband_distances_by_vif(
        dummy_data, 1e-9
    )
    indexes = processor.get_local_minimum_indexes(distributions)


def test_vifextractor():
    dummy_data = np.random.rand(10, 200, 10, 10)
    extractor = VIFExtractor()
    extractor.fit(dummy_data, threshold=5)
    extractor.transform(dummy_data)
