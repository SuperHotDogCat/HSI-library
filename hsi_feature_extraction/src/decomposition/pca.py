from typing import Union
from sklearn.decomposition import PCA
import numpy as np
from src.core.core import BaseFeatureExtractor


class PCAFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, n_components, **kwargs):
        """
        Reduce redundancy optical features by PCA
        Args:
            n_components: The arguments of sklearn PCA. you can see more details in https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.
            **kwargs: The arguments of sklearn PCA. you can see more details in https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html.
        Returns:
            None
        """
        super().__init__()
        self.feature_extractor = PCA(n_components, **kwargs)
        self.n_components = n_components

    def transform(self, x: np.array) -> np.array:
        """
        Execute the reduction of redundancy optical features by PCA
        The process:
            1. reshape x's shape (B, C, H, W) -> (B, C * H * W)
            2. reduce features by PCA, and then return the reduced array. (B, C * H * W) -> (B, Features) where Features < C * H * W.

        Args:
            x: np.array, the shape is (B, C, H, W)
        Returns:
            x_transformed: np.array, the shape is 2-dimensional.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        x_transformed = self.feature_extractor.transform(x)
        return x_transformed

    def fit(self, x: np.array):
        """
        Execute the fitting of PCA
        The process:
            1. reshape x's shape (B, C, H, W) -> (B, C * H * W)
            2. train PCA.

        Args:
            x: np.array, the shape is (B, C, H, W)
        Returns:
            None
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        self.feature_extractor.fit(x)
        self.n_components = self.feature_extractor.n_components_
        return
