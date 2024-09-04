from sklearn.decomposition import FastICA
import numpy as np
from src.core.core import BaseFeatureExtractor


class ICAFeatureExtractor(BaseFeatureExtractor):
    """Extract features using Independent Component Analysis (ICA).

    This class uses FastICA to transform input data into a set of independent components,
    which can be used as features for machine learning tasks.

    For detailes o Fast ICA, see https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    """

    def __init__(self, n_components=None, **kwargs):
        """
        Args:
            n_components (int, optional): Number of components to extract.
                If `None`, all components are kept.
            **kwargs: Additional keyword arguments passed to `FastICA`.
        """
        super().__init__()
        self.feature_extractor = FastICA(n_components, **kwargs)
        self.n_components = n_components

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x: np.array) -> np.array:
        """Transforms the input data.

        Args:
            x (np.array): Input data with shape (B, C, H, W), where B is the batch size,
                C is the number of channels, H is the height, and W is the width.

        Returns:
            np.array: Transformed data with shape (B, n_components).
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        x_transformed = self.feature_extractor.transform(x)
        return x_transformed

    def fit(self, x: np.array):
        """Fits the ICA model to the input data.

        Args:
            x (np.array): Input data with shape (B, C, H, W).

        Returns:
            None
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        self.feature_extractor.fit(x)
        _, self.n_components = self.feature_extractor.components_.shape
        return