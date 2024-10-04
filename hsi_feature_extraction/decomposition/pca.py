from sklearn.decomposition import PCA
import numpy as np
from ..core.core import BaseFeatureExtractor


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
        if x.ndim == 3:
            C, H, W = x.shape
            x = x.transpose(1, 2, 0).reshape(-1, C)
            x_transformed = self.feature_extractor.transform(x)
            x = x_transformed.reshape(H, W, -1).transpose(2, 0, 1)
            return x
        B, C, H, W = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        x_transformed = self.feature_extractor.transform(x)
        x = x_transformed.reshape(B, H, W, -1).transpose(0, 3, 1, 2)
        return x

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
        x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        self.feature_extractor.fit(x)
        self.n_components = self.feature_extractor.n_components_
        return

    def get_num_channels(self) -> int:
        if not hasattr(self, "n_components"):
            raise AttributeError(
                "n_components property does not exist in the class."
            )

        return self.n_components
