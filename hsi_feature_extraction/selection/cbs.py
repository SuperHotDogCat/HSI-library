import numpy as np
from scipy import linalg
from ..core.core import BaseFeatureExtractor
from collections import Counter


class LinearlyConstrainedMinimumVarianceCBS(BaseFeatureExtractor):
    """
    Linearly Constrained Minimum Variance Constrained Band Selection (LCMV-CBS) for hyperspectral imagery.

    This class implements the LCMV-CBS algorithm for band selection in hyperspectral images
    as described in the paper "Constrained Band Selection for Hyperspectral Imagery"
    by Chein-I Chang and Su Wang. It supports batch processing of multiple images.

    paper:
        Chein-I Chang and Su Wang,
        "Constrained band selection for hyperspectral imagery,"
        in IEEE Transactions on Geoscience and Remote Sensing,
        vol. 44, no. 6, pp. 1575-1585, June 2006, doi: 10.1109/TGRS.2006.864389.
        https://ieeexplore.ieee.org/document/1634721

    """

    def __init__(self, num_bands_to_select):
        """
        Initialize the LCMV_CBS object.
        Args:
            num_bands_to_select (int): Number of bands to select.
        """
        self.num_bands_to_select = num_bands_to_select

    def get_num_channels(
        self,
    ):
        if not hasattr(self, "num_bands_to_select"):
            raise AttributeError(
                "num_bands_to_select property does not exist in the class."
            )
        return self.num_bands_to_select

    def fit(self, X: np.array) -> None:
        if X.ndim == 3:
            self.selected_bands = self._fit_single(X)
        elif X.ndim == 4:
            self._fit_batch(X)

    def transform(self, X: np.array) -> np.array:
        """
        Perform band selection using the LCMV-CBS algorithm.

        This method supports both single image and batch processing.

        Args:
            X (np.ndarray): Either a single hyperspectral image cube with shape (n_channels, height, width),
                            or a batch of hyperspectral image cubes with shape (batch_size, n_channels height, width).

        Returns:
            np.ndarray: If input is a single image, returns a single image of selected bands.
                        If input is a batch, returns selected bands for each image with shape (batch_size, n_selected_bands height, width).

        Raises:
            ValueError: If X is not a 3D or 4D array.
        """
        if X.ndim == 3:
            return X[self.selected_bands, :, :]
        elif X.ndim == 4:
            return X[:, self.selected_bands, :, :]
        else:
            raise ValueError(
                "Input X must be a 3D array (single image) or 4D array (batch of images)."
            )

    def _fit_single(self, X: np.array) -> np.array:
        """
        Perform band selection on a single hyperspectral image.

        Args:
            X(np.ndarray):  a single hyperspectral image cube with shape (n_channels, height, width)

        Returns:
            np.ndarray: a single image of selected bands.

        """
        c, h, w = X.shape
        b = X.reshape(-1, c).T  # Reshape to (n_channels, n_pixels)

        # Compute correlation matrix
        R = np.cov(b)

        # Compute weights for each band
        weights = np.zeros((c, c))
        for i in range(c):
            e = np.zeros(c)
            e[i] = 1
            weights[:, i] = linalg.solve(R, e) / (e.T @ linalg.solve(R, e))

        # Compute band correlation measure
        bcm = np.sum(weights * R @ weights, axis=0)

        # Select top num_bands_to_select
        selected_bands = np.argsort(bcm)[::-1][: self.num_bands_to_select]

        return selected_bands  # X[self.selected_bands, :, :]

    def _fit_batch(self, X: np.array) -> None:
        """
        Perform band selection on a batch of hyperspectral images.

        Args:
            X(np.ndarray):  a batch of hyperspectral image cubes with shape (batch_size, n_channels height, width).

        Returns:
            np.ndarray: selected bands for each image with shape (batch_size, n_selected_bands height, width).

        """
        batch_size, c, h, w = X.shape
        img_selected_bands = np.zeros(
            (batch_size, self.num_bands_to_select), dtype=X.dtype
        )

        for i in range(batch_size):
            img_selected_bands[i] = self._fit_single(X[i])

        self.select_most_frequent(img_selected_bands)

    def select_most_frequent(self, matrix):
        n = self.num_bands_to_select
        flattened = matrix.flatten()
        counter = Counter(flattened)
        most_common = counter.most_common(n)
        self.selected_bands_detail = sorted(most_common, key=lambda x: x[0])
        self.selected_bands = [int(i[0]) for i in self.selected_bands_detail]
