import numpy as np
from typing import Tuple
from ..core.core import BaseFeatureExtractor

"""
Implementation of Fisher's Linear Discriminant.
"""


class FisherLinearDiscriminant(BaseFeatureExtractor):
    """A band selection algorithm based on Fisher's Linear Discriminant.

    Paper:
        Y. Harari, Z. Bar-Yehuda and S. R. Rotman,
        "Band selection in hyperspectral images based on the Fisher Linear Discriminant classifier,"
        2010 IEEE 26-th Convention of Electrical and Electronics Engineers in Israel, Eilat, Israel,
        2010, pp. 000060-000062, doi: 10.1109/EEEI.2010.5661942.
        URL: https://ieeexplore.ieee.org/document/5661942

    """

    def __init__(self, num_bands_to_select: int):
        super().__init__()
        self.num_bands_to_select = num_bands_to_select
        self.selected_bands = []

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input hyperspectral image(s) by Fisher's Linear Discriminant.

        Args:
            X (np.ndarray): Hyperspectral image data of shape
                (i) (Batch, Channels, Height, Width).
                (ii) (Channels, Height, Width).

        Returns:
            np.array: Transformed image data with shape
                (i) (Batch, K-Channels, Height, Width)
                (ii) (K-Channels, Height, Width)
            where K-Channels are the selected bands.
        """

        if X.ndim == 3:
            return X[self.selected_bands, :, :]
        elif X.ndim == 4:
            return X[:, self.selected_bands, :, :]
        else:
            raise ValueError(
                "Input X must be a 3D array (single image) or 4D array (batch of images)."
            )

    def get_num_channels(self) -> int:
        """
        Get the current number of channels

        Returns:
            num_bands_to_select (int): Number of bands to select.
        """

        if not hasattr(self, "num_bands_to_select"):
            raise AttributeError(
                "num_bands_to_select property does not exist in the class."
            )
        return self.num_bands_to_select

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Fisher Linear Discriminant model.

        Args:
            X (np.ndarray): Input data of shape
                (i) (Batch, Channels, Height, Width)
                (ii) (Channels, Height, Width)
            y (np.ndarray): Ground truth labels of shape
                (i) (Batch, Height, Width)
                (ii) (Height, Width)
        """

        if X.ndim == 3:
            # Reshape the size of the input data to (1, Channels, Height, Width) using newaxis
            X = X[np.newaxis, :, :, :]
            y = y[np.newaxis, :, :]

        S_W, S_B = self.compute_scatter_matrices(X, y)

        # Solve the eigenvalue problem: (S_W^-1 * S_B) * w = λ * w
        eig_values, eig_vectors = np.linalg.eig(np.matmul(np.linalg.inv(S_W), S_B))

        # Sort the eigenvectors by eigenvalues in descending order
        sorted_indices = np.argsort(eig_values)[::-1]
        sorted_eig_vectors = eig_vectors[:, sorted_indices]

        # Get the band ranking using the top eigenvector that corresponds to the highest separation
        top_eig_vector = sorted_eig_vectors[:, 0]
        band_ranking = np.argsort(np.abs(top_eig_vector))[::-1]

        # Select the top num_bands_to_select bands
        self.selected_bands = band_ranking[: self.num_bands_to_select]

    def compute_scatter_matrices(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute within-class scatter matrix (S_W) and between-class scatter matrix (S_B).

        Args:
            X (np.ndarray): Hyperspectral image of shape (Channels, Height, Width).
            y (np.ndarray): Ground truth labels of shape (Height, Width).

        Returns:
            S_W (np.ndarray): Within-class scatter matrix of shape (Channels, Channels).
            S_B (np.ndarray): Between-class scatter matrix of shape (Channels, Channels).
        """

        B, C, H, W = X.shape

        # Reshape X into (Channels, Pixels = Batch*Height*Width)
        X = X.reshape(C, -1)

        # Reshape y into (Pixels = Batch*Height*Width,)
        y = y.flatten()

        # Calculate the overall mean vector (Channels,)
        overall_mean = np.mean(X, axis=1)

        # Initialize the within-class scatter matrix S_W (C, C) and between-class scatter matrix S_B (C, C)
        S_W = np.zeros((C, C))
        S_B = np.zeros((C, C))

        # Get the unique class labels
        classes = np.unique(y)

        # Calculate the within-class scatter matrix and between-class scatter matrix
        for cls in classes:
            # Get indices of pixels belonging to the current class
            indices_in_class = np.where(y == cls)[0]

            # Extract pixels of the current class
            X_class = X[:, indices_in_class]

            # Calculate the mean vector for each class
            class_mean = np.mean(X_class, axis=1, keepdims=True)

            # Calculate the within-class scatter matrix
            centered_X_class = X_class - class_mean
            S_W += np.dot(centered_X_class, centered_X_class.T)

            # Calculate the between-class scatter matrix
            num_pixels_in_class = indices_in_class.size
            mean_diff = class_mean - overall_mean.reshape(-1, 1)
            S_B += num_pixels_in_class * np.dot(mean_diff, mean_diff.T)

        return S_W, S_B
