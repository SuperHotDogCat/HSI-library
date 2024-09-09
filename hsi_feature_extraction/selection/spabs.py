import numpy as np
from typing import Tuple, List, Union
from tqdm import tqdm
from ..core.core import BaseFeatureExtractor


class SpaBS(BaseFeatureExtractor):
    """
    Sparsity-based Band Selection (SpaBS) for hyperspectral images.

    This class implements the SpaBS algorithm, which uses K-SVD to obtain a sparse
    representation of hyperspectral image data and selects the most important bands
    based on the sparsity of the coefficients.
    """

    def __init__(
        self,
        sparsity_level: float,
        num_bands_to_select: int,
        max_iter: int = 30,
        tol: float = 1e-6,
    ):
        """
        Initialize the SpaBS band selector.

        Args:
            sparsity_level: The desired sparsity level, between 0 and 1.
            num_bands_to_select: The number of bands to select.
            max_iter: Maximum number of iterations for K-SVD algorithm. Defaults to 30.
            tol: Tolerance for convergence in K-SVD algorithm. Defaults to 1e-6.
        """
        self.sparsity_level = sparsity_level
        self.num_bands_to_select = num_bands_to_select
        self.max_iter = max_iter
        self.tol = tol
        self.selected_bands = None

    def fit(self, Y: np.ndarray) -> None:
        """
        Fit the band selector on a batch of hyperspectral images.

        Args:
            Y: Batch of hyperspectral image data. Shape: (B, C, H, W) where B is the batch size,
               C is the number of channels (bands), H is the height, and W is the width.

        Returns:
            self: The fitted SpaBSBandSelector object.
        """
        if Y.ndim != 4:
            raise ValueError("Y must be a 4D array with shape (B, C, H, W)")

        B, C, H, W = Y.shape

        if not 0 < self.sparsity_level < 1:
            raise ValueError("sparsity_level must be between 0 and 1")
        if self.num_bands_to_select <= 0 or self.num_bands_to_select > C:
            raise ValueError(
                "num_bands_to_select must be positive and not greater than the number of bands"
            )

        K = int(self.sparsity_level * C)

        # Reshape batch to (C, B*H*W)
        Y_reshaped = Y.transpose(1, 0, 2, 3).reshape(C, -1)

        # Apply K-SVD
        ksvd = KSVD(n_components=C, max_iter=self.max_iter, tol=self.tol)
        _, X = ksvd.fit(Y_reshaped)

        # Select top K entries for each column
        X_s = np.argsort(-np.abs(X), axis=0)[:K, :]

        # Calculate histogram
        hist = np.bincount(X_s.flatten(), minlength=C)

        # Select top K bands based on histogram
        self.selected_bands = np.sort(np.argsort(-hist)[: self.num_bands_to_select])

    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Transform the input hyperspectral image(s) by selecting only the chosen bands.

        Args:
            Y: Hyperspectral image data. Can be:
               - A single image with shape (C, H, W)
               - A batch of images with shape (B, C, H, W)
               - A list of images, each with shape (C, H, W)

        Returns:
            Transformed image(s) with only selected bands. The output shape will be:
            - (K, H, W) for a single input image
            - (B, K, H, W) for a batch input
            - A list of arrays with shape (K, H, W) for a list input
            where K is num_bands_to_select.
        """
        if self.selected_bands is None:
            raise ValueError("Fit the selector first before transforming")

        if isinstance(Y, list):
            return [self._transform_single(img) for img in Y]
        elif Y.ndim == 3:
            return self._transform_single(Y)
        elif Y.ndim == 4:
            return np.stack([self._transform_single(img) for img in Y])
        else:
            raise ValueError(
                "Invalid input shape. Expected 3D or 4D array, or a list of 3D arrays."
            )

    def _transform_single(self, Y: np.ndarray) -> np.ndarray:
        return Y[self.selected_bands, :, :]


class KSVD:
    """
    Implementation of the K-SVD (K-Singular Value Decomposition) algorithm.

    This algorithm combines sparse coding and dictionary learning techniques
    to approximate input data as a linear combination of a few basis vectors (dictionary).
    """

    def __init__(self, n_components: int, max_iter: int = 30, tol: float = 1e-6):
        """
        Initializes the KSVD class.

        Args:
            n_components: Number of dictionary columns (basis vectors).
            max_iter: Maximum number of iterations for the algorithm. Defaults to 30.
            tol: Tolerance for convergence. Defaults to 1e-6.
        """

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fits the K-SVD algorithm to the data, learning the dictionary and sparse codes.

        Args:
            X: Input data matrix of shape (n_features, n_samples).

        Returns:
            A tuple containing:
                - Learned dictionary of shape (n_features, n_components).
                - Corresponding sparse codes of shape (n_components, n_samples).
        """
        n_features, n_samples = X.shape
        dictionary = np.random.randn(n_features, self.n_components)
        dictionary /= np.linalg.norm(dictionary, axis=0, keepdims=True)
        sparse_codes = np.zeros((self.n_components, n_samples))

        for _ in range(self.max_iter):
            # Sparse coding step
            sparse_codes = self._update_sparse_codes(X, dictionary)

            # Dictionary update step
            for k in range(self.n_components):
                index = np.nonzero(sparse_codes[k, :])[0]
                if len(index) == 0:
                    continue

                D_k = dictionary[:, k][:, np.newaxis]
                R_k = (
                    X
                    - np.dot(dictionary, sparse_codes)
                    + np.dot(D_k, sparse_codes[k : k + 1, :])
                )

                R_k_subset = R_k[:, index]
                U, S, V = np.linalg.svd(R_k_subset, full_matrices=False)
                dictionary[:, k] = U[:, 0]
                sparse_codes[k, index] = S[0] * V[0, :]

            # Check for convergence
            error = np.linalg.norm(X - np.dot(dictionary, sparse_codes))
            if error < self.tol:
                break

        return dictionary, sparse_codes

    def _update_sparse_codes(self, X: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
        """
        Updates sparse codes using the Orthogonal Matching Pursuit (OMP) algorithm.

        Args:
            X: Input data matrix of shape (n_features, n_samples).
            dictionary: Current dictionary of shape (n_features, n_components).

        Returns:
            Updated sparse codes of shape (n_components, n_samples).
        """

        n_samples = X.shape[1]
        sparse_codes = np.zeros((self.n_components, n_samples))

        for i in tqdm(range(n_samples)):
            residual = X[:, i]
            support = []
            for _ in range(self.n_components):
                correlations = np.dot(dictionary.T, residual)
                index = np.argmax(np.abs(correlations))
                support.append(index)

                D_support = dictionary[:, support]
                x_support, _, _, _ = np.linalg.lstsq(D_support, X[:, i], rcond=None)

                residual = X[:, i] - np.dot(D_support, x_support)

                if np.linalg.norm(residual) < 1e-6:
                    break

            sparse_codes[support, i] = x_support

        return sparse_codes
