import numpy as np
from numpy.core.multiarray import array as array
from sklearn.cluster import SpectralClustering

from ..core.core import BaseFeatureExtractor


class LaplacianregularizedLowRankSubspaceClustering(BaseFeatureExtractor):
    """
    Batch Laplacian-regularized Low-Rank Subspace Clustering (LLRSC) for hyperspectral image band selection.

    This class implements the LLRSC algorithm for effective band selection in multiple hyperspectral images.

    Paper:
        H. Zhai, H. Zhang, L. Zhang and P. Li,
        "Laplacian-Regularized Low-Rank Subspace Clustering for Hyperspectral Image Band Selection,"
        in IEEE Transactions on Geoscience and Remote Sensing,
        vol. 57, no. 3, pp. 1723-1740, March 2019,
        doi: 10.1109/TGRS.2018.2868796.
        URL: https://ieeexplore.ieee.org/document/8485428
    """

    def __init__(
        self,
        lambda_param=0.1,
        alpha=0.1,
        p=2,
        max_iter=100,
        tol=1e-4,
    ):
        """
        Initialize the BatchLLRSC algorithm.

        Args:
            lambda_param (float): Regularization parameter for the low-rank term.
            alpha (float): Regularization parameter for the Laplacian term.
            p (float): Order of the gradient in the Laplacian regularizer.
            max_iter (int): Maximum number of iterations for the optimization.
            tol (float): Tolerance for convergence of the optimization.
        """
        
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
    
    def get_channels(self):
        """
        Args:
            num_bands_to_select(int or None): The Number of bands to select. If "None", it will be Automatically determined.
        """
        return self.num_bands_to_select

    def fit(self, X: np.array) -> None:
        """
        Fit the BatchLLRSC model to the input data.

        Args:
            X (np.ndarray): Input hyperspectral image data of shape (batch_size, n_channels, height, width).

        Returns:
            self: The fitted BatchLLRSC object.
        """
        self.batch_size, self.n_channels, self.height, self.width = X.shape
        self.X = X.reshape(
            self.batch_size, self.n_channels, -1
        )  # Reshape to (batch_size, n_channels, height*width)
        self.C = self._optimize_batch_llrsc()
        self.adjacency_matrices = self._construct_adjacency_matrices()
        if not self.num_bands_to_select:
            self.num_bands_to_select = int(np.mean(self.estimate_band_subset_size()))
        self.selected_bands = self.select_bands(self.num_bands_to_select)

    def transform(self, x: np.array) -> np.array:
        """
        Args:
            X (np.ndarray): Input hyperspectral image data of shape (batch_size, n_channels, height, width).

        Returns:
            HSI data: hyperspectral image data of shape (batch_size, num_bands_to_select, height, width).
        """
        return x[:, self.selected_bands, :, :]

    def select_bands(self, n_select: int) -> np.array:
        """
        Select the most representative bands using spectral clustering for each image in the batch.

        Args:
            n_select (int): Number of bands to select.

        Returns:
            np.ndarray: Indices of the selected bands for each image in the batch.
        """
        selected_bands = []
        for i in range(self.batch_size):
            clustering = SpectralClustering(n_clusters=n_select, affinity="precomputed")
            labels = clustering.fit_predict(self.adjacency_matrices[i])

            image_selected_bands = []
            for cluster in range(n_select):
                cluster_indices = np.where(labels == cluster)[0]
                centroid = np.mean(self.X[i, cluster_indices, :], axis=0)
                closest_band = cluster_indices[
                    np.argmin(
                        np.sum((self.X[i, cluster_indices, :] - centroid) ** 2, axis=1)
                    )
                ]
                image_selected_bands.append(closest_band)

            selected_bands.append(np.array(image_selected_bands))

        return np.array(selected_bands)

    def _optimize_batch_llrsc(self):
        """
        Optimize the BatchLLRSC objective function using ADMM for each image in the batch.

        Returns:
            list: The optimized representation coefficient matrices for each image in the batch.
        """
        C_list = []
        for i in range(self.batch_size):
            C = np.zeros((self.n_channels, self.n_channels))
            Z = np.zeros((self.n_channels, self.n_channels))
            A = np.zeros((self.n_channels, self.n_channels))
            D1 = np.zeros((self.n_channels, self.n_channels))
            D2 = np.zeros((self.n_channels, self.n_channels))

            for _ in range(self.max_iter):
                C_old = C.copy()

                C = self._update_C(Z, A, D1, D2, i)
                Z = self._update_Z(C, D1)
                A = self._update_A(C, D2)

                D1 = D1 - C + Z
                D2 = D2 - C + A

                if np.linalg.norm(C - C_old, "fro") < self.tol:
                    break

            C_list.append(C)

        return C_list

    def _update_C(self, Z, A, D1, D2, image_index):
        """
        Update the representation coefficient matrix C for a specific image.

        Args:
            Z (np.ndarray): Auxiliary variable Z.
            A (np.ndarray): Auxiliary variable A.
            D1 (np.ndarray): Lagrange multiplier D1.
            D2 (np.ndarray): Lagrange multiplier D2.
            image_index (int): Index of the current image being processed.

        Returns:
            np.ndarray: Updated representation coefficient matrix C.
        """
        I = np.eye(self.n_channels)
        X = self.X[image_index]
        return np.linalg.inv(self.lambda_param * X @ X.T + 2 * I) @ (
            self.lambda_param * X @ X.T + Z + D1 + A + D2
        )

    def _update_Z(self, C, D1):
        """
        Update the auxiliary variable Z using singular value thresholding.

        Args:
            C (np.ndarray): Current representation coefficient matrix.
            D1 (np.ndarray): Lagrange multiplier D1.

        Returns:
            np.ndarray: Updated auxiliary variable Z.
        """
        U, s, Vt = np.linalg.svd(C - D1, full_matrices=False)
        s = np.maximum(s - 1 / self.lambda_param, 0)
        return U @ np.diag(s) @ Vt

    def _update_A(self, C, D2):
        """
        Update the auxiliary variable A.

        Note: This is a simplified version. The actual update depends on the choice of p.

        Args:
            C (np.ndarray): Current representation coefficient matrix.
            D2 (np.ndarray): Lagrange multiplier D2.

        Returns:
            np.ndarray: Updated auxiliary variable A.
        """
        return C - D2

    def _construct_adjacency_matrices(self):
        """
        Construct the adjacency matrices from the representation coefficient matrices.

        Returns:
            list: The constructed adjacency matrices for each image in the batch.
        """
        adjacency_matrices = []
        for C in self.C:
            C_max = np.max(C)
            adjacency_matrices.append(C**2 / C_max)
        return adjacency_matrices

    def estimate_band_subset_size(self, threshold=0.5):
        """
        Estimate the appropriate size of the band subset using eigenvalue analysis for each image in the batch.

        Args:
            threshold (float): Threshold for eigenvalue selection.

        Returns:
            np.ndarray: Estimated number of bands to select for each image in the batch.
        """
        estimated_sizes = []
        for i in range(self.batch_size):
            corr_matrix = np.corrcoef(self.X[i])
            eigenvalues = np.linalg.eigvals(corr_matrix)
            estimated_sizes.append(np.sum(eigenvalues > threshold))
        return np.array(estimated_sizes)