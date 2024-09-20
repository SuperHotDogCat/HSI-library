import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
from src.core.core import BaseFeatureExtractor


class EnhancedFastDensityPeakbasedClustering(BaseFeatureExtractor):
    """
    Enhanced Fast Density Peakbased Clustering (E-FDPC) Band Selection for hyperspectral image.

    paper:
        S. Jia, G. Tang, J. Zhu and Q. Li,
        "A Novel Ranking-Based Clustering Approach for Hyperspectral Band Selection,"
        in IEEE Transactions on Geoscience and Remote Sensing,
        vol. 54, no. 1, pp. 88-102, Jan. 2016, doi: 10.1109/TGRS.2015.2450759.

    """

    def __init__(self, num_bands_to_select: int, cutoff: float = 1.0):
        """
        Args:
            num_bands_to_select(int): The number of bands to select by selection
            cutoff(float): Initial cutoff value used when calculating local density.
        """
        self.num_bands_to_selsect = num_bands_to_select
        self.cutoff = cutoff
        self.selected_bands = []

    def transform(self, x: np.array) -> np.array:
        """
        Args:
            x(np.array): Hyper Spectral Image with shape of (Batchsize, Channels, Height, Width)
        Return:
            Transformed image(s) with only selected bands.
        """
        b, c, h, w = x.shape
        selected_data = np.zeros((b, self.num_bands_to_selsect, h, w))
        d = self.cutoff / np.exp(self.num_bands_to_selsect / c)
        for i in range(b):
            selected_index, _ = self.transform_single(x[i], d)
            selected_data[i] = x[i][selected_index]
            self.selected_bands.append(selected_index)

        return selected_data

    def transform_single(self, x: np.array, d: float) -> tuple[list, float]:
        scaled_distance_matrix = self.compute_scaled_distance_matrix(x)
        selected_index, score = self.applyEDPBC(
            scaled_distance_matrix, self.num_bands_to_selsect, d
        )
        return selected_index, score

    def applyEDPBC(self, D: np.array, k: int, dc: float) -> tuple[np.array, np.array]:
        """
        Selects bands and computes labels based on a scaled similarity matrix.

        Args:
            D (numpy.ndarray): A scaled similarity matrix of shape (L, L).
            k (int): The number of selected bands.
            dini (float): The initial value for dc.

        Returns:
            tuple: A tuple containing:
                - C (numpy.ndarray): The indexes of the selected bands of shape (k,).
                - A (numpy.ndarray): The labels for each band of shape (L,).
        """
        L = D.shape[0]

        # Compute local_density for each band
        local_density = np.zeros(L)
        for i in range(L):
            local_density[i] = np.sum(np.exp(-((D[i, :] / dc) ** 2)))

        # Sort local_density in descending order
        I = np.argsort(-local_density)

        # Calculate δ for each band
        δ = np.zeros(L)
        δ[I[0]] = -1  # δI1 = -1
        for i in range(1, L):
            δ[I[i]] = np.max(D[I[i], :])
            for j in range(i):
                if D[I[i], I[j]] < δ[I[i]]:
                    δ[I[i]] = D[I[i], I[j]]

        δ[I[0]] = np.max(δ)

        # Normalize local_density and δ
        local_density_min, local_density_max = np.min(local_density), np.max(
            local_density
        )
        δ_min, δ_max = np.min(δ), np.max(δ)

        local_density_normalized = (local_density - local_density_min) / (
            local_density_max - local_density_min
        )
        δ_normalized = (δ - δ_min) / (δ_max - δ_min)

        # Compute score using square weight measure
        score = local_density_normalized * δ_normalized**2

        # Sort score in descending order
        I = np.argsort(-score)

        # Get the indexes of the selected bands
        C = I[:k]

        # Compute the labels for each band
        A = np.zeros(L)
        for i in range(L):
            j = I[i]
            A[i] = np.min(D[C, j])

        return C, A

    def squared_distance(self, Ri: np.array, Rj: np.array):
        Ri = np.array(Ri)
        Rj = np.array(Rj)

        distance = np.sum((Ri - Rj) ** 2)
        return distance

    def compute_scaled_distance_matrix(self, matrices: np.array):
        L = matrices.shape[0]
        distance_matrix = np.zeros((L, L))

        for i in range(L):
            for j in range(i, L):
                distance = self.squared_distance(matrices[i], matrices[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

        scaled_distance_matrix = np.sqrt(distance_matrix) / L
        return scaled_distance_matrix
