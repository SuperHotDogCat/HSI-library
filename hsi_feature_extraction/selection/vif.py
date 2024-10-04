from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
from tqdm import trange
from ..core.core import BaseFeatureExtractor

"""
vifの実装をする https://github.com/NISL-MSU/HSI-BandSelection
"""


class VIFExtractor(BaseFeatureExtractor):
    """
    This method implementation refers to Hyperspectral Dimensionality Reduction Based on Inter-Band Redundancy Analysis and Greedy Spectral Selection[https://www.mdpi.com/2072-4292/13/18/3649].
    """

    def __init__(self):
        """
        Reduce redundancy optical features by a VIF-based method.
        Args:
            None
        Returns:
            None
        """
        super().__init__()
        self.feature_extractor = VIFProcessor()

    def transform(self, x: np.array) -> np.array:
        """
        Execute the reduction of redundancy optical features by a VIF-based method.
        Extract wavelengths based on non-redundant features selected using the VIF method.
        Args:
            x: np.array, the shape is (B, C_before, H, W)
        Returns:
            x[:, self.local_minimum_indexes, :, :]: np.array, the shape is (B, C_after, H, W) where C_after <= C_before.
        """
        if len(x.shape) == 3:
            """
            x: np.array, the shape is (C_before, H, W)
            """
            return x[self.local_minimum_indexes, :, :]
        return x[:, self.local_minimum_indexes, :, :]

    def fit(self, x: np.array, threshold: float = 1e-9):
        """
        Execute the fitting of VIF
        The process is executed by VIFProcessor Class
        Args:
            x: np.array, the shape is (B, C, H, W)
            threshold: threshold used to judge whether the features are redundant or not.
        Returns:
            None
        """
        distance_distributions = (
            self.feature_extractor.calculate_interband_distances_by_vif(
                x, threshold
            )
        )
        local_minimum_indexes = (
            self.feature_extractor.get_local_minimum_indexes(
                distance_distributions
            )
        )
        self.local_minimum_indexes = local_minimum_indexes
        self.distance_distributions = distance_distributions

    def get_num_channels(self):
        if not hasattr(self, "local_minimum_indexes"):
            raise AttributeError(
                "local_minimum_indexes property does not exist in the class."
            )
        return len(self.local_minimum_indexes)


class VIFProcessor:
    def _calculate_pair_vif_value(
        self,
        data: np.array,
        feature_channel: int,
        target_channel: int,
    ) -> float:
        """
        Calculate the VIF between the channels of a feature channel and a target channel.
        Args:
            data: np.array, the shape is (B, C, H, W)
            feature_channel: the channel index of the explanatory variables needed to calculate a vif value
            target_channel: the channel index of the dependent variables needed to calculate a vif value
        Returns:
            vif_value: the VIF between the channels of a feature channel and a target channel.
        """
        x = data[
            :,
            feature_channel,
            :,
            :,
        ]
        y = data[
            :,
            target_channel,
            :,
            :,
        ]
        B, C, H, W = data.shape
        x = x.reshape(B * H * W, 1)
        y = y.reshape(B * H * W, 1)
        model = LinearRegression()
        model.fit(x, y)
        r2_score_for_x_and_y = r2_score(model.predict(x), y)
        vif_value = (1 / (1 - r2_score_for_x_and_y)) ** 2
        return vif_value

    def calculate_interband_distances_by_vif(self, x, threshold) -> np.array:
        """
        Returns the distributions of the difference of vif.
        Args:
            data: np.array, the shape is (B, C, H, W)
            threshold: The region below this threshold is considered the boundary of the VIF distance.
        Returns:
            distributions[band]: each element stores the l1 norm of `vif_distances_left - vif_distances_right`.
        """
        n_hsi_channels = x.shape[1]
        distances_left = np.zeros((n_hsi_channels))
        distances_right = np.zeros((n_hsi_channels))
        table = np.zeros((n_hsi_channels, n_hsi_channels))
        for band in trange(n_hsi_channels):
            d = 1
            vif_value = np.inf
            while vif_value > threshold and (band - d) > 0:
                if table[band, band - d] == 0:
                    table[band, band - d] = self._calculate_pair_vif_value(
                        x, band, band - d
                    )
                    table[band - d, band] = table[band, band - d]
                vif_value = table[band, band - d]
                d += 1
            distances_left[band] = d - 1

            d = 1
            vif_value = np.inf
            while vif_value > threshold and (band + d) < n_hsi_channels:
                if table[band, band + d] == 0:
                    table[band, band + d] = self._calculate_pair_vif_value(
                        x, band, band + d
                    )
                    table[band + d, band] = table[band, band + d]
                vif_value = table[band, band + d]
                d += 1
            distances_right[band] = d - 1
        return np.abs(distances_left - distances_right)

    def get_local_minimum_indexes(
        self, distance_distributions: np.array
    ) -> np.array:
        """
        Find local argmin vif distance distributions.
        Args:
            distance_distributions: calculated by calculate_interband_distances_by_vif
        Returns:
            local_minimum_indexes: local argmin vif distances
        """
        forward_roll = np.roll(distance_distributions, 1)
        forward_roll[0] = forward_roll[1]
        backward_roll = np.roll(distance_distributions, -1)
        backward_roll[-1] = backward_roll[-2]
        differential_1 = (
            forward_roll - distance_distributions >= 0
        )  # d(x_n-1) - d(x_n) >= 0
        differential_2 = (
            distance_distributions - backward_roll >= 0
        )  # d(x_n+1) - d(x_n) >= 0
        local_minimum_indexes = np.where(differential_1 & differential_2)[0]
        return local_minimum_indexes
