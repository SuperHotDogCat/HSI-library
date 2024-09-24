import numpy as np
from sklearn.linear_model import Lasso
from ..core.core import BaseFeatureExtractor
from ..utils.utils import retrieve_square_images


"""
教師データが必要。
data: (Batch,  Channels, Height, Width)
gt(ground truth): (Batch, Label)

"""


class LassoExtractor(BaseFeatureExtractor):
    """A feature extractor that uses Lasso regression for feature selection.

    This class extends the `BaseFeatureExtractor` and utilizes Lasso regression
    to select important features from the input data.

    For details of Lasso, see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    def __init__(
        self,
    ):
        self.feature_extractor = Lasso(alpha=1)

    def transform(self, x) -> np.array:
        """Transforms the input data by selecting specific channels.

        Args:
            x (np.array): Input data of shape (Batch, Channels, Height, Width).

        Returns:
            np.array: Transformed data with selected channels.
        """
        return x[:, self.selected_indices.astype(np.int32), :, :]

    def fit(self, data, gt, n=0, num_batches=25):
        """Fits the Lasso model and selects important features.

        This method extracts features from the data using Lasso regression
        and selects either a specified number of top features or all non-zero
        features based on the model coefficients.

        Args:
            data (np.array): Input data of shape (Batch, Height, Width, Channels).
            gt (np.array): Ground truth labels of shape (Batch, Labels).
            n (int): Number of features to select. If `n > 0`, the top `n` features are selected
                     based on the absolute values of the coefficients. If `n <= 0`, all features
                     with non-zero coefficients are selected.
            num_batches (int): Number of batches to use for fitting the Lasso model.

        """
        # (B, C, H, W) -> (B*C*H, C, 1, 1)　to perform Lasso regression for the channel dimension
        x, y = retrieve_square_images(1, data[0], gt[0])
        for i in range(1, num_batches):
            tmp_x, tmp_y = retrieve_square_images(1, data[i], gt[i])

            x = np.concatenate([x, tmp_x])
            y = np.concatenate([y, tmp_y])

        b, c, h, w = x.shape

        x = x.reshape(b, -1)
        y = y.reshape(b, -1)

        self.feature_extractor.fit(x, y)
        self.n_channels = self.feature_extractor.n_features_in_

        if n > 0:
            x = self.feature_extractor.coef_
            # 絶対値でソートされたインデックスを取得
            sorted_indices = np.argsort(np.abs(x))[::-1].copy()
            # 上位n個のインデックスを取得
            self.selected_indices = sorted_indices[:n].astype(np.float32)

        else:
            self.selected_indices = np.where(
                self.feature_extractor.coef_ != 0
            )[0].astype(np.float32)

    def get_num_channels(self) -> int:
        if hasattr(self, "n_channels"):
            raise AttributeError(
                "n_channels property does not exist in the class."
            )
        return self.n_channels
