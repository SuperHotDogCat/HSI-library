from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseFeatureExtractor(ABC):
    def __init__(self):
        """
        BaseFeatureExtractor: Base Class that transforms high dimensional HSI data to lower dimensional HSI data and has a transform method.
                           The transform method receives np.array: (Batch, Channels_before, H_before, W_before) and returns np.array: (Batch, Channels_after, H_after, W_after) where C_low < C_high.
                           However, in some methods, for example PCA, the shape of the return array is not (Batch, Channels_after, H_after, W_after), but (Batch, Features).
        Args:
            None Base class has no arguments to be passed.
        Returns:
            None
        """

    def __call__(self, x: np.array) -> np.array:
        """
        __call__ method is a wrapper for the transform method.
        Args:
            x: HSI Data, its shape is (Batch, Channels_before, H_before, W_before).
        Returns:
            processed HSI Data, its shape is (Batch, Channels_after, H_after, W_after). Make sure that Channels_before > Channels_after.
        """
        return self.transform(x)

    @abstractmethod
    def transform(self, x: np.array) -> np.array:
        """
        transform HSI Data into the one that has reduced channels
        Args:
            x: HSI Data, its shape is (Batch, Channels_before, H_before, W_before)
        Returns:
            processed HSI Data, its shape is (Batch, Channels_after, H_after, W_after). Make sure that Channels_before > Channels_after.
            However, in some methods, for example PCA, the shape of the return array is not (Batch, Channels_after, H_after, W_after), but (Batch, Features).
        """
        pass

    def fit(self, *args) -> None:
        """
        Optional method. Developers should implement the method if necessary.
        Args:
            args: Variable length arguments. You can add anything you think it necessary for the fit method.
        Returns:
            None
        """
        pass
