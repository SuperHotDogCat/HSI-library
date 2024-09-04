"""
code from https://github.com/spaceml-org/RaVAEn
"""

from abc import abstractmethod
from typing import Any, Dict, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, batch: Tensor, *inputs: Any, **kwargs) -> Tensor:
        raise NotImplementedError


class BaseAE(BaseModel):
    def __init__(self, visualisation_channels):
        super().__init__()

        self.visualisation_channels = visualisation_channels

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        z = self.encode(torch.nan_to_num(input))
        return self.decode(z)

    def loss_function(
        self, input: Tensor, results: Dict, mask_invalid: bool = True, **kwargs
    ) -> Dict:

        if not mask_invalid:
            recons_loss = F.mse_loss(results, torch.nan_to_num(input))
        else:
            invalid_mask = torch.isnan(input)
            recons_loss = F.mse_loss(
                results[~invalid_mask], input[~invalid_mask]
            )

        return {
            "loss": recons_loss,
            "Reconstruction_Loss": recons_loss,
        }

    def _visualise_step(self, batch):
        result = self.forward(batch)
        rec_error = (batch - result).abs()

        return (
            batch[:, self.visualisation_channels],
            result[:, self.visualisation_channels],
            rec_error.max(1)[0],
        )

    @property
    def _visualisation_labels(self):
        return ["Input", "Reconstruction", "Rec error"]


class BaseVAE(BaseAE):
    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def _visualise_step(self, batch):
        result = self.forward(batch)  # [reconstruction, mu, log_var]
        # Just select the reconstruction
        result = result[0]
        rec_error = (batch - result).abs()

        return (
            batch[:, self.visualisation_channels],
            result[:, self.visualisation_channels],
            rec_error.max(1)[0],
        )
