# Riemann Inception Distance
from typing import Any, Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torchmetrics.metric import Metric
import re
import torch.nn as nn
from utils.utils import PositiveDefiniteRectifier


class RID(Metric):
    """
    FID = Frechet inception distance
    RID = Reimannian inception distance
    """
    def __init__(self,
                 scoring_method: nn.Module,
                 channel_wise_covariance: Optional[bool] = True,
                 embedding_kernel: Optional[nn.Module] = None,
                 compute_on_step: Optional[bool] = False,
                 dist_sync_on_step: Optional[bool] = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Optional[Callable] = None
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.scoring = scoring_method
        self.channel_wise_covariance = channel_wise_covariance
        self.embedding_kernel = embedding_kernel
        self.PSDrectif = PositiveDefiniteRectifier(high=1)
        self.real_prepared = False
        self.add_state("score", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("ref", [], dist_reduce_fx=None)
        self.add_state("num_obs", torch.zeros(1), dist_reduce_fx="sum")

    @staticmethod
    def method_name(method: nn.Module):
        return re.sub('[ ,=()]', '_', method.__repr__())

    def _setup(self) -> Tensor:
        """ Internal method to store and prepare the reference data.
        Returns: Score of the reference data
        """
        if not self.real_prepared:
            raise Exception("reference data not updated/computed.")
        data = torch.cat(self.ref, dim=0)
        return self.scoring.fit(data)

    def update(self, embedding: Tensor, real: bool) -> None:  # type: ignore
        """ Update the state with extracted features
        Args:
            embedding: tensor with images fed to the feature extractor
            real: is this the reference data ?
        """
        # print(embedding.shape)  # [100, 2048, 8, 8]
        if self.embedding_kernel:
            covariance = self.embedding_kernel(embedding, channel_wise=self.channel_wise_covariance)
        else:
            if len(embedding.shape) > 3:
                if not self.channel_wise_covariance:
                    # B, C, H, W --> B, H, W, C  # [100, 8, 8, 2048]
                    embedding = embedding.permute(0, 2, 3, 1).contiguous()

                    # B, H, W, C --> B, H * W, C  # [100, 64, 2048] ~ each element in the batch is a matrix with 64 vectors
                    embedding = embedding.view(embedding.shape[0], embedding.shape[1] * embedding.shape[2], -1)
                else:
                    embedding = embedding.view(embedding.shape[0], embedding.shape[1], -1)
            # First substract the mean (Qcov = (X - mu) * (X - mu)_T: https://en.wikipedia.org/wiki/Sample_mean_and_covariance)
            embedding -= embedding.mean(0).mean(dim=1).unsqueeze(1)
            # [100, 64, 64] each element in the batch is a covariance matrix
            covariance = embedding @ torch.transpose(embedding, 1, 2)
        for i in range(covariance.shape[0]):
            covariance[i] = self.PSDrectif(covariance[i])  # make sure covariance is PD
        if real:
            # If real, simply store the covariance matrices as reference
            self.ref.append(covariance)
        else:
            # If not real, get the score for each matrix and aggregate
            self.score += torch.sum(self.scoring(covariance))
            self.num_obs += embedding.shape[0]

    def compute(self) -> Tensor:
        """ Method to call after calling update on each batch of a given dataset
        Returns: The score of the dataset. If the real data has not yet been prepared,
        then the method prepares the real data and returns the score of the real data.
        """
        if not self.real_prepared:
            self.real_prepared = True
            self._persistent['ref'] = True
            return self._setup()
        self.reset()
        return self.score / self.num_obs
