from typing import Optional
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
import numpy as np
from typing import Callable
from torchmetrics.utilities import rank_zero_warn
from torch.distributions import constraints
from sklearn.svm import OneClassSVM
from utils.reimann import RiemannianMeanEmbedding, riemann_dist
from imgaug import augmenters as iaa


class PositiveDefiniteRectifier(torch.nn.Module):
    """
        Adds small values on the input matrix diagonal to ensure it's PSD according to
    torch.distributions constraints. The smallest value is added to the diagonal
    (search is performed on a logarithmic scale)
    Args:
        low:
            Search lower-bound
        high:
            Search upper-bound
        step:
            Number of logarithmic step in the search
    """
    def __init__(self,
                 minimal_increment: Optional[bool] = False,
                 low: Optional[float] = 1e-6,
                 high: Optional[float] = 10,
                 step: Optional[int] = 100):
        super().__init__()
        self.minimal_increment = minimal_increment
        self.low = low
        self.high = high
        self.step = step

    def forward(self, mat):
        """
        Args:
            mat: SPSD matrix

        Returns: PSD matrix
        """
        if not self.minimal_increment:
            return mat + torch.eye(*mat.size(), out=torch.empty_like(mat)) * self.low
        for i in range(mat.shape[0]):
            if not constraints.positive_definite.check(mat):
                # Numerical trick to deal with SPD matrices
                for idx, reg in enumerate(np.exp(np.linspace(np.log(self.low), np.log(self.high), self.step))):
                    mat += torch.eye(*mat.size(), out=torch.empty_like(mat)) * reg
                    if constraints.positive_definite.check(mat):
                        break
                    if idx == self.step:
                        raise ValueError("SPSD Matrix")
        return mat


class GaussianKernel(LightningModule):
    """
        Gaussian Kernel applied to a series of matrices.
    Args:
        sigma:
            A single float controlling the standard deviation of the RBF kernel
        p:
            the Euclidean norm power (for L2 norm use p=2)
    Based on:
            https://github.com/Davidbens/ICASSP2020/blob/master/algorithm.py
    """
    def __init__(self,
                 sigma: Optional[float] = 1,
                 p: Optional[float] = 2.):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma
        self.p = p

    def forward(self, x: Tensor, channel_wise=True) -> Tensor:
        """
            For each tensor Ti(C x H x W) (0<=i<B) in the batch,
        Dissimilarity(Ti) i,j = || T[i].flat - T[j].flat ||_p
        GK(Ti) = exp(-Dissimilarity(Ti)/ (median(Dissimilarity) * sigma)^2)
        Args:
            x: tensor of feature maps (dim B x C x H x W)

        Returns:
            tensor of PD matrices (dim B x C x C)
        """
        if type(x) is not Tensor:
            x = x[0]
        B, C, H, W = x.shape
        if channel_wise:
            x = x.view(B, C, H * W)
        else:
            x = x.permute(0, 2, 3, 1).view(B, H * W, C)
        dissimilarity = torch.cdist(x, x, p=self.p, compute_mode='use_mm_for_euclid_dist')
        epsilon = torch.median(dissimilarity)
        gk = torch.exp(-dissimilarity / (epsilon * self.sigma) ** 2)
        return gk


class KNN(LightningModule):
    """
        K-nearest neighbors module. Stores a given dataset in self.data. May have a significant memory footprint.
    The module is implemented according to torchmetrics' Metric abstract class (from Pytorch lightning) in order
    to support distributed GPU updates (reduction across GPUs is performed automatically).
    """
    def __init__(self,
                 dist: Optional[Callable] = riemann_dist,
                 k: Optional[int] = 10,
                 reduction: Optional[str] = 'max') -> None:
        super().__init__()
        rank_zero_warn(
            "Module `KNN` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )
        supported_reductions = ['max', 'mean']
        if reduction not in supported_reductions:
            raise ValueError("reduction should be one of: 'max', 'mean'")
        self.k = k
        self.reduction = reduction
        self.dist = dist
        self.data = None

    def score(self, dissimilarity):
        k_dist, k_idx = torch.topk(dissimilarity, self.k, largest=False)
        if self.reduction == 'max':
            return torch.max(k_dist)
        elif self.reduction == 'mean':
            return torch.mean(k_dist)

    def fit(self, data: Tensor):
        self.data = data
        return self.ref_score()

    def ref_score(self):
        """
        Returns: Score of the reference data
        """
        dissimilarity = self.dist(self.data, self.data)
        return self.score(dissimilarity)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features (B x M x M): batch of PD matrices to compare to the reference data
        Return:
            dist (B x K): For each matrix in batch, the smallest-k distances to the ref matrices
                          and the corresponding indices of the k closest matrices.
        """
        return torch.stack([self.score(self.dist(self.data, f.repeat(self.data.shape[0], 1, 1))) for f in features])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k}, reduction={self.reduction})'


class OneSVM(LightningModule):
    """
        Pytorch Porting of sklearn's oneSVM.
    """
    def __init__(self, **svm_kwargs) -> None:
        super().__init__()
        self.svm_kwargs = svm_kwargs
        self.svm = OneClassSVM(**svm_kwargs)
        self.euclidean_embedding = RiemannianMeanEmbedding()

    def fit(self, data: Tensor):
        self.euclidean_embedding.fit(data)
        self.svm.fit(self.euclidean_embedding(data).cpu())
        return self(data)

    def forward(self, features: Tensor) -> Tensor:
        return torch.from_numpy(self.svm.score_samples(self.euclidean_embedding(features).cpu())).to(device=features.device)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.svm_kwargs})'


class DestroyNaturality(torch.nn.Module):
    def __init__(self):
        super().__init__()
        children = [
            iaa.ChannelShuffle(p=0.5),
            iaa.AdditiveGaussianNoise(scale=(3, 10), per_channel=True),
            iaa.Cutout(nb_iterations=(1, 2), size=(0.01, 0.05), squared=False, fill_mode=['constant', 'gaussian'],
                       cval=(0, 255), fill_per_channel=True),
            iaa.Dropout(p=(0.001, 0.01), per_channel=True),
            iaa.SaltAndPepper(p=(0.001, 0.01), per_channel=True),
            iaa.CoarseSaltAndPepper(p=(0.001, 0.01),
                                    size_percent=(0.05, 0.4)),
            iaa.Solarize(0.2, per_channel=True),
            iaa.JpegCompression(compression=(5, 15)),
            iaa.Invert(p=0.5, per_channel=True),
            iaa.AdditiveLaplaceNoise(scale=(3, 15), per_channel=True),
            iaa.AdditivePoissonNoise(lam=(0.02, 1.0), per_channel=True),
            iaa.GaussianBlur(sigma=(0.1, 1.0)),
            iaa.AverageBlur(k=(1, 2)),
            iaa.MedianBlur(k=(1, 3)),
            iaa.BilateralBlur(d=(1, 2)),
            iaa.MotionBlur(k=(3, 4)),
            iaa.MeanShiftBlur(),
            iaa.RandAugment(n=(1, 4), m=(6, 12)),
            iaa.ElasticTransformation(alpha=(1.0, 30.0), sigma=(1.0, 6.0))
        ]
        self.aug = iaa.Sequential([
            iaa.SomeOf(
                n=(1, 3),
                children=children,
                random_order=True
            )
        ])

    def forward(self, x):
        return self.aug.augment_images(x)


class PILtoNumpy:
    def __call__(self, pil_img):
        return np.asarray(pil_img)
