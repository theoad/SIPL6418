from typing import Any, Callable, List, Optional, Tuple
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import numpy as np
import torch
from torch import Tensor
from torch.autograd import Function
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


class NoTrainInceptionV3(FeatureExtractorInceptionV3):

    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool) -> 'NoTrainInceptionV3':
        """ the inception network should not be able to be switched away from evaluation mode """
        return super().train(False)

    def encode_image(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out[0].reshape(x.shape[0], -1)


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    All credit to:
        https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        import scipy

        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        import scipy
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


def _compute_fid(
    mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    r"""
    Adjusted version of
        https://github.com/photosynthesis-team/piq/blob/master/piq/fid.py
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2

    covmean = sqrtm(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        rank_zero_info(
            f'FID calculation produces singular product; adding {eps} to diagonal of '
            'covaraince estimates'
        )
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
        covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FID(Metric):
    def __init__(
        self,
        feature_size: Optional[int],
        compute_on_step: Optional[bool] = False,
        dist_sync_on_step: Optional[bool] = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("real_sum", torch.zeros(feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("real_correlation", torch.zeros(feature_size, feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("fake_sum", torch.zeros(feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("fake_correlation", torch.zeros(feature_size, feature_size, dtype=torch.double), dist_reduce_fx="sum")
        self.add_state("num_real_obs", torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("num_fake_obs", torch.zeros(1), dist_reduce_fx="sum")
        self.real_mean, self.real_cov = None, None
        self.real_prepared = False

    def update(self, features: Tensor, real: bool) -> None:  # type: ignore
        """ Update the state with extracted features
        Args:
            features: tensor with images fed to the feature extractor
            real: bool indicating if features belong to the real or the fake distribution
        """
        if self.real_prepared and real:
            pass
        features = features.double().view(features.shape[0], -1)
        sum_features = features.sum(dim=0)
        correlation = features.t().mm(features)
        if real:
            self.real_sum += sum_features
            self.real_correlation += correlation
            self.num_real_obs += features.shape[0]
        else:
            self.fake_sum += sum_features
            self.fake_correlation += correlation
            self.num_fake_obs += features.shape[0]

    def compute(self, update_real_only: bool = False) -> Tensor:
        """ Calculate FID score based on accumulated extracted features from the two distributions """
        if not self.real_prepared:
            self.real_mean = self.real_sum / self.num_real_obs
            self.real_cov = (1.0 / (self.num_real_obs - 1.0)) * self.real_correlation - (self.num_real_obs / (self.num_real_obs - 1.0)) * self.real_mean.unsqueeze(1).mm(self.real_mean.unsqueeze(0))
            self.real_prepared = True
            self._persistent['real_mean'] = True
            self._persistent['real_cov'] = True
            return _compute_fid(self.real_mean, self.real_cov, self.real_mean, self.real_cov)
        if update_real_only:
            return None
        fake_mean = self.fake_sum / self.num_fake_obs
        fake_cov = (1.0 / (self.num_fake_obs - 1.0)) * self.fake_correlation - (self.num_fake_obs / (self.num_fake_obs - 1.0)) * fake_mean.unsqueeze(1).mm(fake_mean.unsqueeze(0))
        # compute fid
        return _compute_fid(self.real_mean, self.real_cov, fake_mean, fake_cov)



