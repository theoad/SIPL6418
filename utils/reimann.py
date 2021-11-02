import torch
from pytorch_lightning import LightningModule
from torch import Tensor
'''
Porting of pyRiemann to pyTorch
Original Author: Alexandre Barachant
https://github.com/alexandrebarachant/pyRiemann
'''


# To avoid numerical errors, torch equivalent of np.around(array, decimals=9)
def around(t: torch.Tensor, decimals=9) -> torch.Tensor:
    return torch.round((t * 10 ** decimals))/10 ** decimals


# Function from the pyRiemann package ported in pytorch
def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    eigvals, eigvects = torch.linalg.eigh(Ci, UPLO='L')
    eigvals = torch.diag(operator(around(eigvals)))
    return torch.matmul(torch.matmul(around(eigvects), around(eigvals)), around(eigvects).T)


def sqrtm(Ci):
    """
    :param Ci: the coavriance matrix
    :returns: the matrix square root
    """
    return _matrix_operator(Ci, torch.sqrt)


def logm(Ci):
    """
    :param Ci: the coavriance matrix
    :returns: the matrix logarithm
    """
    return _matrix_operator(Ci, torch.log)


def expm(Ci):
    """"
    :param Ci: the coavriance matrix
    :returns: the matrix exponential
    """
    return _matrix_operator(Ci, torch.exp)


def invsqrtm(Ci):
    """
    :param Ci: the coavriance matrix
    :returns: the inverse matrix square root
    """
    isqrt = lambda x: 1. / torch.sqrt(x)
    return _matrix_operator(Ci, isqrt)


def riemann_dist(x: Tensor, y: Tensor) -> Tensor:
    """
    pairwise Riemann distance between x and y
    Args:
        x (N x M x M): tensor of PD matrices
        y (N x M x M): tensor of PD matrices
    Return:
        out (1 x N): || log(x^-1 y) ||F.
    """
    eig, _ = torch.lobpcg(x, x.shape[1]//3, y)
    return torch.sum(torch.log(eig) ** 2, dim=1).T


def riemann_dist2(x: Tensor, y: Tensor) -> Tensor:
    """
    pairwise Riemann distance between x and y
    Args:
        x (N x M x M): tensor of PD matrices
        y (N x M x M): tensor of PD matrices
    Return:
        out (1 x N): || log(x) - log(y) ||F.
    """
    return torch.mean((torch.log(x) - torch.log(y)) ** 2, dim=(1, 2))


def riemann_dist3(x: Tensor, y: Tensor) -> Tensor:
    """
    pairwise Riemann distance between x and y
    Args:
        x (N x M x M): tensor of PD matrices
        y (N x M x M): tensor of PD matrices
    Return:
        out (1 x N): || x^-1/2 y x^-1/2 ||F.
    """
    return torch.stack([torch.sum(((invsqrtm(s1).mm(s2)).mm(invsqrtm(s1))) ** 2) for s1, s2 in zip(x, y)])


class RiemannianMeanEmbedding(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.m = None

    def fit(self, covmats: Tensor) -> None:
        self.m = invsqrtm(self._mean_riemann(covmats))

    def _get_sample_weight(self, sample_weight: Tensor, data: Tensor) -> Tensor:
        """
        Get the sample weights.
        If none provided, weights init to 1. otherwise, weights are normalized.
        """
        if sample_weight is None:
            sample_weight = torch.ones(data.shape[0], device=self.device)

        if len(sample_weight) != data.shape[0]:
            raise ValueError("len of sample_weight must be equal to len of data.")

        sample_weight /= torch.sum(sample_weight)
        return sample_weight

    def _mean_riemann(self, covmats: Tensor, tol=10e-7, maxiter=25, init=None, sample_weight=None) -> Tensor:
        """
        Return the mean covariance matrix according to the Riemannian metric.
        The procedure is similar to a gradient descent minimizing the sum of
        riemannian distance to the mean.
        :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
        :param tol: the tolerance to stop the gradient descent
        :param maxiter: The maximum number of iteration, default 50
        :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
        :param sample_weight: the weight of each sample
        :returns: the mean covariance matrix
        """
        # init
        sample_weight = self._get_sample_weight(sample_weight, covmats)
        Nt, Ne, Ne = covmats.shape
        if init is None:
            C = torch.mean(covmats, dim=0)
        else:
            C = init
        k = 0
        nu = 1.0
        tau = crit = torch.finfo(torch.float32).max
        # stop when J<10^-9 or max iteration = 50
        while (crit > tol) and (k < maxiter) and (nu > tol):
            k += 1
            C12 = sqrtm(C)
            Cm12 = invsqrtm(C)
            J = torch.zeros((Ne, Ne), device=covmats.device)

            for index in range(Nt):
                tmp = torch.matmul(torch.matmul(Cm12, covmats[index, :, :]), Cm12)
                J += sample_weight[index] * logm(tmp)

            crit = torch.norm(J)
            h = nu * crit
            C = torch.matmul(torch.matmul(C12, expm(nu * J)), C12)
            if h < tau:
                nu = 0.95 * nu
                tau = h
            else:
                nu = 0.5 * nu
        return C

    def forward(self, x: Tensor) -> Tensor:
        x = self.m @ torch.log(x) @ self.m
        # Trick to vectorized upper triangle of each matrix in batch
        return x[:, torch.triu(torch.ones(x.shape[1], x.shape[2])) == 1]
