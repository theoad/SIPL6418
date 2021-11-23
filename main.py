import os
import torch
import clip
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from rid import RID
from utils.utils import KNN, OneSVM, GaussianKernel, DestroyNaturality, PILtoNumpy
from utils.reimann import riemann_dist2, riemann_dist3
from utils.inception import NoTrainInceptionV3NoAvg
from torchvision.transforms import ToTensor, Compose, Lambda
from fid import FID, NoTrainInceptionV3
from typing import Optional

METRIC = 'fid'  # 'rid', 'fid'
MODEL = 'inception'  # 'inception', 'clip'
CHANNELS = 2048  # 64, 192, 768, 2048
SCORING = 'knn'  # 'svm', 'knn'; ignored if METRIC == 'fid'


def to256(x: torch.Tensor) -> torch.Tensor:
    """
    Takes a [0,1] image and returns the same image normalized to [0, 255]
    Args:
        x: input image as a pytorch tensor stored in the range [0, 1]

    Returns: x stored in the range [0, 255]
    """
    # TODO: figure this out (with or without min/max)
    return (255 * (x - torch.min(x)) / torch.max(x)).type(torch.uint8)


def main(metric: Optional[str] = 'fid',  # 'rid', 'fid'
         model_name: Optional[str] = 'inception',  # 'inception', 'clip'
         channels: Optional[int] = 2048,  # 64, 192, 768, 2048
         scoring: Optional[str] = 'knn'  # 'svm', 'knn'; ignored if METRIC == 'fid'
         ):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = None, None
    if model_name == 'inception':
        preprocess = Compose([ToTensor(), Lambda(to256)])
        preprocess_degradation = Compose([ToTensor(), Lambda(lambda x: x + torch.randn_like(x) * 10), Lambda(to256)])
        if metric == 'rid':
            model = NoTrainInceptionV3NoAvg(name='inception-v3-compat', features_list=[str(channels)]).to(device)
        elif metric == 'fid':
            model = NoTrainInceptionV3(name='inception-v3-compat', features_list=[str(channels)]).to(device)
    elif model_name == 'clip':
        channels = 512
        model, preprocess = clip.load('ViT-B/32', device)  # TODO: try with clip network, how to deal with 1-d latent vectors
        preprocess_degradation = Compose([preprocess, Lambda(lambda x: x + torch.randn_like(x) * 10)])
    else:
        raise ValueError(f"MODEL = '{model_name}' not supported. Set MODEL = 'inception' or 'clip'.")

    # Initialize the metric
    if metric == 'fid':
        metric = FID(channels).to(device)
    elif metric == 'rid':
        scoring = OneSVM() if scoring == 'svm' else KNN(dist=riemann_dist3)
        metric = RID(scoring, channel_wise_covariance=False, embedding_kernel=GaussianKernel()).to(device)  # TODO: try using embedding_kernel=GaussianKernel()
    else:
        raise ValueError(f"METRIC = '{metric}' not supported. Set METRIC = 'rid' or 'fid'.")

    # Load the dataset
    root = os.path.expanduser("data/")
    train = CIFAR100(root, download=True, train=True, transform=preprocess)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    with torch.no_grad():
        for images, _ in tqdm(DataLoader(train, batch_size=100), leave=False, desc='train dataset scoring'):
            features = model.encode_image(images.to(device))
            metric.update(features, True)

    train_score = metric.compute()
    print(f"Score of the train dataset: {train_score.item():.3f}")

    # Compute the score for the test dataset (different images but same distribution)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(test, batch_size=100), leave=False, desc='test dataset'):
            features = model.encode_image(images.to(device))
            metric.update(features, False)

    test_score = metric.compute()
    print(f"Score of the test dataset: {test_score.item():.3f}")

    # Compute the score for the test dataset augmented with some degradation
    test = CIFAR100(root, download=False, train=False, transform=preprocess_degradation)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(test, batch_size=100), leave=False, desc='degraded test dataset'):
            features = model.encode_image(images.to(device))
            metric.update(features, False)

    augmentation_score = metric.compute()
    print(f"Score of the destroyed images: {augmentation_score.item():.3f}")


if __name__ == "__main__":
    main(METRIC, MODEL, CHANNELS, SCORING)
