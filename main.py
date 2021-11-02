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

METRIC = 'fid'  # 'rid', 'fid'
MODEL = 'inception'  # 'inception', 'clip'
CHANNELS = 2048  # 64, 192, 768, 2048
SCORING = 'knn'  # 'svm', 'knn'; ignored if METRIC == 'fid'

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = None, None
if MODEL == 'inception':
    preprocess = Compose([ToTensor(), Lambda(lambda x: (255 * (x - torch.min(x)) / torch.max(x)).type(torch.uint8))])
    preprocess_degradation = Compose([ToTensor(), Lambda(lambda x: x + torch.randn_like(x) * 10), Lambda(lambda x: (255 * (x - torch.min(x)) / torch.max(x)).type(torch.uint8))])
    if METRIC == 'rid':
        model = NoTrainInceptionV3NoAvg(name='inception-v3-compat', features_list=[str(CHANNELS)]).to(device)
    elif METRIC == 'fid':
        model = NoTrainInceptionV3(name='inception-v3-compat', features_list=[str(CHANNELS)]).to(device)
elif MODEL == 'clip':
    CHANNELS = 512
    model, preprocess = clip.load('ViT-B/32', device)  # TODO: try with clip network, how to deal with 1-d latent vectors
    preprocess_degradation = Compose([preprocess, Lambda(lambda x: x + torch.randn_like(x) * 10)])
else:
    raise ValueError(f"MODEL = '{MODEL}' not supported. Set MODEL = 'inception' or 'clip'.")

# Initialize the metric
if METRIC == 'fid':
    metric = FID(CHANNELS).to(device)
elif METRIC == 'rid':
    scoring = OneSVM() if SCORING == 'svm' else KNN(dist=riemann_dist3)
    metric = RID(scoring, channel_wise_covariance=False, embedding_kernel=GaussianKernel()).to(device)  # TODO: try using embedding_kernel=GaussianKernel()
else:
    raise ValueError(f"METRIC = '{METRIC}' not supported. Set METRIC = 'rid' or 'fid'.")

# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform=preprocess)
test = CIFAR100(root, download=True, train=False, transform=preprocess)

with torch.no_grad():
    for images, labels in tqdm(DataLoader(train, batch_size=100)):
        features = model.encode_image(images.to(device))
        metric.update(features, True)

train_score = metric.compute()
print(f"Score of the train dataset: {train_score.item():.3f}")

# Compute the score for the test dataset (different images but same distribution)
with torch.no_grad():
    for images, labels in tqdm(DataLoader(test, batch_size=100)):
        features = model.encode_image(images.to(device))
        metric.update(features, False)

test_score = metric.compute()
print(f"Score of the test dataset: {test_score.item():.3f}")


# Compute the score for the test dataset augmented with some degredation
test = CIFAR100(root, download=True, train=False, transform=preprocess_degradation)
with torch.no_grad():
    for images, labels in tqdm(DataLoader(test, batch_size=100)):
        features = model.encode_image(images.to(device))
        metric.update(features, False)

augmentation_score = metric.compute()
print(f"Score of the destroyed images: {augmentation_score.item():.3f}")
