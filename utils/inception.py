from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from typing import List, Optional, Union, Tuple
from torch import Tensor
import torch
from torch_fidelity.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x


class NoTrainInceptionV3NoAvg(FeatureExtractorInceptionV3):
    """
    Code duplication from original FeatureExtractorInceptionV3 to extract features before AvgPool layers.
    """
    def __init__(
        self,
        name: str,
        features_list: List[str],
        feature_extractor_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__(name, features_list, feature_extractor_weights_path)
        # put into evaluation mode
        self.eval()

    def train(self, mode: bool):
        """ the inception network should not be able to be switched away from evaluation mode """
        return super().train(False)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        assert torch.is_tensor(x) and x.dtype == torch.uint8, 'Expecting image as torch.Tensor with dtype=torch.uint8'
        features = {}
        remaining_features = self.features_list.copy()

        x = x.float()
        # N x 3 x ? x ?

        x = interpolate_bilinear_2d_like_tensorflow1x(
            x,
            size=(self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE),
            align_corners=False,
        )
        # N x 3 x 299 x 299

        # x = (x - 128) * torch.tensor(0.0078125, dtype=torch.float32, device=x.device)  # really happening in graph
        x = (x - 128) / 128  # but this gives bit-exact output _of this step_ too
        # N x 3 x 299 x 299

        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.MaxPool_1(x)
        # N x 64 x 73 x 73

        if '64' in remaining_features:
            features['64'] = x
            remaining_features.remove('64')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.MaxPool_2(x)
        # N x 192 x 35 x 35

        if '192' in remaining_features:
            features['192'] = x
            remaining_features.remove('192')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        if '768' in remaining_features:
            features['768'] = x
            remaining_features.remove('768')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        if '2048' in remaining_features:
            features['2048'] = x
            remaining_features.remove('2048')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        x = self.AvgPool(x)
        # N x 2048 x 1 x 1

        x = torch.flatten(x, 1)
        # N x 2048

        if 'logits_unbiased' in remaining_features:
            x = x.mm(self.fc.weight.T)
            # N x 1008 (num_classes)
            features['logits_unbiased'] = x
            remaining_features.remove('logits_unbiased')
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)
            # N x 1008 (num_classes)

        features['logits'] = x
        return tuple(features[a] for a in self.features_list)

    def encode_image(self, img: Tensor):
        return self(img)[0]
