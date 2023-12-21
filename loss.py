import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import grey_opening

from utils import gaussian_kernel_3d  # Assuming you have a 3D gaussian kernel function

class NCutLoss3D(nn.Module):
    def __init__(self, radius: int = 4, sigma_1: float = 5, sigma_2: float = 1):
        super(NCutLoss3D, self).__init__()
        self.radius = radius
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def forward(self, labels: Tensor, inputs: Tensor) -> Tensor:
        num_classes = labels.shape[1]
        kernel = gaussian_kernel_3d(radius=self.radius, sigma=self.sigma_1, device=labels.device.type)
        loss = 0

        for k in range(num_classes):
            class_probs = labels[:, k].unsqueeze(1)
            class_mean = torch.mean(inputs * class_probs, dim=(2, 3, 4), keepdim=True) / \
                torch.add(torch.mean(class_probs, dim=(2, 3, 4), keepdim=True), 1e-5)
            diff = (inputs - class_mean).pow(2).sum(dim=1).unsqueeze(1)

            weights = torch.exp(diff.pow(2).mul(-1 / self.sigma_2 ** 2))

            # Use 3D convolutions for spatial filtering
            numerator = torch.sum(class_probs * F.conv3d(class_probs * weights, kernel, padding=self.radius, stride=1))
            denominator = torch.sum(class_probs * F.conv3d(weights, kernel, padding=self.radius, stride=1))
            loss += nn.L1Loss()(numerator / torch.add(denominator, 1e-6), torch.zeros_like(numerator))

        return num_classes - loss

class OpeningLoss3D(nn.Module):
    def __init__(self, radius: int = 2):
        super(OpeningLoss3D, self).__init__()
        self.radius = radius

    def forward(self, labels: Tensor, *args) -> Tensor:
        smooth_labels = labels.clone().detach().cpu().numpy()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                smooth_labels[i, j] = grey_opening(smooth_labels[i, j], self.radius)

        smooth_labels = torch.from_numpy(smooth_labels.astype(np.float32))
        if labels.device.type == 'cuda':
            smooth_labels = smooth_labels.cuda()

        return nn.MSELoss()(labels, smooth_labels.detach())

