import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import initialize_weights


class PatchSampleF(nn.Module):
    """Patch sampler and optional Expander projection network used by CUT."""

    def __init__(self, use_expander=True, nc=256):
        super().__init__()
        self.use_expander = use_expander
        self.nc = nc
        self.expanders = nn.ModuleList()
        self._expander_initialized = False

    def _create_expanders(self, features):
        self.expanders = nn.ModuleList()
        for feature in features:
            input_channels = feature.shape[1]
            self.expanders.append(nn.Sequential(
                nn.Linear(input_channels, self.nc),
                nn.ReLU(inplace=True),
                nn.Linear(self.nc, self.nc),
            ).to(feature.device))
        self._expander_initialized = True

    def initialize(self, features, init_type='xavier', init_gain=0.02):
        if self.use_expander and not self._expander_initialized:
            self._create_expanders(features)
            initialize_weights(self, init_type, init_gain)

    def forward(self, features, num_patches=256, patch_ids=None,
                return_projected=False):
        if self.use_expander and not self._expander_initialized:
            self._create_expanders(features)
        sampled_features = []
        sampled_ids = []
        projected_features = []
        for index, feature in enumerate(features):
            batch_size, _, height, width = feature.shape
            flattened = feature.permute(0, 2, 3, 1).reshape(
                batch_size, height * width, -1)
            if num_patches > 0:
                if patch_ids is None:
                    ids = np.random.permutation(height * width)[:num_patches]
                    ids = torch.as_tensor(ids, device=feature.device)
                else:
                    ids = patch_ids[index].to(feature.device)
                sampled = flattened[:, ids, :].reshape(-1, flattened.shape[-1])
            else:
                ids = torch.arange(height * width, device=feature.device)
                sampled = flattened.reshape(-1, flattened.shape[-1])
            if self.use_expander:
                sampled = self.expanders[index](sampled)
            if return_projected:
                projected_features.append(sampled)
            sampled_features.append(F.normalize(sampled, dim=1))
            sampled_ids.append(ids)
        if return_projected:
            return sampled_features, sampled_ids, projected_features
        return sampled_features, sampled_ids