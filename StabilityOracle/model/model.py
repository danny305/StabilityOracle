from typing import List, Literal
import torch
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from timm.models.vision_transformer import Block as AttentionBlock
import torch.nn as nn
import torch.nn.functional as F
from timm.utils import ModelEma

from StabilityOracle.model import Backbone, Transformer, Mlp


class SiameseGraphormer(nn.Module):
    def __init__(
        self, args, hidden_channel: int = 128, num_class: int = 1, aa_dim: int = 128
    ):
        super().__init__()

        base_model = Backbone

        self.backbone = base_model(
            depth=args.depth,
            _DEFAULT_V_DIM=(aa_dim, 0),
            num_rbf=4,
            embedding_size=aa_dim,
            drop_rate=0,
            atom_drop_rate=0,
            num_dist_mask=1,
            edge_encoding=False,
            use_physical_properties_input=True,
            mask_closest_atom=False,
            dist_thres=8,
            args=args,
        )

        self.aa_dim = aa_dim
        backbone_dim = aa_dim * len(args.return_idx)
        self.max_atom = 513 * 1

        proj0_dim = backbone_dim
        if not args.clstoken_unify and self.aa_dim == backbone_dim:
            self.proj_clstoken = nn.Identity()
        else:
            self.proj_clstoken = nn.Linear(self.aa_dim, backbone_dim)
        init_values = None
        num_heads = 8

        self.regression_attn = nn.Sequential(
            AttentionBlock(dim=proj0_dim, num_heads=num_heads, init_values=init_values),
            AttentionBlock(dim=proj0_dim, num_heads=num_heads, init_values=init_values),
        )
        self.proj1 = nn.Identity()
        head_dim = proj0_dim

        self.head = nn.Sequential(
            nn.Linear(head_dim * 1, hidden_channel // 2),
            nn.BatchNorm1d(hidden_channel // 2),
            nn.SiLU(),
            nn.Linear(hidden_channel // 2, num_class),
        )

        self.args = args

        if args.freeze_backbone:
            self.backbone.requires_grad_(False)

        self.add_aa = nn.Parameter(torch.randn([1, self.aa_dim]) * 0.01)

    def forward(self, feats, atom_types, pp, coords, ca, mask, aa_feats):

        aa_embeds = self.backbone.dense[-1].weight
        aa_embeds = torch.cat([aa_embeds, self.add_aa], dim=0)

        local_env_feats, out = self.backbone(
            feats=feats,
            atom_types=atom_types, 
            pp=pp,
            coords=coords,
            ca=ca,
            mask=mask,
            return_idx=self.args.return_idx,
        )

        from_feats = local_env_feats
        to_feats = local_env_feats
        bs, num_atom, _ = from_feats.shape

        hidden = from_feats.shape[-1]
        aa_embeds = self.proj_clstoken(aa_embeds)
    
        from_aa = aa_embeds[aa_feats[:, 0].long()].reshape(-1, 1, hidden)
        to_aa = aa_embeds[aa_feats[:, 1].long()].reshape(-1, 1, hidden)

        bs = from_feats.shape[0]
        mutations = from_aa.shape[0] // bs
        hidden = from_feats.shape[-1]

        from_feats = (
            from_feats.unsqueeze(dim=1)
            .repeat(1, mutations, 1, 1)
            .reshape(mutations * bs, -1, hidden)
        )
        from_feats = torch.cat([from_aa, from_feats[:, 1:, :]], dim=1)
        to_feats = torch.cat([to_aa, from_feats[:, 1:, :]], dim=1)

        from_feats = self.proj1(self.regression_attn(from_feats).permute(0, 2, 1)).permute(
            0, 2, 1
        )
        to_feats = self.proj1(self.regression_attn(to_feats).permute(0, 2, 1)).permute(0, 2, 1)

        feat_delta = (to_feats - from_feats)[:, :1, :].mean(dim=1)

        phenotype_delta = self.head(F.dropout(feat_delta, 0.0))

        return 2 * phenotype_delta.reshape(-1), None
