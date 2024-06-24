from functools import partial

import torch, random, scipy, math
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from timm.models.vision_transformer import LayerScale
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath


_NUM_ATOM_TYPES = 9
_element_mapping = lambda x: {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "S": 5,
    "P": 6,
    "F": 7,
    "Cl": 7,
    "CL": 7,
    "Br": 7,
    "BR": 7,
    "I": 7,
}.get(x, 8)
_amino_acids = lambda x: {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLU": 5,
    "GLN": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}.get(x, 20)


class BottleneckAdapter(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = [True, True]
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        talk_head=False,
        use_attention_pool=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.talk_head = talk_head

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        if self.talk_head:
            self.proj_pre = nn.Linear(num_heads, num_heads, bias=False)
            self.proj_post = nn.Linear(num_heads, num_heads, bias=False)

    def forward(self, x, attn_mask=None, attn_bias=None, attn_pool_mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn += attn_mask.unsqueeze(dim=1)
        if attn_bias is not None:
            attn += attn_bias
        if self.talk_head:
            attn = self.proj_pre(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        if attn_pool_mask is not None:
            attn *= attn_pool_mask.unsqueeze(dim=2)

        if self.talk_head:
            attn = self.proj_post(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        talk_head=False,
        num_rbf=16,
        use_attention_pool=False,
        use_attn_bias=True,
        select_thres=128,
        num_dist_mask=1,
        args=None,
    ):
        super().__init__()

        self.args = args
        self.num_dist_mask = num_dist_mask
        self.use_attention_pool = use_attention_pool

        self.norm0 = norm_layer(dim)
        self.attn0 = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            talk_head=talk_head,
        )
        self.ls0 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path0 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            talk_head=talk_head,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.use_attn_bias = use_attn_bias
        self.num_rbf = num_rbf

        if self.use_attn_bias:
            self.distance_encoder = nn.Sequential(
                norm_layer(self.num_rbf * 2),
                nn.Linear(self.num_rbf * 2, num_heads),
                act_layer(),
            )
            rbf_mu = torch.linspace(0, 20, self.num_rbf).view(1, 1, 1, -1)
            rbf_sigma = (torch.ones(self.num_rbf) * 20.0 / self.num_rbf).view(
                1, 1, 1, -1
            )
            self.register_buffer("rbf_mu", rbf_mu)
            self.register_buffer("rbf_sigma", rbf_sigma)
            self.edge_type_emb = nn.Embedding(4, self.num_rbf)
            if "fix_edge_type" in self.args.debug:
                self.edge_type_emb = nn.Embedding(_NUM_ATOM_TYPES**2 + 1, self.num_rbf)

    def forward(
        self,
        x=None,
        attn_mask=None,
        attn_bias=None,
        edge_type=None,
        local_mask=None,
        attn_gate=None,
        edge_feat=None,
    ):
        if self.use_attn_bias:
            device = attn_bias.get_device()
            if device < 0:
                device = "cpu"
            attn_bias_feat = torch.exp(
                -(((attn_bias.unsqueeze(-1) - self.rbf_mu) / self.rbf_sigma) ** 2)
            )
            attn_bias_feat = torch.cat(
                (attn_bias_feat, self.edge_type_emb(edge_type)), dim=-1
            )
            attn_bias = self.distance_encoder(attn_bias_feat).permute(0, 3, 1, 2)
            if edge_feat is not None:
                attn_bias += edge_feat

        else:
            attn_bias = None
        if self.num_dist_mask == 0:
            first_attn_mask = attn_mask
            second_attn_mask = attn_mask
        elif self.num_dist_mask == 1:
            first_attn_mask = attn_mask + local_mask
            second_attn_mask = attn_mask
        else:
            first_attn_mask = attn_mask + local_mask
            second_attn_mask = attn_mask + local_mask
        if self.use_attention_pool:
            x = x + self.drop_path0(
                self.ls0(
                    self.attn_pool0(x)
                    * self.attn0(
                        self.norm0(x),
                        attn_mask=first_attn_mask,
                        attn_bias=attn_bias,
                    )
                )
            )
            x = x + self.drop_path1(
                self.ls1(
                    self.attn_pool1(x)
                    * self.attn(
                        self.norm1(x), attn_mask=second_attn_mask, attn_bias=attn_bias
                    )
                )
            )
        else:
            x = x + self.drop_path0(
                self.ls0(
                    self.attn0(
                        self.norm0(x),
                        attn_mask=first_attn_mask,
                        attn_bias=attn_bias,
                    )
                )
            )

            x = x + self.drop_path1(
                self.ls1(
                    self.attn(
                        self.norm1(x), attn_mask=second_attn_mask, attn_bias=attn_bias
                    )
                )
            )
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class Backbone(nn.Module):
    def __init__(
        self,
        num_rbf=4,
        embedding_size=128,
        depth=8,
        _DEFAULT_V_DIM=(128, 10),
        norm_layer=nn.LayerNorm,
        drop_rate=0.0,
        head_drop_rate=0.0,
        atom_drop_rate=0.0,
        use_physical_properties=True,
        use_physical_properties_input=True,
        talk_head=False,
        use_attention_pool=False,
        edge_encoding=False,
        mask_closest_atom=False,
        num_dist_mask=1,
        dist_thres=8,
        args=None,
    ):

        super().__init__()
        self.args = args
        self.use_physical_properties = use_physical_properties
        self.mask_closest_atom = mask_closest_atom
        self.use_physical_properties_input = use_physical_properties_input
        self.num_rbf = num_rbf
        self.atom_drop = nn.Dropout(p=atom_drop_rate)

        self.embed_atoms = nn.Embedding(_NUM_ATOM_TYPES, embedding_size)

        if self.use_physical_properties:
            w_indim = embedding_size + 2
        else:
            w_indim = embedding_size

        self.pp_linear = nn.Linear(2, embedding_size)
        self.sasa_cat_embed = nn.Embedding(2, embedding_size)
        self.charge_cat_embed = nn.Embedding(3, embedding_size)

        self.W_v = nn.Sequential(
            norm_layer(w_indim), nn.Linear(w_indim, _DEFAULT_V_DIM[0])
        )
        self.num_heads = max(8, _DEFAULT_V_DIM[0] // 32)

        self.layers = []
        for _ in range(depth):
            if _ == depth - 1:
                use_ap = use_attention_pool
                pool_select = 128
            else:
                use_ap = False
                pool_select = None

            self.layers.append(
                Transformer(
                    dim=_DEFAULT_V_DIM[0],
                    num_heads=self.num_heads,
                    mlp_ratio=2.0,
                    num_rbf=self.num_rbf,
                    num_dist_mask=num_dist_mask,
                    talk_head=talk_head,
                    init_values=1e-3,
                    use_attention_pool=use_ap,
                    drop_path=drop_rate,
                    select_thres=pool_select,
                    args=self.args,
                )
            )
        self.layers = nn.ModuleList(self.layers)

        ns, _ = _DEFAULT_V_DIM

        # self.dense = nn.Sequential(nn.Dropout(p=head_drop_rate), nn.Linear(1 * ns, 24))
        self.dense = nn.Sequential(nn.Dropout(p=head_drop_rate), nn.Linear(1 * ns, 20))

        self.mask_node = nn.Parameter(torch.zeros(w_indim))

        self.drop = nn.Dropout(p=0.0)
        self.mb = 1
        self.dt = [dist_thres, 4.5, 2.2]
        self.pd = dist_thres
        self.use_pp_loss = False

        self.edge_encoding = edge_encoding
        if self.edge_encoding:
            self.edge_encoding_layer = nn.Sequential(
                nn.Linear(_DEFAULT_V_DIM[0], _DEFAULT_V_DIM[0] // 4),
                Transformer(
                    dim=_DEFAULT_V_DIM[0] // 4,
                    num_heads=self.num_heads,
                    mlp_ratio=2.0,
                    num_dist_mask=num_dist_mask,
                ),
                nn.Linear(self.num_heads, self.num_heads),
            )
        if "node3d" in self.args.debug:
            self.coord_emb = nn.Linear(3, embedding_size)

    def forward(self, feats, atom_types, pp, coords, ca, mask, return_idx=[3]):
        
        if feats[0] == None: 
            feats = self.embed_atoms(atom_types)

            charge = pp[..., 0]
            charge_cat = torch.zeros_like(charge, dtype=torch.long)  # convert to categorical
            charge_cat = torch.where(charge > 0.5, 1, charge_cat)
            charge_cat = torch.where(charge < -0.5, -1, charge_cat)
            charge_delta = charge - charge_cat
            charge_delta = ((charge_delta - 0.0915) / 0.1319).clamp(-3, 3)
            charge_cat[charge_cat == -1] = 2  # categorial input need to be [0, 1, 2] not [-1, 0, 1]

            # Charge is mostly 0, rest is one-sided gaussian
            sasa = pp[..., 1]
            sasa_neutral = (sasa == 0).long()
            sasa_norm = (sasa / 10).clamp(-3, 3)

            feats = feats + self.charge_cat_embed(charge_cat) + self.sasa_cat_embed(sasa_neutral) +\
                self.pp_linear(torch.stack([charge_delta, sasa_norm], dim=-1))
            
            feats = torch.cat((feats, pp), dim=-1)
            feats = self.W_v(feats)
        
        batchsize = feats.shape[0]
        self._max_length = feats.shape[1]
        self.max_length = int(self._max_length + self.mb)
        mb = self.mb
        h_V = feats

        if self.mask_closest_atom:
            d = torch.norm(ca.unsqueeze(dim=1) - coords, dim=-1)
            mask = torch.where(d < 4.5, 0, mask)

        mask_node = self.mask_node.reshape(1, 1, -1).repeat(batchsize, self.mb, 1)
        input_data = h_V.reshape(batchsize, self._max_length, -1)

        input_data = torch.cat((self.W_v(mask_node), input_data), dim=1)

        l_neg = torch.finfo(h_V.dtype).min
        our_mask = mask.reshape(-1, self._max_length)
        our_mask = torch.cat([our_mask.new_ones(batchsize, self.mb), our_mask], dim=1)
        attention_mask = our_mask.unsqueeze(dim=-1) * our_mask.unsqueeze(dim=-2)
        attention_mask = torch.where(attention_mask > 0.5, 0, l_neg)

        location = torch.cat(
            (
                ca.reshape([batchsize, self.mb, -1]),
                coords.reshape([batchsize, self._max_length, -1]),
            ),
            dim=1,
        )
        relative_dis = (location.unsqueeze(-3) - location.unsqueeze(-2)).norm(dim=-1)
        attn_bias = relative_dis

        edge_type = h_V.new_zeros(
            [batchsize, self.max_length, self.max_length], dtype=torch.long
        )
        edge_type[relative_dis < self.dt[0]] = 1
        edge_type[relative_dis < self.dt[1]] = 2
        edge_type[relative_dis < self.dt[2]] = 3

        local_mask = torch.where(relative_dis < self.dt[0], 0, l_neg)
        final_mask = torch.where(relative_dis < self.pd, attention_mask, l_neg)

        if self.edge_encoding:
            edge_encode_mask = attention_mask

            edge_feat = self.edge_encoding_layer[0](input_data)
            edge_feat = self.edge_encoding_layer[1](
                edge_feat,
                attn_mask=edge_encode_mask,
                edge_type=edge_type,
                attn_bias=attn_bias,
                local_mask=local_mask,
            )
            edge_feat = edge_feat.unsqueeze(dim=1) - edge_feat.unsqueeze(dim=2)
            es = edge_feat.shape
            edge_feat = torch.exp(
                -torch.norm(
                    edge_feat.reshape(es[0], es[1], es[2], self.num_heads, -1), dim=-1
                ).permute(0, 3, 1, 2)
            )
        else:
            edge_feat = None

        x = input_data
        local_enc_feats = []
        count = 0
        return_idx = [len(self.layers) - 1]
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attention_mask,
                local_mask=local_mask,
                attn_bias=attn_bias,
                edge_type=edge_type,
                edge_feat=edge_feat,
            )
            if count in return_idx:
                local_enc_feats.append(x)
            count += 1

        h_pool = h_V.new_zeros((x.shape[0], self.mb, x.shape[2]))
        for mb_index in range(self.mb):
            head_mask = our_mask.bool() & (relative_dis[:, mb_index] < self.pd)
        h_pool[:, mb_index, :] = (x * head_mask[..., None]).sum(dim=1) / head_mask.sum(
            dim=1, keepdim=True
        )
        pred_aa = self.dense(self.drop(h_pool))[:, 0, :]

        local_enc_feats = torch.cat(local_enc_feats, dim=-1)

        return local_enc_feats, pred_aa


if __name__ == "__main__":
    model = Backbone().cuda()
    feats = torch.ones(2, 512, 128).long().cuda()
    coords = torch.ones(2, 512, 3).cuda()
    ca = torch.ones(2, 3).cuda()
    mask = torch.ones(2, 512).cuda()
    model(feats=feats, coords=coords, ca=ca, mask=mask)
