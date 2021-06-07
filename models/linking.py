#!/usr/bin/env python
"""
    File Name   :   s3g-linking
    date        :   2/12/2019
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import torch
from torch import nn
import MinkowskiEngine as ME
from MinkowskiEngine.utils import get_coords_map


class Linking(nn.Module):
    def __init__(self, fea2d_dim, fea3d_dim, viewNum=3):
        super(Linking, self).__init__()
        self.viewNum = viewNum
        self.fea2d_dim = fea2d_dim

        self.view_fusion = nn.Sequential(
            ME.MinkowskiConvolution(fea2d_dim * viewNum, fea2d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(fea2d_dim),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(fea2d_dim, fea3d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(fea3d_dim),
            ME.MinkowskiReLU(inplace=True)
        )

        self.fuseTo3d = nn.Sequential(
            ME.MinkowskiConvolution(fea3d_dim * 2, fea3d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(fea3d_dim),
            ME.MinkowskiReLU(inplace=True)
        )

        self.view_sep = nn.Sequential(
            ME.MinkowskiConvolution(fea3d_dim, fea2d_dim, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(fea2d_dim),
            ME.MinkowskiReLU(inplace=True)
        )
        self.fuseTo2d = nn.Sequential(
            nn.Conv2d(fea2d_dim * 2, fea2d_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fea2d_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_2d_all, feat_3d, links, init_3d_data=None):
        """
        :param feat_2d_all: V_B * C * H * WV
        :param feat_3d: SparseTensor, Feature of N*C
        :return:
        """
        feat_3d_for_2d = self.view_sep(feat_3d).F
        V_B, C, H, W = feat_2d_all.shape
        feat_2d_all = feat_2d_all.view(self.viewNum, -1, C, H, W)

        # Link
        coords_map_in, coords_map_out = get_coords_map(init_3d_data, feat_3d)
        current_links = torch.zeros([feat_3d.shape[0], links.shape[1], links.shape[2]], dtype=torch.long).cuda()
        current_links[coords_map_out, :] = links[coords_map_in, :]

        feat_3d_to_2d = torch.zeros_like(feat_2d_all)
        feat_2d_to_3d = torch.zeros([feat_3d.F.shape[0], self.viewNum * self.fea2d_dim], dtype=torch.float).cuda()
        for v in range(self.viewNum):
            # pdb.set_trace()
            f = feat_2d_all[v, current_links[:, 0, v], :, current_links[:, 1, v], current_links[:, 2, v]]
            f *= current_links[:, 3, v].unsqueeze(dim=1).float()
            feat_2d_to_3d[:, v * self.fea2d_dim:(v + 1) * self.fea2d_dim] = f
            feat_3d_to_2d[v, current_links[:, 0, v], :, current_links[:, 1, v], current_links[:, 2, v]] = feat_3d_for_2d

        feat_3d_to_2d = feat_3d_to_2d.view(V_B, C, H, W)
        feat_2d_all = feat_2d_all.view(V_B, C, H, W)
        fused_2d = self.fuseTo2d(torch.cat([feat_2d_all, feat_3d_to_2d], dim=1))

        feat_2d_to_3d = ME.SparseTensor(feat_2d_to_3d, feat_3d.C)
        feat_2d_to_3d = self.view_fusion(feat_2d_to_3d)
        # pdb.set_trace()
        feat_3d._F = torch.cat([feat_3d._F, feat_2d_to_3d._F], dim=-1)
        fused_3d = self.fuseTo3d(feat_3d)

        return fused_3d, fused_2d
