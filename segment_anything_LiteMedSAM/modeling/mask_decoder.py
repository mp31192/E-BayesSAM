# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import copy

import matplotlib.pyplot as plt
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d, MLPBlock
from segment_anything_LiteMedSAM.modeling.transformer import Attention
from torch.distributions import Normal

# from NormalizingFlowDensity import NormalizingFlowDensity
import segmentation_models_pytorch as smp
from fastkan import FastKANLayer

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class CausalMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        Causality: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.causality = Causality

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        if self.causality:
            self.tokens_std = None
            self.mask_feature = None


    def forward(
        self,
        image_embeddings: torch.Tensor,   #[B, 256, 64, 64]
        image_pe: torch.Tensor,           #[1, 256, 64, 64]
        sparse_prompt_embeddings: torch.Tensor, #[B, 3, 256]
        dense_prompt_embeddings: torch.Tensor,  #[B, 256, 64, 64]
        multimask_output: bool,
        uncertainty_output: bool,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        # Select the correct mask or masks for output
        if uncertainty_output:
            prediction_dict = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                uncertainty_output=uncertainty_output,
            )

            return prediction_dict
        else:
            prediction_dict = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                uncertainty_output=uncertainty_output
            )
            return prediction_dict
            # if multimask_output:
            #     mask_slice = slice(1, None)
            # else:
            #     mask_slice = slice(0, 1)
            # masks = masks[:, mask_slice, :, :]
            # iou_pred = iou_pred[:, mask_slice]
            #
            # # Prepare output
            # return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        uncertainty_output: bool,
    ):
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )

        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        # print(torch.max(output_tokens), torch.min(output_tokens))
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # print(torch.max(tokens), torch.min(tokens))
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings

        pos_src = image_pe
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)

        # print("************")
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out, kan_mode=False)

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)

        upscaled_embedding = self.output_upscaling(src)
        if uncertainty_output:
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :], kan_mode=False)
                )
            hyper_in = torch.stack(hyper_in_list, dim=1)

            hyper_kan_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_kan_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :], kan_mode=True)
                )
            hyper_kan_in = torch.stack(hyper_kan_in_list, dim=1)

            if self.tokens_std is not None:
                hyper_in = torch.repeat_interleave(hyper_in, 10, dim=0)
                hyper_kan_in = torch.repeat_interleave(hyper_kan_in, 10, dim=0)
                eps = torch.randn_like(hyper_in)
                hyper_in = hyper_in + eps * torch.repeat_interleave(self.tokens_std, 10, dim=0)
                hyper_kan_in = hyper_kan_in + eps * torch.repeat_interleave(self.tokens_std, 10, dim=0)

            if self.mask_feature is not None:
                hyper_in = hyper_in * torch.repeat_interleave(self.mask_feature, 10, dim=0)
                hyper_kan_in = hyper_kan_in * torch.repeat_interleave(self.mask_feature, 10, dim=0)

            _, c, h, w = upscaled_embedding.shape
            b, _, _ = hyper_in.shape
            upscaled_embedding = torch.repeat_interleave(upscaled_embedding, b, dim=0)
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
            masks_kan = (hyper_kan_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            return {"masks": masks, "masks_kan": masks_kan,"iou_pred": iou_pred, "mu_tokens": hyper_in, "mu_kan_tokens": hyper_kan_in}
        else:
            hyper_in_list: List[torch.Tensor] = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(
                    self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )

            hyper_in = torch.stack(hyper_in_list, dim=1)

            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            # Generate mask quality predictions
            iou_pred = self.iou_prediction_head(iou_token_out)

            return {"masks": masks, "iou_pred": iou_pred, "mu_tokens": hyper_in}

    def reparameterize_mask(self, mu, var):
        # std = torch.exp(var * 0.5)
        # std = var
        # sigma = F.softplus(var)
        eps = torch.randint_like(mu, high=2)
        # eps = torch.tanh_(eps)
        # eps[eps <= -1] = -1
        # eps[eps >= 1] = 1
        return eps * mu

    def get_kernelmean(self, mu):
        return mu
    def get_kernelstd(self, var):
        var_squared = var ** 2
        return var_squared

    # def kl_divergence_gaussians(self, mu1, sigma1, mu2, sigma2):
    #     term1 = torch.log(sigma2 / sigma1)
    #     term2 = (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2)
    #     term3 = -0.5
    #     return term1 + term2 + term3

    def kl_divergence_gaussians(self, mu1, mu2, sigma1, sigma2):
        sigma1 = torch.exp(sigma1 * 0.5)
        sigma2 = torch.exp(sigma2 * 0.5)
        kl_div = -0.5 * (1 + torch.log_(sigma2 / sigma1) - sigma2 / (sigma1 + 1e-6) - (mu2 - mu1) ** 2 / (sigma2 + 1e-6))
        return kl_div

    def Get_Geodesic(self, source_data, target_data):
        # source_data = source_data[0:1, :]
        # target_data = target_data[0:1, :]
        # source_data = torch.transpose(source_data, 1, 0)
        # target_data = torch.transpose(target_data, 1, 0)
        # source_data = torch.from_numpy(source_data)
        # target_data = torch.from_numpy(target_data)
        # source_data = F.adaptive_avg_pool2d(source_data, output_size=[1, 1])
        # target_data = F.adaptive_avg_pool2d(target_data, output_size=[1, 1])
        source_data = source_data.view(source_data.size(0), -1)
        target_data = target_data.view(target_data.size(0), -1)
        # source_data = source_data.cpu()
        # target_data = target_data.cpu()
        # print(torch.min(source_data), torch.max(source_data))
        u_s, s_s, v_s = torch.svd(source_data.t())
        u_t, s_t, v_t = torch.svd(target_data.t())

        pa = torch.mm(u_s.t(), u_t)
        # pa = torch.mm(u_s, u_t.t())
        # pa = torch.mm(source_data, target_data.t())

        p_s, cospa, p_t = torch.svd(pa)
        # cospa9 = cospa*0.99
        cospa2 = 1.00 - cospa.pow(2)#torch.pow(cospa, 2)
        cospa2 = torch.sigmoid(cospa2)
        # cospa = cospa*0.999
        sinpa = torch.sqrt(cospa2)
        # sinpa = (1-torch.pow(cospa, 2))**1
        rsd = torch.norm(sinpa, 1) + 0.01 * torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)
        # rsd = torch.mean(torch.abs_(sinpa))
        rsd.requires_grad_(True)
        if torch.any(torch.isnan(rsd)):
            print('RSD NaN appears')
        return rsd, cospa

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.KANlayers = nn.ModuleList(
            FastKANLayer(n, k, grid_min=-8, grid_max=8, num_grids=16) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.KANlayers[0].base_linear.weight = self.layers[0].weight
        self.KANlayers[1].base_linear.weight = self.layers[1].weight
        self.KANlayers[2].base_linear.weight = self.layers[2].weight
        self.KANlayers[0].base_linear.bias = self.layers[0].bias
        self.KANlayers[1].base_linear.bias = self.layers[1].bias
        self.KANlayers[2].base_linear.bias = self.layers[2].bias
        self.sigmoid_output = sigmoid_output

    def forward(self, x, kan_mode=False):
        if kan_mode:
            for i, layer in enumerate(self.KANlayers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.sigmoid_output:
                x = F.sigmoid(x)
        else:
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.sigmoid_output:
                x = F.sigmoid(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        loss = sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.KANlayers
        )
        return loss


class MLPstd(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)
        #
        self.std_output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x) #源码
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x_std = self.std_output(x)
                x = layer(x)
                if self.sigmoid_output:
                    x = F.sigmoid(x)
                x = torch.cat([x, x_std], dim=-1)
        # if self.sigmoid_output:
        #     x = F.sigmoid(x)
        return x
