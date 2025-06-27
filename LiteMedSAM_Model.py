from segment_anything_LiteMedSAM.modeling import MaskDecoder, CausalMaskDecoder, PromptEncoder, TwoWayTransformer, CausalPromptEncoder, TwoWayCausalTransformer
from tiny_vit_sam import TinyViT
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import random
import cv2
import torch.nn.functional as F
########### Creat Model #############
# class MedSAM_Lite(nn.Module):
#     def __init__(self,
#                  image_encoder,
#                  mask_decoder,
#                  prompt_encoder
#                  ):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.mask_decoder = mask_decoder
#         self.prompt_encoder = prompt_encoder
#
#     def forward(self, image, boxes):
#         image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
#
#         sparse_embeddings, dense_embeddings = self.prompt_encoder(
#             points=None,
#             boxes=boxes,
#             masks=None,
#         )
#         low_res_masks, iou_predictions = self.mask_decoder(
#             image_embeddings=image_embedding,  # (B, 256, 64, 64)
#             image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
#             sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
#             dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
#             multimask_output=False,
#         )  # (B, 1, 256, 256)
#
#         return low_res_masks, iou_predictions
#
#     @torch.no_grad()
#     def postprocess_masks(self, masks, new_size, original_size):
#         """
#         Do cropping and resizing
#         """
#         # Crop
#         masks = masks[:, :, :new_size[0], :new_size[1]]
#         # Resize
#         masks = F.interpolate(
#             masks,
#             size=(original_size[0], original_size[1]),
#             mode="bilinear",
#             align_corners=False,
#         )
#
#         return masks

class MedSAM_Lite(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes, uncertainty_output=False):
        device = image.device
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # image_embedding = F.interpolate(image_embedding, size=[64, 64], mode="bilinear")

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        # low_res_masks_bayes = torch.zeros(self.montesampling, 1, 1, 256, 256).to(device)
        # iou_predictions_bayes = torch.zeros(self.montesampling, 1, 1).to(device)
        # log_prior = torch.zeros(self.montesampling, 1).to(device)
        # log_variational_posterior = torch.zeros(self.montesampling, 1).to(device)

        if uncertainty_output:
            low_res_masks, iou_predictions, low_res_masks_std, iou_predictions_std, mu_i, var_i, cos_tokens = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                mu_var = None,
                multimask_output=False,
                uncertainty_output=uncertainty_output,
            )  # (B, 1, 256, 256)
            return low_res_masks, iou_predictions, low_res_masks_std, iou_predictions_std, mu_i, var_i, cos_tokens
        else:
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                mu_var=None,
                multimask_output=False,
                uncertainty_output=uncertainty_output,
            )  # (B, 1, 256, 256)
            return low_res_masks, iou_predictions

        # low_res_masks_bayes[s] = torch.sigmoid(low_res_masks)
        # log_prior[s] = self.mask_decoder.log_prior()
        # log_variational_posterior[s] = self.mask_decoder.log_variational_posterior()
        # iou_predictions_bayes[s] = iou_predictions

        # low_res_masks = low_res_masks_bayes.mean(0)
        # outputs_seg_std = low_res_masks_bayes.std(0)
        # iou_predictions = iou_predictions_bayes.mean(0)
        # log_prior = log_prior.mean(0)
        # log_variational_posterior = log_variational_posterior.mean(0)
        # uncertainty = torch.mean(low_res_masks_bayes * torch.log(low_res_masks_bayes))  # / SAMPLES
        # uncertainty2 = torch.sum(outputs_seg_std) / torch.sum(low_res_masks)

        # return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks


class MedSAMCausalPrompt_Lite(nn.Module):
    def __init__(self,
                 image_encoder,
                 mask_decoder,
                 prompt_encoder
                 ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, point, boxes, mask_prompt, uncertainty_map, mu_var, uncertainty_output=False):
        device = image.device
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        if uncertainty_map is not None:
            uncertainty_map = torch.repeat_interleave(uncertainty_map, 3, 1)
            uncertainty_embedding = self.image_encoder(uncertainty_map).detach()  # (B, 256, 64, 64)
            # uncertainty_embedding = uncertainty_embedding + image_embedding
        else:
            uncertainty_embedding = None

        # image_embedding = F.interpolate(image_embedding, size=[64, 64], mode="bilinear")

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point,
            boxes=boxes,
            masks=mask_prompt,
            mu_var=mu_var,
            uncertainty=uncertainty_embedding,#uncertainty_map,
            image_embed=image_embedding
        )

        # low_res_masks_bayes = torch.zeros(self.montesampling, 1, 1, 256, 256).to(device)
        # iou_predictions_bayes = torch.zeros(self.montesampling, 1, 1).to(device)
        # log_prior = torch.zeros(self.montesampling, 1).to(device)
        # log_variational_posterior = torch.zeros(self.montesampling, 1).to(device)

        if uncertainty_output:
            low_res_masks, iou_predictions, low_res_masks_std, iou_predictions_std, mu_i, var_i, cos_tokens = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                mu_var=mu_var,
                multimask_output=False,
                uncertainty_output=uncertainty_output,
            )  # (B, 1, 256, 256)
            return low_res_masks, iou_predictions, low_res_masks_std, iou_predictions_std, mu_i, var_i, cos_tokens
        else:
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding,  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
                uncertainty_output=uncertainty_output,
            )  # (B, 1, 256, 256)
            return low_res_masks, iou_predictions

        # low_res_masks_bayes[s] = torch.sigmoid(low_res_masks)
        # log_prior[s] = self.mask_decoder.log_prior()
        # log_variational_posterior[s] = self.mask_decoder.log_variational_posterior()
        # iou_predictions_bayes[s] = iou_predictions

        # low_res_masks = low_res_masks_bayes.mean(0)
        # outputs_seg_std = low_res_masks_bayes.std(0)
        # iou_predictions = iou_predictions_bayes.mean(0)
        # log_prior = log_prior.mean(0)
        # log_variational_posterior = log_variational_posterior.mean(0)
        # uncertainty = torch.mean(low_res_masks_bayes * torch.log(low_res_masks_bayes))  # / SAMPLES
        # uncertainty2 = torch.sum(outputs_seg_std) / torch.sum(low_res_masks)

        # return low_res_masks, iou_predictions

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing
        """
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks

# %%
def get_LiteMedSAM(pretrained_encoder=None, pretrained_model=None, uncertainty=False):
    # medsam_lite_image_encoder = TinyViT(
    #     img_size=256,
    #     in_chans=3,
    #     embed_dims=[
    #         64, ## (64, 256, 256)
    #         128, ## (128, 128, 128)
    #         160, ## (160, 64, 64)
    #         320 ## (320, 64, 64)
    #     ],
    #     depths=[2, 2, 6, 2],
    #     num_heads=[2, 4, 5, 10],
    #     window_sizes=[7, 7, 14, 7],
    #     mlp_ratio=4.,
    #     drop_rate=0.,
    #     drop_path_rate=0.0,
    #     use_checkpoint=False,
    #     mbconv_expand_ratio=4.0,
    #     local_conv_size=3,
    #     layer_lr_decay=0.8
    # )

    medsam_lite_image_encoder=ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=256,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
        adapter_train=False,
    )

    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder,
        MonteSampling = uncertainty
    )
    medsam_lite_checkpoint = pretrained_model
    if medsam_lite_checkpoint is not None and pretrained_encoder is None:
        medsam_lite_ckpt = torch.load(
            medsam_lite_checkpoint,
            map_location="cpu"
        )
        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=True)
    return medsam_lite_model


def get_CausalLiteMedSAM(pretrained_encoder=None, pretrained_model=None, tokens_number=3, image_patch_embedding=64,Causality=False, DAG=False):
    if pretrained_encoder is None:
        medsam_lite_image_encoder = TinyViT(
            img_size=256,
            in_chans=3,
            embed_dims=[
                64, ## (64, 256, 256)
                128, ## (128, 128, 128)
                160, ## (160, 64, 64)
                320 ## (320, 64, 64)
            ],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8
        )
    else:
        medsam_lite_image_encoder=ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=256,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
            adapter_train=True,
        )

    if Causality:
        medsam_lite_prompt_encoder = CausalPromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_patch_embedding, image_patch_embedding),
            input_image_size=(256, 256),
            mask_in_chans=16
        )
    else:
        medsam_lite_prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_patch_embedding, image_patch_embedding),
            input_image_size=(256, 256),
            mask_in_chans=16
        )

    if DAG:
        medsam_lite_mask_decoder = CausalMaskDecoder(
            num_multimask_outputs=tokens_number,
            transformer=TwoWayCausalTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
    else:
        medsam_lite_mask_decoder = CausalMaskDecoder(
            num_multimask_outputs=tokens_number,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=256,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=256,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
        )

    if Causality:
        medsam_lite_model = MedSAMCausalPrompt_Lite(
            image_encoder = medsam_lite_image_encoder,
            mask_decoder = medsam_lite_mask_decoder,
            prompt_encoder = medsam_lite_prompt_encoder
        )
    else:
        medsam_lite_model = MedSAM_Lite(
            image_encoder = medsam_lite_image_encoder,
            mask_decoder = medsam_lite_mask_decoder,
            prompt_encoder = medsam_lite_prompt_encoder
        )

    medsam_lite_checkpoint = pretrained_model
    if medsam_lite_checkpoint is not None and pretrained_encoder is None:
        medsam_lite_ckpt = torch.load(
            medsam_lite_checkpoint,
            map_location="cpu"
        )

        # mask_tokens_weight = medsam_lite_ckpt["mask_decoder.mask_tokens.weight"].data.clone()
        # iou_prediction_tokens_weight = medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.weight"].data.clone()
        # iou_prediction_tokens_bias = medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.bias"].data.clone()
        #
        # num_mask_tokens = medsam_lite_mask_decoder.num_mask_tokens
        #
        # mask_tokens_weight_new = F.interpolate(mask_tokens_weight.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens, 256], mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
        # iou_prediction_tokens_weight_new = F.interpolate(iou_prediction_tokens_weight.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens, 256], mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
        # # print(iou_prediction_tokens_bias)
        # iou_prediction_tokens_bias_new = F.interpolate(iou_prediction_tokens_bias.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens], mode="linear", align_corners=True).squeeze(0).squeeze(0)
        # # print(iou_prediction_tokens_bias_new)
        #
        # medsam_lite_ckpt["mask_decoder.mask_tokens.weight"].data = mask_tokens_weight_new
        # medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.weight"].data = iou_prediction_tokens_weight_new
        # medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.bias"].data = iou_prediction_tokens_bias_new

        medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=False)
    elif medsam_lite_checkpoint is not None and pretrained_encoder is not None:
        medsam_lite_ckpt = torch.load(
            medsam_lite_checkpoint,
            map_location="cpu"
        )
        medsam_lite_ckpt = medsam_lite_ckpt["model"]
        pretrained_encoder_ckpt = torch.load(
            pretrained_encoder,
            map_location="cpu"
        )

        # mask_decoder_medsam_lite = {k: v for k, v in medsam_lite_ckpt.items() if "mask_decoder" in k}
        mask_decoder_sammed2d_lite = {k: v for k, v in pretrained_encoder_ckpt['model'].items() if "mask_decoder" in k}
        prompt_encoder_sammed2d_lite = {k: v for k, v in pretrained_encoder_ckpt['model'].items() if "prompt_encoder" in k}

        # new_pretrained_encoder_ckpt = load_from(medsam_lite_image_encoder, pretrained_encoder_ckpt, image_size=256, vit_patch_size=16)
        new_state_dict = {k.replace("image_encoder.", ""): v for k, v in pretrained_encoder_ckpt['model'].items() if "image_encoder" in k}# and "Adapter" not in k}
        # medsam_lite_image_encoder_state = medsam_lite_model.image_encoder.state_dict()
        medsam_lite_model.image_encoder.load_state_dict(new_state_dict, strict=True)

        for k, v in medsam_lite_ckpt.items():
            if "mask_decoder" in k:
                medsam_lite_ckpt[k] = mask_decoder_sammed2d_lite[k]
            if "prompt_encoder" in k:
                medsam_lite_ckpt[k] = prompt_encoder_sammed2d_lite[k]

        # mask_decoder_medsam_lite = {k: v for k, v in medsam_lite_ckpt.items() if "mask_decoder" in k}

        mask_tokens_weight = medsam_lite_ckpt["mask_decoder.mask_tokens.weight"].data.clone()
        iou_prediction_tokens_weight = medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.weight"].data.clone()
        iou_prediction_tokens_bias = medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.bias"].data.clone()

        num_mask_tokens = medsam_lite_mask_decoder.num_mask_tokens

        mask_tokens_weight_new = F.interpolate(mask_tokens_weight.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens, 256], mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
        iou_prediction_tokens_weight_new = F.interpolate(iou_prediction_tokens_weight.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens, 256], mode="bilinear", align_corners=True).squeeze(0).squeeze(0)
        # print(iou_prediction_tokens_bias)
        iou_prediction_tokens_bias_new = F.interpolate(iou_prediction_tokens_bias.unsqueeze(0).unsqueeze(0), size=[num_mask_tokens], mode="linear", align_corners=True).squeeze(0).squeeze(0)
        # print(iou_prediction_tokens_bias_new)
        medsam_lite_ckpt["mask_decoder.mask_tokens.weight"].data = mask_tokens_weight_new
        medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.weight"].data = iou_prediction_tokens_weight_new
        medsam_lite_ckpt["mask_decoder.iou_prediction_head.layers.2.bias"].data = iou_prediction_tokens_bias_new

        new_image_encoder_state_dict = {k.replace("image_encoder.", ""): v for k, v in medsam_lite_ckpt.items() if "image_encoder" in k}
        new_mask_decoder_state_dict = {k.replace("mask_decoder.", ""): v for k, v in medsam_lite_ckpt.items() if "mask_decoder" in k}
        new_prompt_encoder_state_dict = {k.replace("prompt_encoder.", ""): v for k, v in medsam_lite_ckpt.items() if "prompt_encoder" in k}

        # medsam_lite_model.image_encoder.load_state_dict(new_image_encoder_state_dict, strict=True)
        medsam_lite_model.mask_decoder.load_state_dict(new_mask_decoder_state_dict, strict=False)
        medsam_lite_model.prompt_encoder.load_state_dict(new_prompt_encoder_state_dict, strict=False)
    return medsam_lite_model
#####################################
############### Data Preprocessor ################
class DataPreprocessor():
    def __init__(self, image_size=256, bbox_shift=5, data_aug=True):
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug

    def preprocess(self, image, mask):
        img_3c = image  # (H, W, 3)
        img_resize = self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8,
                                                               a_max=None)  # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize)  # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1))  # (3, 256, 256)
        assert np.max(img_padded) <= 1.0 and np.min(img_padded) >= 0.0, 'image should be normalized to [0, 1]'
        if mask is not None:
            gt = mask  # multiple labels [0, 1,4,5...], (256,256)
            gt = cv2.resize(
                gt,
                (img_resize.shape[1], img_resize.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
            gt = self.pad_image(gt)  # (256, 256)
            label_ids = np.unique(gt)[1:]
            try:
                gt2D = np.uint8(gt == random.choice(label_ids.tolist()))  # only one label, (256, 256)
            except:
                gt2D = np.uint8(gt == np.max(gt))  # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                if mask is not None:
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                if mask is not None:
                    gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        if mask is not None:
            gt2D = np.uint8(gt2D > 0)
            y_indices, x_indices = np.where(gt2D > 0)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bboxes = np.array([x_min, y_min, x_max, y_max])
        if mask is not None:
            return {
                "image": torch.tensor(img_padded).float(),
                "gt2D": torch.tensor(gt2D[None, :, :]).long(),
                "bboxes": torch.tensor(bboxes[None, None, ...]).float(),  # (B, 1, 4)
                "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
                "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
            }
        else:
            return {
                "image": torch.tensor(img_padded).float(),
                "gt2D": torch.zeros_like(torch.tensor(img_padded)).long(),
                "bboxes": None,  # (B, 1, 4)
                "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
                "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
            }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3:  ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else:  ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded


def load_from(sam, state_dicts, image_size, vit_patch_size):

    for k, v in state_dicts['model'].items():
        print(k)

    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts['model'].items() if
                      k.split('.')[-1] in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if
                                                        '2' in k or
                                                        '5' in k or
                                                        '7' in k or
                                                        '8' in k or
                                                        '11' in k or
                                                        '13' in k or
                                                        '15' in k or
                                                        '23' in k or
                                                        '31' in k]
        # print(sam_dict)
        for k in global_rel_pos_keys:
            h_check, w_check = sam_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]

    sam_dict.update(new_state_dict)
    return sam_dict



# data_preprocessor = DataPreprocessor(image_size=256, bbox_shift=5, data_aug=True)
##################################################

