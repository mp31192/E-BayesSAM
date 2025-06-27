import os
from argparse import Namespace

import numpy as np
import torch

import matplotlib.pyplot as plt
import cv2
from augmentation_bin import get_train_transform_2D
from SAM_ConstructModel import get_SAM
import torch.nn.functional as F

from PIL import Image
import copy
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    global refPt, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

        img_show_box = copy.deepcopy(img_array)

        cv2.rectangle(img_show_box, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", img_show_box)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_uncertainty(mask, ax):
    # mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    # color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)# * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap='hot')

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    point0 = box[0]
    point1 = box[1]
    x0, y0 = point0[0], point0[1]
    w, h = point1[0] - point0[0], point1[1] - point0[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=5))

args = Namespace()

save_result_flag = True
model_save = "/media/ulab/work/SAMMED2D-master/Checkpoint/SAM_BNN_OneShot"
if os.path.exists(model_save) == False and save_result_flag:
    os.makedirs(model_save)

box_available = False

data_npy_path = "data_npy.npz"
data_npy = np.load(data_npy_path, allow_pickle=True)
data_mu_all = data_npy['mu']
data_mu_all = data_mu_all[:, 0, :]
data_mu = np.mean(data_mu_all, axis=0)
data_std = np.std(data_mu_all, axis=0)

tokens_number = 3
DAG_flag = False
model, data_preprocessor = get_SAM(model_name="LiteMedSAM", pretrain_model=True, num_tokens=tokens_number, Causality=True, DAG=DAG_flag)
model = model.to(device)

data_std = torch.from_numpy(data_std).to(device)
data_std = data_std.unsqueeze(0).unsqueeze(0)
data_std = torch.repeat_interleave(data_std, 4, 1)
model.mask_decoder.tokens_std = data_std

#############################################
transformers_val = get_train_transform_2D(patch_size=(256, 256))['val']

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--image_path', help='path of image file', required=True)
parser.add_argument('-u', '--show_uncertainty', help='whether to show uncertainty', default=False)
args = parser.parse_args()

ImagePath = args.image_path

with torch.no_grad():
    model.eval()
    # ImagePath = "/media/ulab/work/SAMMED2D-master/E-BayesSAM/data_demo/images/amos_0507_31.png"
    raw_image = Image.open(ImagePath)
    img_array = np.asarray(raw_image)

    ori_shape = np.shape(img_array)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    cv2.imshow("image", img_array)
    while True:
        if cv2.waitKey(1) & 0xFF == ord(" "):
            break
    cv2.destroyAllWindows()
    if len(img_array.shape) < 3:
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.repeat(img_array, 3, 2)

    image = img_array  # .astype('float')

    image = np.expand_dims(image, 0)
    image = np.transpose(image, [0, 3, 1, 2])

    image = image.astype('float32')

    image = image / 255
    mask = np.zeros_like(image)
    LT = refPt[0]
    RB = refPt[1]
    mask[:, :, LT[1]:RB[1], LT[0]:RB[0]] = 1

    data_dict = dict(data=image, seg=mask)
    augmented = transformers_val(**data_dict)
    image_spatial = copy.deepcopy(augmented.get("data"))
    mask_spatial = copy.deepcopy(augmented.get("seg"))
    image_spatial = np.transpose(image_spatial, [0, 2, 3, 1])
    mask_spatial = np.transpose(mask_spatial, [0, 2, 3, 1])

    image_spatial[image_spatial < 0] = 0
    image_spatial[image_spatial > 1] = 1
    image_spatial = image_spatial * 255
    image_spatial = image_spatial.astype('uint8')

    input_dict = data_preprocessor.preprocess(image_spatial[0, :, :, :], mask_spatial[0, :, :, 0], data_aug=False)

    input_image = input_dict['image'].to(device).unsqueeze(0)
    input_box = input_dict['bboxes'].to(device).unsqueeze(0)
    labels_mask = input_dict['gt2D'].to(device).unsqueeze(0)

    prediction = model(
        image=input_image, boxes=input_box, uncertainty_output=True)

    iou_predictions = prediction['iou_pred']
    logits_pred_all = prediction['masks_kan']

    mask_slice = torch.argmax(iou_predictions)
    iou_predictions = iou_predictions[:, mask_slice]

    selected_cls_index = [mask_slice]

    scores_all = torch.sigmoid(logits_pred_all)
    scores_all = scores_all[:, mask_slice:mask_slice + 1, :, :]
    scores = torch.mean(scores_all, dim=0, keepdim=True)

    uncertainty_map = torch.std(scores_all, dim=0, keepdim=True)

    input_image = F.interpolate(input_image, size=(ori_shape[0], ori_shape[1]), mode="bilinear")
    scores = F.interpolate(scores, size=(ori_shape[0], ori_shape[1]), mode="bilinear")
    uncertainty_map = F.interpolate(uncertainty_map, size=(ori_shape[0], ori_shape[1]), mode="bilinear")

    plt.figure(figsize=(10, 10))
    plt.imshow(input_image[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
    show_mask(torch.round(scores[0, 0, :, :]).detach().cpu().numpy(), plt.gca(), random_color=False)
    show_box(refPt, plt.gca())
    if args.show_uncertainty:
        show_uncertainty(uncertainty_map[0, 0, :, :].detach().cpu().numpy(), plt.gca())
    plt.axis("off")
    plt.show()

