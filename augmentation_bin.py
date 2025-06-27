import cv2
import numpy as np
import random

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, SpatialTransform_2, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform, LocalSmoothingTransform, LocalContrastTransform

def get_train_transform_2D(patch_size):
    img_trans = {
        'Spatial': Compose([ResizeTransform(target_size=patch_size),
                            SpatialTransform(patch_size, [i // 2 for i in patch_size],
                                             do_elastic_deform=True, alpha=(0., 100.), sigma=(10., 13.),
                                             do_rotation=True, angle_x=(-90 / 360. * 2 * np.pi, 90 / 360. * 2 * np.pi),
                                             angle_y=(-90 / 360. * 2 * np.pi, 90 / 360. * 2 * np.pi),
                                             do_scale=True, scale=(0.75, 1.25),
                                             border_mode_data='constant', border_cval_data=0, order_data=3,
                                             border_mode_seg='constant', border_cval_seg=0, order_seg=0,
                                             random_crop=False, data_key="data", label_key="seg",
                                             p_el_per_sample=0.5, p_scale_per_sample=0.5, p_rot_per_sample=0.5),
                          ]),
        'Local': Compose([#BrightnessGradientAdditiveTransform(-0.1, 0.1),
                          LocalGammaTransformBin(gamma_range=(0.5, 1.5), patch_scale=16,
                                                 invert_image=False, per_channel=False,
                                                 p_per_sample=1),
                          #LocalSmoothingTransform((0, 1)),
                          ]),
        'Inten': Compose([BrightnessTransformBin((-0.1, 0.1), 0, per_channel=False, data_key="data", p_per_sample=0.5,
                                                 p_per_channel=0.5),
                          GammaTransformBin(gamma_range=(0.75, 1.25), invert_image=False, per_channel=False,
                                            p_per_sample=0.5),
                          GaussianNoiseTransform(noise_variance=(0.001, 0.1), p_per_sample=0.5),
                          ]),
        'trainex': Compose([ResizeTransform(target_size=patch_size),
                          # SharpenTransform(magnitude=(10, 30), p=0.5),
                          # GaussianBlurTransform(blur_sigma=(0.25, 1.5), different_sigma_per_channel=False,
                          #                       p_per_channel=0.5, p_per_sample=0.5),
                          # MirrorTransform(),
                          # PerturbationTransformBin(add_magnitude=(0, 0.1), multi_magnitude=(0.9, 1.1), p=0.5),
                          # GaussianNoiseTransform(noise_variance=(0.01, 0.5), p_per_sample=0.5),
                          # BrightnessTransformBin((-0.2, 0.2), 0, per_channel=False, data_key="data", p_per_sample=0.5,
                          #                        p_per_channel=0.5),
                          # GammaTransformBin(gamma_range=(0.5, 1.5), invert_image=False, per_channel=False,
                          #                   p_per_sample=0.5),
                          SpatialTransform(patch_size, [i // 2 for i in patch_size],
                                           do_elastic_deform=True, alpha=(0., 100.), sigma=(10., 13.),
                                           do_rotation=True, angle_x=(-45 / 360. * 2 * np.pi, 45 / 360. * 2 * np.pi),
                                           angle_y=(-45 / 360. * 2 * np.pi, 45 / 360. * 2 * np.pi),
                                           do_scale=True, scale=(0.6, 1.4),
                                           border_mode_data='constant', border_cval_data=0, order_data=3,
                                           border_mode_seg='constant', border_cval_seg=0, order_seg=0,
                                           random_crop=False, data_key="data", label_key="seg",
                                           p_el_per_sample=0.5, p_scale_per_sample=0.5, p_rot_per_sample=0.5),
                          # BrightnessTransformBin((-0.1, 0.1), 0, per_channel=False, data_key="data", p_per_sample=0.5,
                          #                        p_per_channel=0.5),
                          # GammaTransformBin(gamma_range=(0.5, 1.5), invert_image=False, per_channel=False,
                          #                   p_per_sample=0.5),
                          # GaussianNoiseTransform(noise_variance=(0.01, 0.1), p_per_sample=0.5),
                          ]),
        'val': Compose([ResizeTransform(target_size=patch_size)]),
    }

    return img_trans

class SharpenTransform(AbstractTransform):
    """Adds Sharpen

    """

    def __init__(self, magnitude=(10, 30), data_key="data", label_key="seg", p=0.5):
        self.blurr = GaussianBlurTransform(blur_sigma=(0.25, 1.5), different_sigma_per_channel=True,
                                           p_per_channel=0.5, p_per_sample=1)
        self.magnitude = magnitude
        self.p = p

    def __call__(self, **data_dict):
        mask = data_dict.get("seg")
        if np.random.uniform() < self.p:
            blurr_1 = self.blurr(**data_dict)
            blurr_2 = self.blurr(**blurr_1)
            blurr_1_image = blurr_1.get("data")
            blurr_2_image = blurr_2.get("data")
            alpha = np.random.uniform(self.magnitude[0], self.magnitude[1])
            sharpen_image = blurr_1_image + (blurr_1_image - blurr_2_image) * alpha
            data_dict = dict(data=sharpen_image, seg=mask)
        return data_dict

class PerturbationTransformBin(AbstractTransform):
    """Perturbation

    """

    def __init__(self, add_magnitude=(0, 0.1), multi_magnitude=(0.9, 1.1), data_key="data", label_key="seg", p=0.5):
        self.brightness = BrightnessTransform(add_magnitude, 0, p_per_sample=1)
        self.contrast = BrightnessMultiplicativeTransform(multi_magnitude, p_per_sample=1)
        self.p = p

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p:
            data_dict = self.contrast(**data_dict)
            data_dict = self.brightness(**data_dict)
        return data_dict


class GammaTransformBin(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        gamma = np.random.uniform(gamma_range[0], gamma_range[1])
        # if np.random.random() < 0.5 and gamma_range[0] < 1:
        #     gamma = np.random.uniform(gamma_range[0], 1)
        # else:
        #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            # if np.random.random() < 0.5 and gamma_range[0] < 1:
            #     gamma = np.random.uniform(gamma_range[0], 1)
            # else:
            #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample

class LocalGammaTransformBin(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), patch_scale = 8, invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.patch_scale = patch_scale
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma_local(data_dict[self.data_key][b],
                                                                  self.gamma_range, self.patch_scale,
                                                                  self.invert_image,
                                                                  per_channel=self.per_channel,
                                                                  retain_stats=self.retain_stats)
        return data_dict

def augment_gamma_local(data_sample, gamma_range=(0.5, 2), patch_scale=8, invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        # if np.random.random() < 0.5 and gamma_range[0] < 1:
        #     gamma = np.random.uniform(gamma_range[0], 1)
        # else:
        #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_shape = data_sample.shape
        patch_size = int(data_shape[1] / patch_scale)
        for x in range(0, data_shape[1] - patch_size + 1, patch_size):
            for y in range(0, data_shape[1] - patch_size, patch_size):
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                data_sample[:, x:x+patch_size, y:y+patch_size] = np.power(((data_sample[:, x:x+patch_size, y:y+patch_size] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        # data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            # if np.random.random() < 0.5 and gamma_range[0] < 1:
            #     gamma = np.random.uniform(gamma_range[0], 1)
            # else:
            #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_shape = data_sample.shape
            patch_size = int(data_shape[1] / patch_scale)
            for x in range(0, data_shape[0] - patch_size + 1, patch_size):
                for y in range(0, data_shape[0] - patch_size, patch_size):
                    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                    data_sample[c, x:x + patch_size, y:y + patch_size] = np.power(
                        ((data_sample[c, x:x + patch_size, y:y + patch_size] - minm) / float(rnge + epsilon)),
                        gamma) * rnge + minm

            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample

class LocalIntenTransformBin(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), patch_scale = 8, invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.patch_scale = patch_scale
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma_local(data_dict[self.data_key][b],
                                                                  self.gamma_range, self.patch_scale,
                                                                  self.invert_image,
                                                                  per_channel=self.per_channel,
                                                                  retain_stats=self.retain_stats)
        return data_dict

def augment_Inten_local(data_sample, gamma_range=(0.5, 1.5), gamma_flag=True, gauss_noise=(0, 0.15), gauss_flag=True,
                        bright_range=(-0.1, 0.1), bright_flag=True, patch_scale=8, invert_image=False, epsilon=1e-7, per_channel=False,
                        retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        # if np.random.random() < 0.5 and gamma_range[0] < 1:
        #     gamma = np.random.uniform(gamma_range[0], 1)
        # else:
        #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_shape = data_sample.shape
        patch_size = int(data_shape[1] / patch_scale)
        for x in range(0, data_shape[1] - patch_size + 1, patch_size):
            for y in range(0, data_shape[1] - patch_size, patch_size):
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                bright = np.random.uniform(bright_range[0], bright_range[1])
                noise = np.random.normal(0, np.random.uniform(gauss_noise[0], gauss_noise[1]),
                                         size=(patch_size, patch_size))

                if bright_flag and np.random.uniform() < 0.5:
                    data_sample[:, x:x + patch_size, y:y + patch_size] = data_sample[:, x:x + patch_size, y:y + patch_size] + bright
                if gamma_flag and np.random.uniform() < 0.5:
                    data_sample[:, x:x+patch_size, y:y+patch_size] = np.power(((data_sample[:, x:x+patch_size, y:y+patch_size] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
                if gauss_flag and np.random.uniform() < 0.5:
                    data_sample[:, x:x + patch_size, y:y + patch_size] = data_sample[:, x:x + patch_size, y:y + patch_size] + noise
        # data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            gamma = np.random.uniform(gamma_range[0], gamma_range[1])
            # if np.random.random() < 0.5 and gamma_range[0] < 1:
            #     gamma = np.random.uniform(gamma_range[0], 1)
            # else:
            #     gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_shape = data_sample.shape
            patch_size = int(data_shape[1] / patch_scale)
            for x in range(0, data_shape[0] - patch_size + 1, patch_size):
                for y in range(0, data_shape[0] - patch_size, patch_size):
                    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                    bright = np.random.uniform(bright_range[0], bright_range[1])
                    noise = np.random.normal(0, np.random.uniform(gauss_noise[0], gauss_noise[1]), size=(patch_size, patch_size))

                    if bright_flag and np.random.uniform() < 0.5:
                        data_sample[c, x:x + patch_size, y:y + patch_size] = data_sample[c, x:x + patch_size,
                                                                             y:y + patch_size] + bright
                    if gamma_flag and np.random.uniform() < 0.5:
                        data_sample[c, x:x + patch_size, y:y + patch_size] = np.power(
                            ((data_sample[c, x:x + patch_size, y:y + patch_size] - minm) / float(rnge + epsilon)),
                            gamma) * rnge + minm
                    if gauss_flag and np.random.uniform() < 0.5:
                        data_sample[c, x:x + patch_size, y:y + patch_size] = data_sample[c, x:x + patch_size,
                                                                             y:y + patch_size] + noise

            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample

class BrightnessTransformBin(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True, data_key="data", p_per_sample=1, p_per_channel=1):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel

    def __call__(self, **data_dict):
        data = data_dict[self.data_key]

        if len(self.mu) > 1:
            mu = random.uniform(self.mu[0], self.mu[1])
        # elif len(self.mu) == 1:
        #     mu = self.mu[0]

        for b in range(data.shape[0]):
            if np.random.uniform() < self.p_per_sample:
                data[b] = augment_brightness_additive(data[b], mu, self.sigma, self.per_channel,
                                                      p_per_channel=self.p_per_channel)

        data_dict[self.data_key] = data
        return data_dict


def augment_brightness_additive(data_sample, mu:float, sigma:float , per_channel:bool=True, p_per_channel:float=1.):
    """
    data_sample must have shape (c, x, y(, z)))
    :param data_sample:
    :param mu:
    :param sigma:
    :param per_channel:
    :param p_per_channel:
    :return:
    """
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                # minm = data_sample[c].min()
                # rnge = data_sample[c].max() - minm
                # data_sample[c] = ((data_sample[c] - minm) / float(rnge + 1e-7) + rnd_nb) * float(rnge + 1e-7) + minm
                data_sample[c] = data_sample[c] + rnd_nb
    else:
        for c in range(data_sample.shape[0]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                # minm = data_sample[c].min()
                # rnge = data_sample[c].max() - minm
                # data_sample[c] = ((data_sample[c] - minm) / float(rnge + 1e-7) + rnd_nb) * float(rnge + 1e-7) + minm
                data_sample[c] = data_sample[c] + rnd_nb
    return data_sample

# if __name__ == '__main__':
#     transforms = get_train_transform_2D((768, 768))
#     print("test")
