import torch
import numpy as np
import cv2
import albumentations as A
import albumentations.pytorch.transforms as APT
import torchvision.transforms.functional as VF

def apply_affine_transformation(x):

    affine_angle, affine_shear, affine_translate_x, affine_translate_y = np.random.random(4)

    x_affine = VF.affine(x,
                        angle = -90 + 180 * affine_angle,
                        scale = 1.0, #0.7 + 0.6 * affine_scale,
                        translate = [int(40 * affine_translate_x - 20), int(40 * affine_translate_y - 20)],
                        shear = -25 + 50 * affine_shear)
    
    return x_affine, torch.as_tensor((affine_angle, affine_shear, affine_translate_x, affine_translate_y), dtype=torch.float16)

class DualImageTransformation:
    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        return self.base_transforms(x), self.base_transforms(x)

class DualImageAffineTransformation:
    def __init__(self, base_transforms):
        self.base_transforms = base_transforms

    def __call__(self, x):
        x0 = self.base_transforms(x)
        x1 = self.base_transforms(x)

        x0_affine, phi0 = apply_affine_transformation(x0)
        x1_affine, phi1 = apply_affine_transformation(x1)

        return x0, x1, x0_affine, x1_affine, phi0, phi1

class PatchShuffle:
    
    def __init__(self, npatches_x, npatches_y):
        self.npatches_x = npatches_x
        self.npatches_y = npatches_y
    
    def apply(self, img, **params):
        
        npatches_x = np.random.choice(self.npatches_x, 1)[0]
        npatches_y = np.random.choice(self.npatches_y, 1)[0]

        patch_width  = img.shape[1] // npatches_x
        patch_height = img.shape[0] // npatches_y
        
        patches = []
        
        # Extract patches
        for x in range(0, img.shape[1], patch_width):
            for y in range(0, img.shape[0], patch_height):
                patches.append(img[y:y+patch_height, x:x+patch_width])
                
        # Put back together in a random fashion
        img_new = np.empty_like(img)
        ixs = np.arange(len(patches))
        np.random.shuffle(ixs)
        
        i = 0

        for x in range(0, img.shape[1], patch_width):
            for y in range(0, img.shape[0], patch_height):
                img_new[y:y+patch_height, x:x+patch_width] = np.asarray(patches[ixs[i]])
                i += 1

        return img_new

class Transformer:

    def __init__(self, aug, img_size, geometric_transforms = False, dropout = False):
        self.aug = aug
        self.img_size = img_size
        self.geometric_transforms = geometric_transforms
        self.dropout = dropout

    def tx(self, img):

        aug = self.aug
        img_size = self.img_size

        if type(img) is not np.ndarray:
            img = np.asarray(img)

        if aug:
            transforms = [A.OneOf([
                            A.Resize(img_size[1], img_size[0], interpolation=0, p = 0.3), # 0 = cv2.INTER_NEAREST
                            A.CenterCrop(img_size[1], img_size[0], p = 0.2),
                            A.RandomResizedCrop(img_size[1], img_size[0], scale=(0.9, 1.2), ratio=(0.5, 0.8), p=0.5)
                            ], p = 1.0)]
        
            inner_transforms = [                
                 # flip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # downscale
                A.OneOf([
                    A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA), p=0.1),
                    A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=dict(upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA), p=0.1),
                    A.Downscale(scale_min=0.75, scale_max=0.95, interpolation=dict(upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR), p=0.8),
                ], p=0.125),
                # contrast
                A.OneOf([
                    A.RandomToneCurve(scale=0.3, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), brightness_by_max=True, always_apply=False, p=0.5)
                ], p=0.5)
            ]

            if self.geometric_transforms:
                inner_transforms.append(
                    # geometric
                    A.OneOf(
                        [
                            A.ShiftScaleRotate(shift_limit=None, scale_limit=[-0.15, 0.15], rotate_limit=[-30, 30], interpolation=cv2.INTER_LINEAR,
                                                border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, shift_limit_x=[-0.1, 0.1],
                                                shift_limit_y=[-0.2, 0.2], rotate_method='largest_box', p=0.6),
                            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                                                value=0, mask_value=None, approximate=False, same_dxdy=False, p=0.2),
                            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                                                value=0, mask_value=None, normalized=True, p=0.2),
                        ], p=0.5))

            if self.dropout:            
                # random erase
                A.CoarseDropout(max_holes=6, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1,
                                fill_value=0, mask_fill_value=None, p=0.25),

            transforms.extend([A.Compose(inner_transforms, p=0.9)])
            
            #transforms.extend([A.Lambda(patch_shuffle.apply, p = 0.5)])
        else:
            transforms = [A.Resize(img_size[1], img_size[0], interpolation=0, p = 1.0)]

        transforms.extend([A.Normalize(mean=0.45, std=0.225, p=1.0),
                            APT.ToTensorV2(p=1.0)])
        
        return A.Compose(transforms)(image=img)["image"]
    
def get_transforms(aug, img_size, geometric = False, dropout = False):

    #patch_shuffle = PatchShuffle([1,2,4,8], [2,4,8,16])
    transformer = Transformer(aug, img_size, geometric, dropout)

    return transformer.tx
