"""
preprocessing.py — All image preprocessing transforms for mammography and ultrasound.

Mammography pipeline:
    Grayscale → Resize 224×224 → CLAHE → 3-channel repeat → Augmentation → Normalize

Ultrasound pipeline (Albumentations-based):
    Grayscale → Albumentations augmentation (RandomResizedCrop, flips,
    rotation, noise, elastic transform) → Normalize → 3-channel repeat
"""

import cv2
import numpy as np
import pywt
from PIL import Image
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── ImageNet normalization constants ───────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CLAHETransform:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    to a single-channel numpy image (uint8).

    Args:
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram equalisation.
    """

    def __init__(self, clip_limit: float = 2.0,
                 tile_grid_size: tuple = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to a uint8 grayscale image.

        Args:
            image: (H, W) uint8 numpy array.

        Returns:
            CLAHE-enhanced (H, W) uint8 numpy array.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )
        return clahe.apply(image)


class WaveletDenoise:
    """Wavelet-based speckle reduction for ultrasound images.

    Uses a single-level 2-D DWT with the 'db2' wavelet and applies
    universal soft-thresholding to the detail sub-bands (cH, cV, cD).

    Returns a 3-channel composite:
        channel 0 = wavelet-denoised image
        channel 1 = original grayscale image
        channel 2 = wavelet-denoised image  (repeat of ch0)

    This preserves original texture alongside the denoised version
    while keeping channel count = 3 for ImageNet-pretrained backbones.
    """

    def __init__(self, wavelet: str = "db2"):
        self.wavelet = wavelet

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising and build 3-channel composite.

        Args:
            image: (H, W) uint8 grayscale numpy array.

        Returns:
            (H, W, 3) uint8 numpy array.
        """
        img_float = image.astype(np.float64)

        # Single-level 2-D DWT
        cA, (cH, cV, cD) = pywt.dwt2(img_float, self.wavelet)

        # Universal soft-threshold (VisuShrink)
        sigma = np.median(np.abs(cD)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(max(image.size, 1)))

        # Soft thresholding on detail coefficients
        cH = pywt.threshold(cH, value=threshold, mode="soft")
        cV = pywt.threshold(cV, value=threshold, mode="soft")
        cD = pywt.threshold(cD, value=threshold, mode="soft")

        # Reconstruct
        denoised = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)

        # Clip and convert back to uint8
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)

        # Ensure sizes match (DWT may pad by 1 pixel)
        h, w = image.shape[:2]
        denoised = denoised[:h, :w]

        # Build 3-channel composite
        composite = np.stack([denoised, image, denoised], axis=-1)
        return composite


class MammographyPipeline:
    """Callable class for mammography preprocessing pipeline."""
    def __init__(self, image_size: int, is_training: bool):
        self.image_size = image_size
        self.clahe = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
        
        aug_transforms = []
        if is_training:
            aug_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        aug_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.pipeline = transforms.Compose(aug_transforms)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        gray = pil_img.convert("L")
        gray = gray.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(gray, dtype=np.uint8)
        arr = self.clahe(arr)
        arr_3ch = np.stack([arr, arr, arr], axis=-1)
        pil_3ch = Image.fromarray(arr_3ch, mode="RGB")
        return self.pipeline(pil_3ch)

def get_mammography_transforms(image_size: int = 224,
                               is_training: bool = True):
    """Build the complete mammography preprocessing pipeline.

    Pipeline:
        1. Load as grayscale PIL → numpy
        2. Resize to (image_size, image_size)
        3. CLAHE contrast enhancement
        4. Repeat grayscale → 3-channel RGB
        5. Training augmentations (flip, rotation)
        6. ImageNet normalisation
    """
    return MammographyPipeline(image_size, is_training)


class UltrasoundPipeline:
    """Callable class for ultrasound preprocessing pipeline.

    Uses Albumentations-based augmentation following the approach from
    the breast cancer detection notebook.  No wavelet denoising — instead
    uses heavier augmentations to prevent overfitting on the small BUSI
    dataset (~780 images).
    """

    def __init__(self, image_size: int, is_training: bool):
        self.image_size = image_size

        transforms_list = []
        if is_training:
            transforms_list.extend([
                A.RandomResizedCrop(
                    height=image_size, width=image_size,
                    scale=(0.7, 1.0), p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3,
                ),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
                A.ElasticTransform(alpha=30, sigma=5, p=0.2),
            ])
        else:
            transforms_list.append(
                A.Resize(height=image_size, width=image_size),
            )

        transforms_list.extend([
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ])

        self.pipeline = A.Compose(transforms_list)

    def __call__(self, pil_img: Image.Image) -> torch.Tensor:
        """Preprocess an ultrasound image.

        Args:
            pil_img: PIL Image (any mode — will be converted to grayscale).

        Returns:
            Float tensor of shape (3, H, W) suitable for pretrained
            backbones (3 identical grayscale channels).
        """
        # Convert to grayscale numpy array (as in the notebook)
        gray = pil_img.convert("L")
        arr = np.array(gray, dtype=np.uint8)

        # Apply Albumentations pipeline → (1, H, W) tensor
        augmented = self.pipeline(image=arr)
        tensor = augmented["image"]          # (1, H, W)

        # Repeat single channel → 3 channels for pretrained backbone
        tensor = tensor.repeat(3, 1, 1)      # (3, H, W)

        return tensor


def get_ultrasound_transforms(image_size: int = 224,
                              is_training: bool = True):
    """Build the complete ultrasound preprocessing pipeline (Albumentations).

    Pipeline:
        1. Convert to grayscale numpy array
        2. Training: RandomResizedCrop + augmentations (flips, rotation,
           brightness/contrast, Gaussian noise, elastic deformation)
           Validation: simple Resize
        3. Normalize (mean=0.5, std=0.5)
        4. Repeat to 3 channels for pretrained backbone
    """
    return UltrasoundPipeline(image_size, is_training)
