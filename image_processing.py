import numpy as np
import cv2
import os
import argparse
from PIL import Image, ImageFilter
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom
from functools import reduce, wraps, partial
from typing import Optional, Union

from utils.file_utils import scan_dir
from utils.image_utils import load_image
from utils.proc_utils import multiproc

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "output_dir",
    help="path to the directory where the output files will be written",
)
parser.add_argument(
    "source_dir",
    nargs="+",
    help="path to the directory containing PNG files",
)
parser.add_argument(
    "--blend",
    type=float,
    default=0.5,
    help="blending percentage for pre-scaled images (only applies if --source2_dir is provided)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=2.0,
    help="scale factor for upscaling",
)
parser.add_argument(
    "--interpolation",
    choices=[
        "auto",
        "nearest",
        "bilinear",
        "hamming",
        "bicubic",
        "lanczos",
        "quadratic",
        "catmull",
        "mitchell",
        "spline36",
        "spline64",
    ],
    default="auto",
    help="type of interpolation to use for upscaling (default: %(default)s)",
)
parser.add_argument(
    "--sharpen",
    choices=["unsharp_mask", "cas"],
    default=None,
    help="type of sharpening filter to apply after upscaling",
)
parser.add_argument(
    "--recurse",
    action="store_true",
    default=False,
    help="recurse down the input directories",
)
parser.add_argument(
    "--silent",
    action="store_true",
    default=False,
    help="silences warnings and outputs",
)
parser.add_argument(
    "--skip",
    action="store_true",
    default=False,
    help="skips overwriting temporary files.",
)
parser.add_argument(
    "-p",
    type=int,
    default=1,
    help="the number of processes (forces the process to be silent if p > 1, default: %(default)s)",
)


class InterpolationSettings:
    Nearest = 0
    Bilinear = 1
    Hamming = 2
    Bicubic = 3
    Lanczos = 4
    Quadratic = 5
    CatmullRom = 6
    MitchellRom = 7
    Spline36 = 8
    Spline64 = 9
    Kaiser = 10


class SharpenSettings:
    UnsharpMask = 0
    CAS = 1


def pil_to_numpy(func) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert PIL image to numpy array
        if isinstance(args[0], Image.Image):
            args = list(args)
            args[0] = np.array(args[0])
        # Call the function
        result = func(*args, **kwargs)
        # Convert result back to PIL image if necessary
        if isinstance(result, np.ndarray):
            result = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
        return result

    return wrapper


def scaler(
    prescale=False, resample=Image.NEAREST, *resize_args, **resize_kwargs
) -> callable:
    def decorator(func):
        @wraps(func)
        def wrapper(image, scale_factor, *args, **kwargs):
            old_size = image.size
            new_size = tuple(map(lambda x: int(np.round(x * scale_factor)), old_size))
            if prescale:
                image = image.resize(
                    new_size, resample=resample, *resize_args, **resize_kwargs
                )
            # Call the function with the scaled image size as additional arguments
            result = func(
                image,
                scale_factor,
                *args,
                old_size=old_size,
                new_size=new_size,
                **kwargs,
            )
            return result

        return wrapper

    return decorator


def srgb_conversion(func) -> callable:
    @wraps(func)
    def wrapper(image, *args, srgb=False, **kwargs):

        def correct_gamma(image, gamma=2.2):
            image = image.astype(np.float32)
            alpha = None
            if image.shape[-1] == 4:
                alpha = image[:, :, 3:]
                image = image[:, :, :3]
            image = np.power(image / 255.0, gamma) * 255.0
            if alpha is not None:
                image = np.concatenate([image, alpha], axis=-1)
            return image

        if srgb:
            image = correct_gamma(image, 2.2)
        else:
            image = image.astype(np.float32)

        result = func(image, *args, **kwargs)

        if srgb:
            result = correct_gamma(result, 1.0 / 2.2)

        return result

    return wrapper


def convolve_separable(image, kernel_1d, mode="same") -> np.ndarray:
    filtered = image.copy()
    if image.ndim > 2:
        for i in range(image.shape[0]):
            for k in range(image.shape[2]):
                filtered[i, :, k] = np.convolve(filtered[i, :, k], kernel_1d, mode=mode)
        for i in range(image.shape[1]):
            for k in range(image.shape[2]):
                filtered[:, i, k] = np.convolve(filtered[:, i, k], kernel_1d, mode=mode)
    else:
        for i in range(image.shape[0]):
            filtered[i, :] = np.convolve(filtered[i, :], kernel_1d, mode=mode)
        for i in range(image.shape[1]):
            filtered[:, i] = np.convolve(filtered[:, i], kernel_1d, mode=mode)

    return filtered


@scaler(prescale=True)
def quadratic(image, scale_factor, old_size, new_size) -> Image.Image:
    image = image.resize(old_size, resample=Image.BICUBIC)
    image = image.resize(new_size, resample=Image.BICUBIC)
    return image


@scaler()
@pil_to_numpy
def catmull(image, scale_factor, old_size, new_size) -> np.ndarray:
    def catmullrom_filter(in_arr):
        spline = CubicSpline(np.arange(in_arr.shape[0]), in_arr)
        return spline(np.linspace(0, in_arr.shape[0] - 1, in_arr.shape[0] * 2))

    image = np.apply_along_axis(catmullrom_filter, axis=1, arr=image)
    image = np.apply_along_axis(catmullrom_filter, axis=0, arr=image)
    return image


@scaler(prescale=True, resample=Image.BILINEAR)
@pil_to_numpy
def mitchell(image, scale_factor, old_size, new_size, B=1 / 3, C=1 / 3) -> np.ndarray:
    def mitchell_netravali(x):
        x = abs(x)
        if x > 2:
            return 0
        elif x > 1:
            return (
                (-B - 6 * C) * x**3
                + (6 * B + 30 * C) * x**2
                + (-12 * B - 48 * C) * x
                + (8 * B + 24 * C)
            )
        else:
            return (
                (12 - 9 * B - 6 * C) * x**3
                + (-18 + 12 * B + 6 * C) * x**2
                + (6 - 2 * B)
            )

    filter_size = int(2 * scale_factor + 1)
    kernel = [
        mitchell_netravali(int(np.abs(i - filter_size / 2)) / scale_factor)
        for i in range(filter_size)
    ]
    kernel = np.array(kernel)
    kernel /= np.sum(kernel)
    image = convolve_separable(image, kernel)

    return image


@scaler()
@pil_to_numpy
def spline36(image, scale_factor, old_size, new_size) -> np.ndarray:
    image = zoom(image, (scale_factor, scale_factor, 1), order=3)
    return image


@scaler()
@pil_to_numpy
def spline64(image, scale_factor, old_size, new_size) -> np.ndarray:
    image = zoom(image, (scale_factor, scale_factor, 1), order=5)
    return image


def unsharp_mask(image, radius=2, amount=1, threshold=0) -> Image.Image:
    blurred = image.filter(ImageFilter.GaussianBlur(radius))
    mask = Image.eval(
        image,
        lambda x: 255 * (x >= blurred.point(lambda i: i - threshold).convert("L")),
    )
    return Image.composite(image, blurred, mask).point(lambda i: i * amount)


@pil_to_numpy
@srgb_conversion
def amd_cas(
    image, sharpness=0.5, contrast=0.5, adaptive_sharpening=0.5, sharpen_alpha=True
) -> np.ndarray:
    def sharpen_mono(y):
        blur = np.zeros_like(y)
        cv2.GaussianBlur(y, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=blur)

        local_contrast = y - blur
        sharpening = sharpness * local_contrast

        adjusted_contrast = contrast * (y - 0.5) + 0.5
        sharpened_y = y + sharpening * adjusted_contrast * adaptive_sharpening
        return sharpened_y

    img_array = image.astype(np.float32) / 255.0

    has_alpha = img_array.shape[2] == 4
    if has_alpha:  # RGBA image
        rgb_array = img_array[:, :, :3]
        alpha_array = img_array[:, :, 3]
    else:  # RGB image
        rgb_array = img_array
        alpha_array = None

    yuv = np.dot(
        rgb_array,
        np.array(
            [
                [0.299, 0.587, 0.114],
                [-0.14713, -0.28886, 0.436],
                [0.615, -0.51498, -0.10001],
            ]
        ),
    )
    y = yuv[:, :, 0]

    sharpened_y = sharpen_mono(y)
    if has_alpha and sharpen_alpha:
        alpha_array = sharpen_mono(alpha_array)

    sharpened_yuv = np.dstack((sharpened_y, yuv[:, :, 1], yuv[:, :, 2]))
    sharpened_rgb = np.dot(
        sharpened_yuv,
        np.array([[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]),
    )

    if has_alpha:
        sharpened = np.dstack((sharpened_rgb, np.expand_dims(alpha_array, axis=2)))
    else:
        sharpened = sharpened_rgb

    sharpened = np.clip(sharpened, 0, 1) * 255.0

    return sharpened


def get_default_interpolation(ratio) -> InterpolationSettings:
    if ratio == 1.0:
        return InterpolationSettings.Bilinear
    elif ratio < 0.5:
        return InterpolationSettings.Lanczos
    elif ratio > 16.0:
        return InterpolationSettings.Quadratic
    elif ratio > 4.0:
        return InterpolationSettings.CatmullRom
    else:
        return InterpolationSettings.Spline36


def upscale_image(image, scale_factor, interpolation=None) -> Image.Image:
    # Compute the new dimensions of the upscaled image
    width, height = image.size
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)

    # Select the interpolation method to use
    if interpolation is None:
        interpolation = get_default_interpolation(scale_factor)

    # Upscale the image using the selected interpolation method
    if interpolation == InterpolationSettings.Quadratic:
        upscaled_image = quadratic(image, scale_factor)
    elif interpolation == InterpolationSettings.CatmullRom:
        upscaled_image = catmull(image, scale_factor)
    elif interpolation == InterpolationSettings.MitchellRom:
        upscaled_image = mitchell(image, scale_factor)
    elif interpolation == InterpolationSettings.Spline36:
        upscaled_image = spline36(image, scale_factor)
    elif interpolation == InterpolationSettings.Spline64:
        upscaled_image = spline64(image, scale_factor)
    else:
        if interpolation == InterpolationSettings.Nearest:
            method = Image.NEAREST
        elif interpolation == InterpolationSettings.Bilinear:
            method = Image.BILINEAR
        elif interpolation == InterpolationSettings.Hamming:
            method = Image.Hamming
        elif interpolation == InterpolationSettings.Bicubic:
            method = Image.BICUBIC
        elif interpolation == InterpolationSettings.Lanczos:
            method = Image.LANCZOS
        else:
            raise ValueError("Invalid interpolation method")

        # Use the standard PIL methods for other interpolation types
        upscaled_image = image.resize((new_width, new_height), resample=method)

    return upscaled_image


def sharpen_image(image, method) -> Image.Image:
    if method == SharpenSettings.UnsharpMask:
        sharpened_image = unsharp_mask(image)
    elif method == SharpenSettings.CAS:
        sharpened_image = amd_cas(image)
    else:
        raise ValueError("Invalid sharpening method")

    return sharpened_image


def normalize_channels(image, channels) -> Image.Image:
    channels = set(c.lower() for c in channels)
    bands = image.split()
    band_names = tuple(b.lower() for b in image.getbands())
    channel_bands = [band for band, name in zip(bands, band_names) if name in channels]
    channel_array = np.array(channel_bands)
    norm = np.sqrt(np.sum(channel_array**2, axis=0, keepdims=True))
    normalized_array = np.uint8(channel_array / norm)
    normalized_bands = [
        (Image.fromarray(normalized_array[i]) if name in channels else band)
        for i, (band, name) in enumerate(zip(bands, band_names))
    ]
    return Image.merge(image.mode, normalized_bands)


def get_upscale_method(interpolation_arg) -> InterpolationSettings:
    if interpolation_arg == "auto":
        interpolation = None
    else:
        interpolation_map = {
            "nearest": InterpolationSettings.Nearest,
            "bilinear": InterpolationSettings.Bilinear,
            "hamming": InterpolationSettings.Hamming,
            "bicubic": InterpolationSettings.Bicubic,
            "lanczos": InterpolationSettings.Lanczos,
            "quadratic": InterpolationSettings.Quadratic,
            "catmull": InterpolationSettings.CatmullRom,
            "mitchell": InterpolationSettings.MitchellRom,
            "spline36": InterpolationSettings.Spline36,
            "spline64": InterpolationSettings.Spline64,
            "kaiser": InterpolationSettings.Kaiser,
        }
        interpolation = interpolation_map[interpolation_arg]

    return interpolation


def get_sharpen_method(sharpen_arg) -> SharpenSettings:
    sharpen_map = {
        "unsharp_mask": SharpenSettings.UnsharpMask,
        "cas": SharpenSettings.CAS,
    }
    sharpening = sharpen_map[sharpen_arg]

    return sharpening


def blend(
    img1: Union[Image.Image, str],
    img2: Union[Image.Image, str],
    blend_rate: float = 0.5,
) -> Image.Image:
    img1 = load_image(img1) if isinstance(img1, str) else img1
    img2 = load_image(img2) if isinstance(img2, str) else img2

    has_alpha1 = "A" in img1.getbands()
    has_alpha2 = "A" in img2.getbands()
    if has_alpha1 and not has_alpha2:
        img2.putalpha(img1.getchannel("A"))
    elif has_alpha2 and not has_alpha1:
        img1.putalpha(img2.getchannel("A"))

    out_image = Image.blend(img1, img2, blend_rate)

    return out_image


def process_image(
    img: Union[Image.Image, str],
    scale: float,
    interpolation: InterpolationSettings,
    sharpen: Optional[SharpenSettings] = None,
) -> Image.Image:
    image = load_image(img) if isinstance(img, str) else img
    out_image = image
    if scale != 1.0:
        out_image = upscale_image(image, scale, interpolation=interpolation)
    if sharpen is not None:
        out_image = sharpen_image(out_image, method=sharpen)
    return out_image


def process_file(path: str, args, config: dict) -> None:
    relpath = os.path.relpath(path, args.source_dir[0])
    dest_path = os.path.join(args.output_dir, relpath)
    paths_ = [
        p for d in args.source_dir if os.path.exists(p := os.path.join(d, relpath))
    ]

    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    except ValueError as e:
        if not args.silent:
            print(e)
        return None

    imgs = [load_image(p) for p in paths_]
    min_size = np.min([img.size for img in imgs], axis=0)
    scaled = [
        process_image(
            img,
            args.scale / (img.size[0] / min_size[0]),  # scale to max size
            config["upscale_method"],
            config["sharpen_method"],
        )
        for img in imgs
    ]
    _, blended = reduce(
        lambda x, y: (
            (zi := y[0] + 1),
            blend(x[1], y[1], blend_rate=1 - args.blend * 2 / zi),
        ),
        enumerate(scaled),
    )
    blended.save(dest_path, optimize=False)


def main():
    args = parser.parse_args()
    config = {}

    upscale_method = (
        get_upscale_method(args.interpolation)
        if args.blend != 1 or args.prescaled_dir is None
        else InterpolationSettings.Nearest
    )
    config = {
        "upscale_method": upscale_method,
        **config,
    }

    if args.sharpen:
        sharpen_method = get_sharpen_method(args.sharpen)
    else:
        sharpen_method = None
    config = {
        "sharpen_method": sharpen_method,
        **config,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    fn_ = partial(process_file, args=args, config=config)
    multiproc(fn_, scan_dir(args.source_dir[0], recurse=args.recurse), args.p)


if __name__ == "__main__":
    main()
