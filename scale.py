import numpy as np
import cv2
import os
import argparse
import multiprocessing as mp
import subprocess
import shutil
from PIL import Image, ImageFilter
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom
from functools import wraps, partial


# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('source_dir', help='path to the directory containing the input DDS files')
parser.add_argument('output_dir', help='path to the directory where the output files will be written')
parser.add_argument('--prescaled_dir', help='path to the directory containing the already scaled images (these will be blended with the pre-scaled input images, if provided)')
parser.add_argument('--temp_dir', default='.tmp', help='path to the temporary directory where temporary images are saved')
parser.add_argument('--interpolation', choices=['auto', 'nearest', 'bilinear', 'hamming', 'bicubic', 'lanczos', 'quadratic', 'catmull', 'mitchell', 'spline36', 'spline64', 'kaiser'], default='auto', help='type of interpolation to use for upscaling (default: %(default)s)')
parser.add_argument('--scale', type=float, default=2.0, help='scale factor for upscaling')
parser.add_argument('--blend', type=float, default=0.5, help='blending percentage for pre-scaled images (only applies if --prescaled_dir is used)')
parser.add_argument('--sharpen', choices=['unsharp_mask', 'cas'], default=None, help='type of sharpening filter to apply after upscaling')
parser.add_argument('--silent', action="store_true", default=False, help='silences warnings and outputs')
parser.add_argument('-p', type=int, default=1, help='the number of processes (forces the process to be silent if p > 1, default: %(default)s)')


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


def pil_to_numpy(func):
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

def scaler(prescale=False, resample=Image.NEAREST, *resize_args, **resize_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(image, scale_factor, *args, **kwargs):
            old_size = image.size
            new_size = tuple(map(lambda x: int(np.round(x * scale_factor)), old_size))
            if prescale:
                image = image.resize(new_size, resample=resample, *resize_args, **resize_kwargs)
            # Call the function with the scaled image size as additional arguments
            result = func(image, scale_factor, *args, old_size=old_size, new_size=new_size, **kwargs)
            return result

        return wrapper
    return decorator

def srgb_conversion(func):
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

def convolve_separable(image, kernel_1d, mode='same'):
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
def quadratic(image, scale_factor, old_size, new_size):
    image = image.resize(old_size, resample=Image.BICUBIC)
    image = image.resize(new_size, resample=Image.BICUBIC)
    return image

@scaler()
@pil_to_numpy
def catmull(image, scale_factor, old_size, new_size):
    def catmullrom_filter(in_arr):
        spline = CubicSpline(np.arange(in_arr.shape[0]), in_arr)
        return spline(np.linspace(0, in_arr.shape[0] - 1, in_arr.shape[0] * 2))
    image = np.apply_along_axis(catmullrom_filter, axis=1, arr=image)
    image = np.apply_along_axis(catmullrom_filter, axis=0, arr=image)
    return image

@scaler(prescale=True, resample=Image.BILINEAR)
@pil_to_numpy
def mitchell(image, scale_factor, old_size, new_size, B=1/3, C=1/3):    
    def mitchell_netravali(x):
        x = abs(x)
        if x > 2:
            return 0
        elif x > 1:
            return (-B - 6*C) * x**3 + (6*B + 30*C) * x**2 + (-12*B - 48*C) * x + (8*B + 24*C)
        else:
            return (12 - 9*B - 6*C) * x**3 + (-18 + 12*B + 6*C) * x**2 + (6 - 2*B)
    
    filter_size = int(2 * scale_factor + 1)
    kernel = [mitchell_netravali(int(np.abs(i - filter_size / 2)) / scale_factor) for i in range(filter_size)]
    kernel = np.array(kernel)
    kernel /= np.sum(kernel)
    image = convolve_separable(image, kernel)

    return image

@scaler()
@pil_to_numpy
def spline36(image, scale_factor, old_size, new_size):
    image = zoom(image, (scale_factor, scale_factor, 1), order=3)
    return image

@scaler()
@pil_to_numpy
def spline64(image, scale_factor, old_size, new_size):
    image = zoom(image, (scale_factor, scale_factor, 1), order=5)
    return image

@scaler(prescale=True, resample=Image.NEAREST)
@pil_to_numpy
def kaiser(image, filter_size=8, beta=2.0, cutoff=0.9):
    filter_size = filter_size + 1 if filter_size % 2 == 0 else filter_size
    filter_size = int(filter_size)
    window = np.kaiser(filter_size, beta)
    filter_coeffs = np.sinc(2 * cutoff * (np.arange(filter_size) - (filter_size - 1) / 2.))
    filter_coeffs *= window
    filter_coeffs /= np.sum(filter_coeffs)
    
    filter_coeffs = kaiser_filter(filter_size)
    image = convolve_separable(image, filter_coeffs)
    
    return image

def unsharp_mask(image, radius=2, amount=1, threshold=0):
    blurred = image.filter(ImageFilter.GaussianBlur(radius))
    mask = Image.eval(image, lambda x: 255 * (x >= blurred.point(lambda i: i - threshold).convert('L')))
    return Image.composite(image, blurred, mask).point(lambda i: i * amount)

@pil_to_numpy
@srgb_conversion
def amd_cas(image, sharpness=0.5, contrast=0.5, adaptive_sharpening=0.5):
    img_array = image.astype(np.float32) / 255.0

    if img_array.shape[2] == 4:  # RGBA image
        rgb_array = img_array[:, :, :3]
        alpha_array = img_array[:, :, 3:]
    else:  # RGB image
        rgb_array = img_array
        alpha_array = None

    yuv = np.dot(rgb_array, np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51498, -0.10001]]))
    y = yuv[:, :, 0]

    blur = np.zeros_like(y)
    cv2.GaussianBlur(y, (0, 0), sigmaX=1.0, sigmaY=1.0, dst=blur)

    local_contrast = y - blur
    sharpening = sharpness * local_contrast

    adjusted_contrast = contrast * (y - 0.5) + 0.5
    sharpened_y = y + sharpening * adjusted_contrast * adaptive_sharpening

    sharpened_yuv = np.dstack((sharpened_y, yuv[:, :, 1], yuv[:, :, 2]))
    sharpened_rgb = np.dot(sharpened_yuv, np.array([[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]))

    if alpha_array is not None:
        sharpened = np.dstack((sharpened_rgb, alpha_array))
    else:
        sharpened = sharpened_rgb

    sharpened = np.clip(sharpened, 0, 1) * 255.0

    return sharpened

def get_default_interpolation(ratio):
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

def upscale_image(image, scale_factor, interpolation=None):
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
    elif interpolation == InterpolationSettings.Kaiser:
        upscaled_image = kaiser(image, scale_factor)
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
            raise ValueError('Invalid interpolation method')

        # Use the standard PIL methods for other interpolation types
        upscaled_image = image.resize((new_width, new_height), resample=method)

    return upscaled_image

def sharpen_image(image, method):
    if method == SharpenSettings.UnsharpMask:
        sharpened_image = unsharp_mask(image)
    elif method == SharpenSettings.CAS:
        sharpened_image = amd_cas(image)
    else:
        raise ValueError('Invalid sharpening method')

    return sharpened_image

def apply_alpha_channel(source_img: Image.Image, dest_img: Image.Image) -> Image.Image:
    # Check size
    if source_img.size != dest_img.size:
        raise ValueError(f'Source and destination images must have the same size: {source_img.size} != {dest_img.size}')

    out_img = dest_img.copy()

    if 'A' not in source_img.getbands():
        return out_img  # Alpha channel not available in images

    alpha_channel = source_img.getchannel('A')
    out_img.putalpha(alpha_channel)

    return out_img


def normalize_channels(image, channels):
    channels = set(c.lower() for c in channels)
    bands = image.split()
    band_names = tuple(b.lower() for b in image.getbands())
    channel_bands = [band for band, name in zip(bands, band_names) if name in channels]
    channel_array = np.array(channel_bands)
    norm = np.sqrt(np.sum(channel_array ** 2, axis=0, keepdims=True))
    normalized_array = np.uint8(channel_array / norm)
    normalized_bands = [(Image.fromarray(normalized_array[i]) if name in channels else band) for i, (band, name) in enumerate(zip(bands, band_names))]
    return Image.merge(image.mode, normalized_bands)

def get_upscale_method(interpolation_arg):
    if interpolation_arg == 'auto':
        interpolation = None
    else:
        interpolation_map = {
            'nearest': InterpolationSettings.Nearest,
            'bilinear': InterpolationSettings.Bilinear,
            'hamming': InterpolationSettings.Hamming,
            'bicubic': InterpolationSettings.Bicubic,
            'lanczos': InterpolationSettings.Lanczos,
            'quadratic': InterpolationSettings.Quadratic,
            'catmull': InterpolationSettings.CatmullRom,
            'mitchell': InterpolationSettings.MitchellRom,
            'spline36': InterpolationSettings.Spline36,
            'spline64': InterpolationSettings.Spline64,
            'kaiser': InterpolationSettings.Kaiser,
        }
        interpolation = interpolation_map[interpolation_arg]

    return interpolation

def get_sharpen_method(sharpen_arg):
    sharpen_map = {
        'unsharp_mask': SharpenSettings.UnsharpMask,
        'cas': SharpenSettings.CAS,
    }
    sharpening = sharpen_map[sharpen_arg]

    return sharpening

def parse_items(lines, *, return_count=False):
    lines = list(filter(len, lines))
    info = {}
    lines_parsed = 0
    while len(lines):
        h = 1
        if ":" in lines[0]:
            item = list(filter(len, (x.strip() for x in lines[0].split(":"))))
            if len(lines) > 1 and any(lines[1].startswith(ws) for ws in ('\t', ' ')):
                ws = lines[1][0]
                w = 1
                for c in lines[1][1:]:
                    if c == ws:
                        w += 1
                    else:
                        break;
                indent = ws * w
                for l in lines[1:]:
                    if l.startswith(indent):
                        h += 1
                    else:
                        break;
                parsed, _ = parse_items([x[w:] for x in lines[1:h]], return_count=True)
                if len(item) == 1 and h:
                    info[item[0]] = parsed
                else:
                    info[item[0]] = (item[1], parsed)
            else:
                if len(item) == 1:
                    info[item[0]] = {}
                else:
                    v = item[1]
                    info[item[0]] = int(v) if v.isnumeric() else v
        else:
            info[lines[0].strip()] = True
        lines_parsed += h
        lines = lines[h:]
    return (info, lines_parsed) if return_count else info

def dict_get(d: dict, path: str):
    path = path.split(".")
    for i, k in enumerate(path):
        if k not in d:
            return None
        d = d[k]
        if isinstance(d, tuple) and len(d) == 2:
            if i + 1 != len(path):
                d = d[1]
    return d

def run_proc(command_string):
    # execute the external program and capture the output
    echo = partial(print, end='')
    with subprocess.Popen(command_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as prc:
        while prc.poll() is None:
            out = prc.stdout.readline()
            echo(out.decode())
        err = prc.stderr.read()
        echo(err.decode())

def get_format_tag(info: dict):
    flags = dict_get(info, "Pixel Format.Flags")[1]
    tag_list = []
    if "DDPF_FOURCC" in flags:
        fourcc = dict_get(info, "Pixel Format.FourCC")
        tag_list.append(fourcc.split()[0].strip("'"))
    else:
        formats = [f.split("DDPF_")[-1] for f in flags.keys()]
        tag_list.extend(formats)
    swizzle = dict_get(info, "Pixel Format.Swizzle")
    if swizzle is not None:
        tag_list.append(swizzle.split()[0].strip("'"))

    tag = "_".join(tag_list)

    return tag


def decompress_and_scale(filename: str, args, config: dict):
    prescaled_ext = (".dds", ".png", ".tiff", ".jpg")
    get_path_w_ext = lambda path_wo_ext: [path_wo_ext + e for e in prescaled_ext if os.path.exists(path_wo_ext + e)][0]

    fl = filename.lower()
    if fl.endswith(".dds"):
        proc = subprocess.Popen(f'nvddsinfo.exe "{os.path.join(args.source_dir, filename)}"', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        info = parse_items(proc.stdout.read().decode().split("\n"))

        f_wo_ext = ".".join(filename.split(".")[:-1])
            
        if args.prescaled_dir:
            path2 = os.path.join(args.prescaled_dir, f_wo_ext)
            try:
                path2 = get_path_w_ext(path2)
            except IndexError:
                if not args.silent:
                    print(f"\"{f_wo_ext}\" not found in prescaled_dir. Skipping...")
                return None

        tag = get_format_tag(info)

        orig_path = os.path.join(args.source_dir, filename)
        temp_subdir = os.path.join(args.temp_dir, tag)
        try:
            os.makedirs(temp_subdir, exist_ok=True)
        except ValueError as e:
            if not args.silent:
                print(e)
            return None

        path = os.path.join(temp_subdir, f_wo_ext + ".png")

        decompress_arg = ""
        if dict_get(info, "Caps.Caps 2.DDSCAPS2_CUBEMAP_ALL_FACES"):
            decompress_arg += " -faces"

        subprocess.run(f"nvdecompress.exe -format png {decompress_arg} \"{orig_path}\" \"{path}\"", stdout=subprocess.DEVNULL)
            
        image = Image.open(path)
        out_image = upscale_image(image, args.scale, interpolation=config["upscale_method"])
        if "sharpen_method" in config:
            out_image = sharpen_image(out_image, method=config["sharpen_method"])

        if args.prescaled_dir:
            image2 = Image.open(path2)
            try:
                image2 = apply_alpha_channel(out_image, image2)
                out_image = Image.blend(out_image, image2, args.blend)
            except ValueError as e:
                if not args.silent:
                    print(e)
                return None

        out_image.save(path, optimize=False)
        return tag


if __name__ == "__main__":
    args = parser.parse_args()
    config = {}

    upscale_method = get_upscale_method(args.interpolation) if args.blend != 1 or args.prescaled_dir is None else InterpolationSettings.Nearest
    config = {
        "upscale_method": upscale_method,
        **config,
    }

    if args.sharpen:
        sharpen_method = get_sharpen_method(args.sharpen)
        config = {
            "sharpen_method": sharpen_method,
            **config,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    fn_ = partial(decompress_and_scale, args=args, config=config)
    if args.p == 1:
        tags = set(fn_(filename) for filename in os.listdir(args.source_dir))
    else:
        args.silent = True
        p_count = args.p if args.p > 0 else mp.cpu_count() - 1
        with mp.Pool(p_count) as p:
            tags = set(p.imap(fn_, os.listdir(args.source_dir), chunksize=5))
    tags.discard(None)


    for tag in tags:
        temp_subdir = os.path.join(args.temp_dir, tag)
        if tag == "DXT5_xGxR":
            compress_str = f"crunch_x64.exe -quiet -renormalize -mipMode Generate -dxtQuality uber -DXT5_xGxR -fileformat dds  -file \"{temp_subdir}\\*\" -outdir \"{args.output_dir}\""
        else:
            if tag == "DXT1":
                format = "bc1"
            elif tag == "DXT3":
                format = "bc2"
            elif tag == "DXT5":
                format = "bc3"
            # elif tag == "RGB" or tag == "RGB_ALPHAPIXELS":
            #    format = "rgb"
            else:
                if not args.silent:
                    print(f"Input texture format \"{tag}\" not supported yet!")
                continue;
            compress_str = f"nvbatchcompress.exe -silent -{format} -mipfilter kaiser -production \"{temp_subdir}\" \"{args.output_dir}\""

        run_proc(compress_str)

        shutil.rmtree(os.path.abspath(temp_subdir))

    try:
        os.rmdir(args.temp_dir)
    except OSError as e:
        if not args.silent:
            print(e)
