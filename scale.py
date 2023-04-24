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
from itertools import product
from typing import Optional, Union


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
parser.add_argument('--recurse', action="store_true", default=False, help='recurse down the input directories')
parser.add_argument('--skip', action="store_true", default=False, help='skips overwriting temporary files.')
parser.add_argument('--keep', action="store_true", default=False, help='keeps temporary files after operation.')
parser.add_argument('--no_compress', action="store_true", default=False, help='do not compress the processed files.')
parser.add_argument('-p', type=int, default=1, help='the number of processes (forces the process to be silent if p > 1, default: %(default)s)')

is_tool = lambda name: shutil.which(name) is not None

# HAS_TEXCONV = is_tool("texconv")
# HAS_TEXASSEMBLE = is_tool("texassemble")
HAS_NVTT = is_tool("nvcompress")
HAS_ATI_MAPGEN = is_tool("CubeMapGen")
# HAS_TEXASSEMBLE = is_tool("texassemble")
HAS_CRUNCH = is_tool("crunch_x64")


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

def scan_dir(path: str, recurse: bool = False, dirs: bool = False):
    for entry in os.scandir(path):
        if entry.is_file():
            if not dirs:
                yield entry.path
        else:
            if dirs and len(next(os.walk(entry))[2]):
                yield entry.path
            yield from scan_dir(entry.path, recurse=recurse, dirs=dirs)

def flatten_dir(directory):
    index = 0
    file_paths = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            new_filename = f"{index}_{filename}"
            index += 1
            src_path = os.path.join(root, filename)
            file_paths[new_filename] =  os.path.relpath(root, directory)
            dest_path = os.path.join(directory, new_filename)
            shutil.move(src_path, dest_path)
    for root, dirs, _ in os.walk(directory, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass
    return file_paths

def unflatten_dir(directory, file_paths):
    for filename, orig_dir in file_paths.items():
        orig_filename = filename.split("_", 1)[1]
        dest_dir = os.path.join(directory, orig_dir)
        dest_path = os.path.join(dest_dir, orig_filename)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(os.path.join(directory, filename), dest_path)

def run_proc(command: str, *, silent: bool = False):
    # execute the external program and capture the output
    echo = partial(print, end='')
    stdout = subprocess.DEVNULL if silent else subprocess.PIPE
    with subprocess.Popen(command, stdout=stdout, stderr=subprocess.PIPE) as prc:
        if not silent:
            while prc.poll() is None:
                out = prc.stdout.readline()
                echo(out.decode())
        err = prc.stderr.read()
        echo(err.decode())

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
def amd_cas(image, sharpness=0.5, contrast=0.5, adaptive_sharpening=0.5, sharpen_alpha=True):
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

    yuv = np.dot(rgb_array, np.array([[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51498, -0.10001]]))
    y = yuv[:, :, 0]

    sharpened_y = sharpen_mono(y)
    if has_alpha and sharpen_alpha:
        alpha_array = sharpen_mono(alpha_array)

    sharpened_yuv = np.dstack((sharpened_y, yuv[:, :, 1], yuv[:, :, 2]))
    sharpened_rgb = np.dot(sharpened_yuv, np.array([[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]))

    if has_alpha:
        sharpened = np.dstack((sharpened_rgb, np.expand_dims(alpha_array, axis=2)))
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

def get_dds_info(path: str):
    proc = subprocess.Popen(f'nvddsinfo "{path}"', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    info = parse_items(proc.stdout.read().decode().split("\n"))
    return info

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
    if dict_get(info, "Caps.Caps 2.DDSCAPS2_CUBEMAP_ALL_FACES"):
        tag_list.append("CUBEMAP")

    tag = "_".join(tag_list)

    return tag

def decompress(orig_path: str, target_path: str, tag: str, *, silent: bool = False):
    decompress_args = []
    is_cubemap = (tag.split("_")[-1] == "CUBEMAP")
    if is_cubemap:
        tag = "_".join(tag.split("_")[:-1])
        decompress_args.append("-faces")
    if tag == "DXT5_xGxR":
        decompress_args.append("-forcenormal")
    d_arg_str = " ".join(decompress_args)
    
    run_proc(f"nvdecompress -format png {d_arg_str} \"{orig_path}\" \"{target_path}\"", silent=silent)

def process_image(img: Union[Image.Image, str], scale: float, interpolation: InterpolationSettings, sharpen: Optional[SharpenSettings] = None):
    image = Image.open(img) if isinstance(img, str) else img
    out_image = image
    if scale != 1.0:
        out_image = upscale_image(image, scale, interpolation=interpolation)
    if sharpen is not None:
        out_image = sharpen_image(out_image, method=sharpen)
    return out_image

def blend(img1: Union[Image.Image, str], img2: Union[Image.Image, str], blend_rate: float = 0.5):
    image1 = Image.open(img1) if isinstance(img1, str) else img1
    image2 = Image.open(img2) if isinstance(img2, str) else img2
    image2_w_alpha = apply_alpha_channel(image1, image2)
    out_image = Image.blend(image1, image2_w_alpha, args.blend)

    return out_image

def get_path_w_ext(base_path, extensions):
    if isinstance(extensions, str):
        extensions = (extensions,)
    for extension in extensions:
        path_with_extension = f"{base_path}.{extension}"
        if os.path.exists(path_with_extension):
            return path_with_extension
    raise FileNotFoundError(f"No existing file found for \"{base_path}\" with any of the given extensions: {extensions}")

def process_file(path: str, args, config: dict):
    if not os.path.splitext(path)[1].lower() == ".dds":
        return None

    relpath = os.path.relpath(path, args.source_dir)
    info = get_dds_info(path)
    is_cubemap = dict_get(info, "Caps.Caps 2.DDSCAPS2_CUBEMAP_ALL_FACES")
    tag = get_format_tag(info)
    
    path2 = None
    if args.prescaled_dir and not is_cubemap:
        path2 = os.path.join(args.prescaled_dir, relpath)
        try:
            path2 = get_path_w_ext(path2, ("dds", "png", "tiff", "jpg"))
        except FileNotFoundError:
            if not args.silent:
                print(f"\"{os.path.splitext(relpath)[0]}\" not found in prescaled_dir. Skipping...")
            return None

    orig_path = os.path.join(args.source_dir, relpath)
    temp_subdir = os.path.join(args.temp_dir, tag)
    path = os.path.join(temp_subdir, os.path.splitext(relpath)[0] + ".png")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except ValueError as e:
        if not args.silent:
            print(e)
        return None

    if args.skip and os.path.exists(path):
        return tag

    decompress(orig_path, path, tag, silent=args.silent)

    try:
        process_file_(path, path2, args=args, tag=tag, config=config)
    except FileNotFoundError:
        if not args.silent:
            print(f"\"{os.path.splitext(relpath)[0]}\" not found in prescaled_dir. Skipping...")
        return None

    return tag

def process_file_(path: str, path2: Optional[str], *, args, tag: str, config: dict):
    is_cubemap = tag.split("_")[-1] == "CUBEMAP"
    if is_cubemap:
        path_wo_ext = os.path.splitext(path)[0]
        path2_wo_ext = os.path.splitext(path)[0] if path2 else None
        raw_tag = "_".join(tag.split("_")[:-1])
        temp_subdir = os.path.join(args.temp_dir, raw_tag)
        get_face_addr = lambda path_wo_ext, i: get_path_w_ext(path_wo_ext + f"_face{i}", "png") if path_wo_ext else None
        for p1, p2 in ((get_face_addr(path_wo_ext, i), get_face_addr(path2_wo_ext, i)) for i in range(6)):
            process_file_(p1, p2, args=args, tag=raw_tag, config=config)
    else:
        out_image = process_image(path, args.scale, interpolation=config["upscale_method"], sharpen=config["sharpen_method"])
        if path2:
            try:
                out_image = blend(out_image, path2, args.blend)
            except ValueError as e:
                if not args.silent:
                    print(e)
                return;
        out_image.save(path, optimize=False)

def transform_op(entity_addr: str, command: str, source_dir: str, target_dir: str, *, silent: bool = False):
    is_file = os.path.isfile(entity_addr)
    rel_p_ = os.path.relpath(entity_addr, source_dir)
    tp_ = os.path.join(target_dir, rel_p_)
    dirname = os.path.dirname(tp_) if is_file else tp_
    os.makedirs(dirname, exist_ok=True)
    command_str = command.format(in_path=entity_addr, out_path=tp_)
    run_proc(command_str, silent=silent)


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
    else:
        sharpen_method = None
    config = {
        "sharpen_method": sharpen_method,
        **config,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    try:

        fn_ = partial(process_file, args=args, config=config)
        if args.p == 1:
            tags = set(map(fn_, scan_dir(args.source_dir, recurse=args.recurse)))
        else:
            p_count = args.p if args.p > 0 else mp.cpu_count() - 1
            with mp.Pool(p_count) as p:
                tags = set(p.imap_unordered(fn_, scan_dir(args.source_dir, recurse=args.recurse), chunksize=8))
        tags.discard(None)
        if not args.no_compress:
            tags.update(tag.replace("_CUBEMAP", "") for tag in tags if tag.endswith("_CUBEMAP"))

        for tag in sorted(tags)[::-1]:
            try:
                temp_subdir = os.path.join(args.temp_dir, tag)
                is_cubemap = tag.endswith("_CUBEMAP")
                if args.no_compress:
                    temp_outdir = os.path.join(args.output_dir, os.path.relpath(temp_subdir, args.temp_dir))
                    run_proc(f"robocopy /s /move /njh /njs /ndl \"{temp_subdir}\" \"{temp_outdir}\"", silent=args.silent)
                else:
                    if is_cubemap:
                        t = tag.replace("_CUBEMAP", "")
                        target_subdir = os.path.join(args.temp_dir, t)
                        os.makedirs(target_subdir, exist_ok=True)
                        mapgen_str = "CubeMapGen -exit -consoleErrorOutput -exportCubeDDS -exportPixelFormat:A8R8G8B8 -exportFilename:\"{out_path}.dds\""
                        for i, (axis, side) in enumerate(product(('X', 'Y', 'Z'), ("Pos", "Neg"))):
                            mapgen_str +=  f" -importFace{axis}{side}:\"" + "{in_path}_face" + str(i) + ".png\""

                        fn_ = partial(transform_op, command=mapgen_str, source_dir=temp_subdir, target_dir=target_subdir, silent=args.silent)
                    
                        unique_files = frozenset(f_[:-10] for f_ in scan_dir(temp_subdir, recurse=args.recurse))
                        if args.p == 1:
                            for f in unique_files:
                                fn_(f)
                        else:
                            with mp.Pool(p_count) as p:
                                for _ in p.imap_unordered(fn_, unique_files, chunksize=3):
                                    pass
                    else:
                        batch = False
                        if tag == "DXT5_xGxR":
                            compress_str = "crunch_x64 -quiet -renormalize -mipMode Generate -dxtQuality uber -DXT5_xGxR -fileformat dds  -file \"{in_path}\" -outdir \"{out_path}\""
                            batch = True
                        else:
                            if tag == "DXT1":
                                format = "bc1"
                            elif tag == "DXT3":
                                format = "bc2"
                            elif tag == "DXT5":
                                format = "bc3"
                            elif tag == "RGB":
                                format = "rgb"
                            elif tag == "RGB_ALPHAPIXELS":
                                format = "rgb -alpha"
                            elif tag == "LUMINANCE":
                                format = "lumi"
                            else:
                                if not args.silent:
                                    print(f"Input texture format \"{t}\" not supported yet!")
                                continue;
                            compress_str = f"nvbatchcompress -silent -{format} -mipfilter kaiser -production " + "\"{in_path}\" \"{out_path}\""
                            batch = True

                        temp_outdir = os.path.join(temp_subdir, "compress_out_")
                        os.makedirs(temp_outdir, exist_ok=True)
                        fn_ = partial(transform_op, command=compress_str, source_dir=temp_subdir, target_dir=temp_outdir)

                        if batch:
                            if args.recurse:
                                file_paths = flatten_dir(temp_subdir)
                            fn_(temp_subdir)
                            if args.recurse:
                                file_paths_dds = dict(zip(map(lambda x: x[:-4] + ".dds", file_paths.keys()), file_paths.values()))
                                unflatten_dir(temp_outdir, file_paths_dds)
                                unflatten_dir(temp_subdir, file_paths)
                        else:
                            subdirs_g = scan_dir(temp_subdir, recurse=args.recurse, dirs=False)
                            if args.p == 1:
                                for d in subdirs_g:
                                    fn_(d)
                            else:
                                with mp.Pool(p_count) as p:
                                    for _ in p.imap_unordered(fn_, subdirs_g, chunksize=4):
                                        pass

                    run_proc(f"robocopy /s /move /njh /njs /ndl \"{temp_outdir}\" \"{args.output_dir}\"", silent=args.silent)
            
            finally:
                if not args.keep:
                    shutil.rmtree(temp_subdir, ignore_errors=True)
    finally:
        try:
            os.rmdir(args.temp_dir)
        except OSError as e:
            if not args.silent:
                print(e)
