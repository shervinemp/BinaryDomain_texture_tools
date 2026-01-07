import os
import argparse
from functools import partial
from itertools import product
import shutil
from PIL import Image

from utils.file_utils import flatten_dir, hardlink_files, is_path_var, scan_dir, unravel_dir
from utils.image_utils import get_ddsinfo, get_format_tag
from utils.proc_utils import multiproc, prevent_sleep, run_proc, transform_op

TEMP_DIR = os.path.abspath(".tmp")

parser = argparse.ArgumentParser()
parser.add_argument(
    "operation",
    choices=["compress", "decompress"],
    default="decompress",
    help='the operation to perform. "compress" converts images to the compressed DDS format. "decompress" converts DDS files to PNG (default: %(default)s)',
)
parser.add_argument(
    "source_dir", help="path to the directory containing the input DDS/PNG files"
)
parser.add_argument(
    "output_dir",
    default=".",
    help="path to the directory where the output files will be written (default: current directory)",
)
parser.add_argument(
    "-p",
    type=int,
    default=1,
    help="the number of processes (silent for <> 1), default: %(default)s)",
)
parser.add_argument(
    "--recurse",
    action="store_true",
    default=True,
    help="recurse down the input directories",
)
parser.add_argument(
    "--silent", action="store_true", default=False, help="silences warnings and outputs"
)
parser.add_argument(
    "--skip",
    action="store_true",
    default=False,
    help="skips overwriting files.",
)

def decompress_op(orig_path: str, target_path: str, tag: str, *, silent: bool = False) -> None:
    decompress_args = []
    is_cubemap = tag.split("_")[-1] == "CUBEMAP"
    if is_cubemap:
        decompress_args.append("-faces")
        tag = tag[: tag.rfind("_")]
    if tag == "DXT5_xGxR":
        decompress_args.append("-forcenormal")
    d_arg_str = " ".join(decompress_args)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    abs_p = os.path.abspath
    run_proc(
        f'nvdecompress -format png {d_arg_str} "{abs_p(orig_path)}" "{abs_p(target_path)}"',
        silent=silent,
    )

def decompress(path: str, args) -> None:
    if os.path.splitext(path)[1].lower() != ".dds":
        return

    relpath = os.path.relpath(path, args.source_dir)
    info = get_ddsinfo(path)
    tag = get_format_tag(info)

    dest_subdir = os.path.join(args.output_dir, tag)
    dest_path = os.path.join(dest_subdir, os.path.splitext(relpath)[0] + ".png")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if args.skip and os.path.exists(path):
        print(f'"{os.path.splitext(relpath)[0]}" already exists. Skipping...')
        return

    decompress_op(path, dest_path, tag, silent=args.silent)

def check_resolution_safe(file_path: str, max_pixels: int = 2048) -> bool:
    """Returns True if the image is safe (<= max_pixels), False otherwise."""
    try:
        with Image.open(file_path) as img:
            w, h = img.size
            if w > max_pixels or h > max_pixels:
                print(f"SKIPPING UNSAFE IMAGE: {os.path.basename(file_path)} ({w}x{h}). Max allowed is {max_pixels}px.")
                return False
            return True
    except Exception as e:
        print(f"Warning: Could not check resolution for {file_path}: {e}")
        return True

def compress_batch(tag_dir: str, args) -> None:
    tag = os.path.basename(tag_dir)
    
    def check_dds_exist(relpath) -> bool:
        dest_path = os.path.join(args.output_dir, os.path.splitext(relpath)[0] + ".dds")
        if args.skip and os.path.exists(dest_path):
            print(f'"{os.path.splitext(relpath)[0]}" already exists. Skipping...')
            return True
        return False

    is_cubemap = tag.endswith("_CUBEMAP")
    if is_cubemap:
        if not is_path_var("CubeMapGen"):
            print("CubeMapGen not found. Make sure it's in your PATH.")
            return

        mapgen_str = 'CubeMapGen -exit -consoleErrorOutput -exportCubeDDS -exportPixelFormat:A8R8G8B8 -exportFilename:"{out_path}.dds"'

        for i, (axis, side) in enumerate(product(("X", "Y", "Z"), ("Pos", "Neg"))):
            mapgen_str += (
                f' -importFace{axis}{side}:"' + "{in_path}_face" + str(i) + '.png"'
            )

        fn_ = partial(
            transform_op,
            command=mapgen_str,
            source_dir=tag_dir,
            target_dir=args.output_dir,
            silent=args.silent,
        )

        files = (
            p[: p.rfind("_face")]
            for p in scan_dir(tag_dir, recurse=args.recurse)
            if os.path.splitext(p)[0].endswith("_face0")
        )
        files = filter(lambda x: not check_dds_exist(os.path.relpath(x, tag_dir)), files)
        multiproc(fn_, files, args.p, chunksize=3)

    else:
        if tag == "DXT5_xGxR":
            if not is_path_var("crunch_x64"):
                print("crunch_x64 not found. Make sure it's in your PATH.")
                return
            compress_str = 'crunch_x64 -quiet -renormalize -mipMode Generate -dxtQuality uber -DXT5_xGxR -fileformat dds  -file "{in_path}" -outdir "{out_path}"'
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
                    print(f'Input texture format "{tag}" not supported yet!')
                    return

            # FIX: Detect if we should use mipmaps
            # UI, Fonts, and 2D elements should NOT have mipmaps in Yakuza engine
            if any(x in tag_dir.lower() for x in ["2d", "ui", "font", "sy", "system"]):
                mip_arg = "-nomips"
                print(f"  [Info] Mipmaps disabled for {tag_dir} (UI/2D detected)")
            else:
                mip_arg = "-mipfilter kaiser"

            compress_str = (
                f"nvcompress -silent -{format} {mip_arg} -production "
                + '"{in_path}" "{out_path}"'
            )

        temp_indir = os.path.join(TEMP_DIR, "_in")
        temp_outdir = os.path.join(TEMP_DIR, "_out")
        hardlink_files(tag_dir, temp_indir)
        
        # FIX: Scan files for safety before processing
        for file_path in scan_dir(temp_indir, recurse=args.recurse):
            rel_path = os.path.relpath(file_path, temp_indir)

            if check_dds_exist(rel_path):
                os.remove(file_path)
                continue

            if not check_resolution_safe(file_path):
                os.remove(file_path) # Remove unsafe file so it is not processed
                continue

        fn_ = partial(
            transform_op,
            file_addr=temp_indir,
            command=compress_str,
            source_dir=temp_indir,
        )

        try:
            if args.recurse:
                file_paths = flatten_dir(temp_indir)
            fn_(target_dir=temp_outdir)
            if args.recurse:
                file_paths_dds = dict(
                    zip(
                        map(
                            lambda x: os.path.splitext(x)[0] + ".dds", file_paths.keys()
                        ),
                        file_paths.values(),
                    )
                )
                unravel_dir(temp_outdir, file_paths_dds)

            run_proc(
                f'robocopy /s /move /njh /njs /ndl "{temp_outdir.rstrip('\\')}" "{args.output_dir.rstrip('\\')}"',
                silent=args.silent,
            )
        finally:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

def main():
    args = parser.parse_args()
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
    is_decompress = args.operation == "decompress"

    try:
        prevent_sleep()
        if is_decompress:
            if not (is_path_var("nvddsinfo") and is_path_var("nvdecompress")):
                print("Make sure you have NVIDIA Texture Tools installed and in your PATH.")
                exit()
            fn_ = partial(decompress, args=args)
            multiproc(fn_, scan_dir(args.source_dir, recurse=args.recurse), args.p)
        else:
            if not is_path_var("nvcompress"):
                print("Make sure you have NVIDIA Texture Tools installed and in your PATH.")
                exit()
            for tag_dir in scan_dir(args.source_dir, recurse=False, return_dirs=True):
                print(f"Processing {tag_dir}...")
                compress_batch(tag_dir, args)
    finally:
        prevent_sleep(False)

if __name__ == "__main__":
    main()
