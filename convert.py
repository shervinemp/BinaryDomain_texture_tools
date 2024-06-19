import os
import argparse
from functools import partial
from itertools import product
import shutil

from utils.file_utils import flatten_dir, hardlink_files, is_path_var, scan_dir, unravel_dir
from utils.image_utils import get_ddsinfo, get_format_tag
from utils.proc_utils import multiproc, prevent_sleep, run_proc, transform_op

TEMP_DIR = os.path.abspath(".tmp")


# Define command-line arguments
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


def decompress_op(orig_path: str, target_path: str, tag: str, *, silent: bool = False):
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


def decompress(path: str, args):
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


def compress_batch(tag_dir: str, args):
    tag = os.path.basename(tag_dir)
    
    def check_dds_exist(relpath):
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

            compress_str = (
                f"nvcompress -silent -{format} -mipfilter kaiser -production "
                + '"{in_path}" "{out_path}"'
            )

        temp_indir = os.path.join(TEMP_DIR, "_in")
        temp_outdir = os.path.join(TEMP_DIR, "_out")
        hardlink_files(tag_dir, temp_indir)
        
        for file_path in scan_dir(temp_indir, recurse=args.recurse):
            if check_dds_exist(os.path.relpath(file_path, temp_indir)):
                os.remove(file_path)

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
                print(
                    "Make sure you have NVIDIA Texture Tools installed and in your PATH."
                )
                exit()
            fn_ = partial(decompress, args=args)
            multiproc(fn_, scan_dir(args.source_dir, recurse=args.recurse), args.p)
        else:
            if not is_path_var("nvcompress"):
                print(
                    "Make sure you have NVIDIA Texture Tools installed and in your PATH."
                )
                exit()
            for tag_dir in scan_dir(args.source_dir, recurse=False, return_dirs=True):
                print(f"Processing {tag_dir}...")
                compress_batch(tag_dir, args)
    finally:
        prevent_sleep(False)


if __name__ == "__main__":
    main()
