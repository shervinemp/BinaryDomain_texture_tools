from functools import partial
import shutil
import subprocess
from typing import Callable, Iterable, Set
from PIL import Image
import os
import multiprocessing as mp

echo = partial(print, end="")


def is_path_var(name):
    return shutil.which(name) is not None


def run_proc(command_string, silent=False):
    with subprocess.Popen(
        command_string,
        stdout=subprocess.DEVNULL if silent else subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    ) as prc:
        try:
            while prc.poll() is None:
                out = prc.stdout.readline()
                echo(out.decode())
            err = prc.stderr.read()
            echo(err.decode())
        except KeyboardInterrupt as e:
            if not silent:
                print(f"Incomplete: {command_string}")
            raise e


def multiproc(fn_: Callable, iter_: Iterable, p: int, chunksize: int = 8) -> Set:
    if p == 1:
        out = set(map(fn_, iter_))
    else:
        p_count = p if p > 0 else mp.cpu_count() - p
        with mp.Pool(p_count) as p:
            out = set(
                p.imap_unordered(
                    fn_,
                    iter_,
                    chunksize=chunksize,
                )
            )
    out.discard(None)
    return out


def transform_op(
    file_addr: str,
    command: str,
    source_dir: str,
    target_dir: str,
    *,
    silent: bool = False,
):
    is_file = os.path.isfile(file_addr)
    rel_p_ = os.path.relpath(file_addr, source_dir)
    tp_ = os.path.join(target_dir, rel_p_)
    dirname = os.path.dirname(tp_) if is_file else tp_
    os.makedirs(dirname, exist_ok=True)
    command_str = command.format(in_path=file_addr, out_path=tp_)
    run_proc(command_str, silent=silent)


def load_image(path: str):
    im = Image.open(path)
    if os.path.exists((p := os.path.splitext(path))[0] + "_alpha." + p[1]):
        alpha = Image.open(p[0] + "_alpha." + p[1]).convert("L")
        im.putalpha(alpha)
    return im


def scan_dir(path: str, recurse: bool = False, return_dirs: bool = False):
    for entry in os.scandir(path):
        if entry.is_file():
            if not return_dirs:
                yield entry.path
        else:
            if return_dirs:
                yield entry.path
            if recurse:
                yield from scan_dir(
                    entry.path, recurse=recurse, return_dirs=return_dirs
                )


def is_par_dir(path):
    return (s_ := os.path.basename(path)).startswith("_") and s_.endswith(".par")


def get_par_dirs(dir_path):
    has_files = set()
    for root, _, files in os.walk(dir_path, topdown=False):
        # check if the directory has files or if it's a subdirectory of a directory with files
        if files or root in has_files:
            dir_path = os.path.dirname(root)
            has_files.add(dir_path)
            has_files.discard(root)
            if is_par_dir(root):
                yield root


def get_path_w_ext(base_path, extensions):
    if isinstance(extensions, str):
        extensions = (extensions,)
    for extension in extensions:
        path_with_extension = f"{base_path}.{extension}"
        if os.path.exists(path_with_extension):
            return path_with_extension
    raise FileNotFoundError(
        f'No existing file found for "{base_path}" with any of the given extensions: {extensions}'
    )


def hardlink_files(src_dir, dest_dir):
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, os.path.relpath(src_path, src_dir))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            link_compat(src_path, dest_path)


def link_compat(src, dest):
    try:
        # try hardlinking first
        os.link(src, dest)
    except OSError:
        # if that fails, copy the file
        shutil.copy2(src, dest)


def flatten_dir(directory):
    # move all files in subdirectories to the root directory
    index = 0
    file_paths = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            new_filename = f"{index}_{filename}"
            index += 1
            src_path = os.path.join(root, filename)
            file_paths[new_filename] = os.path.relpath(root, directory)
            dest_path = os.path.join(directory, new_filename)
            os.rename(src_path, dest_path)
    for root, dirs, _ in os.walk(directory, topdown=False):
        for d in dirs:
            dir_path = os.path.join(root, d)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass
    return file_paths


def unravel_dir(directory, file_paths):
    # move files back to their original subdirectories
    for filename, orig_dir in file_paths.items():
        orig_filename = filename.split("_", 1)[1]
        dest_dir = os.path.join(directory, orig_dir)
        dest_path = os.path.join(dest_dir, orig_filename)
        os.makedirs(dest_dir, exist_ok=True)
        os.rename(os.path.join(directory, filename), dest_path)


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


def get_ddsinfo(path: str):
    proc = subprocess.Popen(
        f'nvddsinfo "{path}"', stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    info = parse_ddsinfo(proc.stdout.read().decode().split("\n"))
    return info


def parse_ddsinfo(lines, *, return_count=False):
    lines = list(filter(len, lines))
    info = {}
    lines_parsed = 0
    while len(lines):
        h = 1
        if ":" in lines[0]:
            item = list(filter(len, (x.strip() for x in lines[0].split(":"))))
            if len(lines) > 1 and any(lines[1].startswith(ws) for ws in ("\t", " ")):
                ws = lines[1][0]
                w = 1
                for c in lines[1][1:]:
                    if c == ws:
                        w += 1
                    else:
                        break
                indent = ws * w
                for l in lines[1:]:
                    if l.startswith(indent):
                        h += 1
                    else:
                        break
                parsed, _ = parse_ddsinfo(
                    [x[w:] for x in lines[1:h]], return_count=True
                )
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


def prevent_sleep(reset=False):
    import ctypes

    # Constants for the Windows API
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    if reset:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    else:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
