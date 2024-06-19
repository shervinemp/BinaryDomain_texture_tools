import hashlib
import os
import shutil


def is_path_var(name):
    return shutil.which(name) is not None


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


def get_dir_size(path: str):
    return sum(
        [
            os.path.getsize(os.path.join(root, file))
            for root, _, files in os.walk(path)
            for file in files
        ]
    )


def is_empty_dir(path: str):
    for _, _, files in os.walk(path):
        if files:
            return False
    return True


def is_par_dir(path):
    return (s_ := os.path.basename(path)).startswith("_") and s_.endswith(".par")


def is_descendant_of(path, root):
    return not os.path.relpath(path, root).startswith("..")


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


def partition(dir_path, max_size=2**31):  # 2GB
    parts = [set()]
    last_part_size = 0
    parts_dir = os.path.join(dir_path, ".parts")
    for dirpath, _, filenames in os.walk(dir_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_size = os.path.getsize(file_path)
            if last_part_size + file_size > max_size:
                parts.append(set())
                last_part_size = 0
            parts[-1].add(file_path)
            last_part_size += file_size
            new_path = os.path.join(
                parts_dir,
                f"{len(parts) - 1}",
                os.path.relpath(file_path, dir_path),
            )
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(file_path, new_path)
    return parts_dir, parts


def md5_hash(path: str):
    hash_func = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()
