import argparse
import os
import shutil
import hashlib
import json
import gzip
from typing import Optional

from utils import (
    get_dir_size,
    get_par_dirs,
    hardlink_files,
    is_descendant_of,
    link_compat,
    prevent_sleep,
    run_proc,
    scan_dir,
)

PAR_TOOL = "ParTool.exe"
PAR_TOOL_ARGS = "add -c 1"
STAGED_DIR = os.path.abspath("__staged")
TEMP_DIR = os.path.abspath(".tmp")
BACKUP_DIR = os.path.abspath("__backup")

parser = argparse.ArgumentParser(description="Update game files with ParTool.")
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Update assuming backup files as the source. Removes previous staged changes applied to the game.",
)


class FileHashLedger:
    def __init__(self, ledger_file: str) -> None:
        self._path: str = ledger_file
        self._table: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with gzip.open(self._path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def _save(self) -> None:
        with gzip.open(self._path, "wt", encoding="utf-8") as f:
            json.dump(self._table, f)

    def _gen_hash(self, file_path: str) -> str:
        hash_func = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def get_diff(self, directory: str) -> dict:
        assert is_descendant_of(directory, TEMP_DIR)
        # If a file is removed, it's not considered a change by design.
        diff = {
            k: v
            for file in scan_dir(directory, recurse=True)
            if self._table.get(k := os.path.relpath(file, TEMP_DIR), None)
            != (v := self._gen_hash(file))
        }
        return diff

    def push(self, diff: dict) -> None:
        self._table.update(diff)
        self._save()

    def clear(self, directory: Optional[str] = None) -> None:
        if directory:
            assert is_descendant_of(directory, TEMP_DIR)
            for file in scan_dir(directory, recurse=True):
                self._table.pop(os.path.relpath(file, TEMP_DIR), None)
        else:
            self._table.clear()
        self._save()


def update_par(
    source_dir: str,
    target_path: str,
    ledger: FileHashLedger,
    backup_as_source: bool = False,
) -> None:
    file_relpath = os.path.relpath(target_path, os.getcwd())
    backup_path = os.path.join(BACKUP_DIR, file_relpath)
    dest_path = os.path.join(TEMP_DIR, file_relpath)

    if not os.path.isfile(backup_path):
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(file_relpath, backup_path)

    if backup_as_source:
        source_path = backup_path
    else:
        source_path = file_relpath
        if not os.path.exists(source_path):  # Fallback to backup if source is missing.
            print(f'File "{source_path}" could not be found. Using backup as source.')
            link_compat(backup_path, source_path)
            ledger.clear(source_dir)

    diff = ledger.get_diff(source_dir)
    if len(diff) == 0:
        print("No changes detected. Skipping update...")
        link_compat(source_path, dest_path)
        shutil.rmtree(source_dir)
        return

    limit_size = 2 * 1024 * 1024 * 1024  # 2GB
    if get_dir_size(source_dir) > limit_size:
        print("Update exceeds 2GB. Splitting into parts...")
        partitions = [set()]
        last_partition_size = 0
        parts_dir = os.path.join(source_dir, ".parts")
        for dirpath, _, filenames in os.walk(source_dir):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                file_size = os.path.getsize(file_path)
                if last_partition_size + file_size > limit_size:
                    partitions.append(set())
                    last_partition_size = 0
                partitions[-1].add(file_path)
                last_partition_size += file_size
                new_path = os.path.join(
                    parts_dir,
                    f"{len(partitions) - 1}",
                    os.path.relpath(file_path, source_dir),
                )
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                os.rename(file_path, new_path)

        for sub_dir in os.listdir(parts_dir):
            update_par(
                source_dir=os.path.join(parts_dir, sub_dir),
                target_path=target_path,
                ledger=ledger,
                backup_as_source=backup_as_source,
            )
            backup_as_source = False
        return

    command_string = (
        f'{PAR_TOOL} {PAR_TOOL_ARGS} "{source_path}" "{source_dir}" "{dest_path}"'
    )
    run_proc(command_string)
    shutil.rmtree(source_dir)

    if os.path.exists(file_relpath):
        os.remove(file_relpath)
    link_compat(dest_path, file_relpath)
    ledger.push(diff)


def main():
    args = parser.parse_args()

    if os.path.exists(TEMP_DIR):
        c = input(
            f"The contents of {TEMP_DIR} must be cleared to avoid conflict. Would you like to proceed? (y/n) "
        )
        if c.lower() == "y":
            shutil.rmtree(TEMP_DIR)
        else:
            exit()

    path_ = os.path.join(BACKUP_DIR, "ledger.json.gz")
    ledger = FileHashLedger(path_)
    if args.fresh:
        ledger.clear()

    try:
        prevent_sleep()
        hardlink_files(STAGED_DIR, TEMP_DIR)

        for dir_path in get_par_dirs(TEMP_DIR):
            dir_relpath = os.path.relpath(dir_path, TEMP_DIR)
            target_path = os.path.join(
                (s_ := os.path.split(dir_relpath))[0], s_[1][1:]
            )  # remove the leading underscore of the directory name.

            print(f'Processing "{dir_relpath}"...')
            update_par(
                source_dir=dir_path,
                target_path=target_path,
                ledger=ledger,
                backup_as_source=args.fresh,
            )

    finally:
        shutil.rmtree(TEMP_DIR)
        prevent_sleep(False)
        print("")

    print("All done!")


if __name__ == "__main__":
    main()
