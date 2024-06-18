from __future__ import annotations
from typing import Self
from utils import (
    get_dir_size,
    get_par_dirs,
    hardlink_files,
    is_descendant_of,
    link_compat,
    md5,
    prevent_sleep,
    run_proc,
    scan_dir,
)

import argparse
import os
import shutil
import json
import gzip


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


class Ledger(dict):

    class HashSnapshot(dict):

        def changed(self, content_dir: str) -> Self:
            diff_entry = Ledger.HashSnapshot(
                {
                    k: v
                    for file in scan_dir(content_dir, recurse=True)
                    if self.get(k := os.path.relpath(file, content_dir), None)
                    != (v := md5(file))
                }
            )
            return diff_entry

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._path = file_path
        if os.path.exists(self._path):
            self.load()

    def __delitem__(self, key: str) -> None:
        if key in self:
            super().__delitem__(key)

    def __getitem__(self, key: str) -> Ledger.HashSnapshot:
        assert is_descendant_of(key, os.getcwd())
        if key not in self:
            self[key] = Ledger.HashSnapshot()
        return super().__getitem__(key)

    def load(self) -> dict:
        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            self.update({k: Ledger.HashSnapshot(v) for k, v in json.load(f).items()})

    def save(self) -> None:
        with gzip.open(self._path, "wt", encoding="utf-8") as f:
            json.dump({k: dict(v) for k, v in self.items()}, f)


def update_par(
    target_par: str,
    content_dir: str,
    ledger: Ledger,
    fresh: bool = False,
) -> Ledger.HashSnapshot:
    assert is_descendant_of(target_par, os.getcwd())

    backup_par = os.path.join(BACKUP_DIR, target_par)
    target_par = os.path.relpath(target_par, os.getcwd())

    if not os.path.isfile(backup_par):  # Create backup if missing.
        os.makedirs(os.path.dirname(backup_par), exist_ok=True)
        shutil.copy2(target_par, backup_par)

    if fresh:
        source_par = backup_par
    else:
        if not os.path.exists(target_par):  # Fallback to backup if source is missing.
            print(f'File "{target_par}" could not be found. Using backup as source.')
            del ledger[target_par]
            link_compat(backup_par, target_par)
        source_par = target_par

    diff = ledger[target_par].changed(content_dir)
    if len(diff) == 0:
        print("No changes detected. Skipping update...")
        temp_par = os.path.join(TEMP_DIR, os.path.relpath(target_par, os.getcwd()))
        link_compat(source_par, temp_par)
        shutil.rmtree(content_dir)
        return

    # remove unchanged files from the content directory
    for file in scan_dir(content_dir, recurse=True):
        if os.path.relpath(file, content_dir) not in diff:
            os.remove(file)

    max_size = 2**31  # 2GB
    parts = None
    if (dir_size := get_dir_size(content_dir)) > max_size:
        print(
            f"Update ({dir_size / 2**30 : 0.2f}GB) exceeds {max_size / 2**30 : 0.2f}GB. Splitting into parts..."
        )
        parts_dir, parts = partition(content_dir, max_size)

    temp_par = os.path.join(TEMP_DIR, os.path.relpath(target_par, os.getcwd()))

    def _op(content_dir: str):
        pack_add(source_par, temp_par, content_dir)
        if os.path.exists(target_par):
            os.remove(target_par)
        link_compat(temp_par, target_par)

    if parts is None:
        _op(content_dir)
        ledger[target_par].update(diff)
        ledger.save()
    else:
        for part in os.listdir(parts_dir):
            print(f"Processing part {part}...")
            part_dir = os.path.join(parts_dir, part)
            diff = ledger[target_par].changed(part_dir)
            _op(part_dir)
            ledger[target_par].update(diff)
            ledger.save()
            source_par = target_par


def partition(content_dir, max_size=2**31):  # 2GB
    parts = [set()]
    last_part_size = 0
    parts_dir = os.path.join(content_dir, ".parts")
    for dirpath, _, filenames in os.walk(content_dir):
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
                os.path.relpath(file_path, content_dir),
            )
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.rename(file_path, new_path)
    return parts_dir, parts


def pack_add(source_par: str, dest_par: str, content_dir: str) -> None:
    command_string = (
        f'{PAR_TOOL} {PAR_TOOL_ARGS} "{source_par}" "{content_dir}" "{dest_par}"'
    )
    run_proc(command_string)
    shutil.rmtree(content_dir)


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
    ledger = Ledger(path_)
    if args.fresh:
        ledger.clear()

    try:
        prevent_sleep()
        hardlink_files(STAGED_DIR, TEMP_DIR)

        for dir_path in get_par_dirs(TEMP_DIR):
            dir_relpath = os.path.relpath(dir_path, TEMP_DIR)
            target_path = os.path.join(
                (s_ := os.path.split(dir_relpath))[0],
                s_[1][1:],  # remove the leading underscore of the directory name.
            )

            print(f'Processing "{dir_relpath}"...')
            update_par(
                target_par=target_path,
                content_dir=dir_path,
                ledger=ledger,
                fresh=args.fresh,
            )

    finally:
        shutil.rmtree(TEMP_DIR)
        prevent_sleep(False)
        print("")

    print("All done!")


if __name__ == "__main__":
    main()
