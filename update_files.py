from __future__ import annotations
from collections import namedtuple
from typing import Self
from utils.file_utils import (
    get_dir_size,
    get_par_dirs,
    hardlink_files,
    is_descendant_of,
    link_compat,
    md5_hash,
    partition,
    scan_dir,
)

import argparse
import os
import shutil
import json
import gzip

from utils.proc_utils import prevent_sleep, run_proc


PAR_TOOL = "ParTool.exe"
PAR_TOOL_ARGS = "add -c 1"
STAGED_DIR = os.path.abspath("__staged")
TEMP_DIR = os.path.abspath(".tmp")
BACKUP_DIR = os.path.abspath("__backup")

parser = argparse.ArgumentParser(description="Update game files with ParTool.")
parser.add_argument(
    "--fresh",
    action="store_true",
    help="Update assuming backup files as the source.",
)
parser.add_argument(
    "--skip",
    action="store_true",
    help="Skip updating unchanged files. (according to the ledger)",
)
parser.add_argument(
    "--ignore_mismatch",
    action="store_true",
    help="Avoid rebuild if the target has been modified since the last update. (according to the ledger)",
)


class Ledger(dict):

    class Snapshot(dict):
        def changed(self, content_dir: str) -> Self:
            diff_entry = Ledger.Snapshot(
                {
                    k: v
                    for file in scan_dir(content_dir, recurse=True)
                    if self.get(k := os.path.relpath(file, content_dir), None)
                    != (v := md5_hash(file))
                }
            )
            return diff_entry

    Entry = namedtuple("Entry", ["hash", "snapshot"])

    def __init__(self, file_path: str) -> None:
        super().__init__()
        self._path = file_path
        if os.path.exists(self._path):
            self.load()

    def __delitem__(self, key: str) -> None:
        if key in self:
            super().__delitem__(key)

    def __getitem__(self, key: str) -> Ledger.Snapshot:
        assert is_descendant_of(key, os.getcwd())
        if key not in self:
            self[key] = Ledger.Entry(md5_hash(key), Ledger.Snapshot())
        return super().__getitem__(key)

    def load(self) -> dict:
        with gzip.open(self._path, "rt", encoding="utf-8") as f:
            self.update(
                {
                    k: Ledger.Entry(v[0], Ledger.Snapshot(v[1]))
                    for k, v in json.load(f).items()
                }
            )

    def save(self) -> None:
        with gzip.open(self._path, "wt", encoding="utf-8") as f:
            json.dump(self, f)


def update_par(
    target_par: str,
    content_dir: str,
    ledger: Ledger,
    fresh: bool = False,
    skip: bool = True,
    strict: bool = True,
) -> Ledger.Snapshot:
    assert is_descendant_of(target_par, os.getcwd())

    backup_par = os.path.join(BACKUP_DIR, target_par)
    create_backup_if_missing(target_par, backup_par)

    target_par = os.path.relpath(target_par, os.getcwd())
    source_par = backup_par if fresh else target_par
    par_dirty = strict and md5_hash(target_par) != ledger[target_par].hash

    diff = ledger[target_par].snapshot.changed(content_dir)
    if par_dirty:
        if not fresh:
            print("File has been modified since last update. Rebuilding...")
    else:
        no_change = len(diff) == 0
        if skip and no_change:
            print("No changes detected. Skipping update...")
            temp_par = os.path.join(TEMP_DIR, os.path.relpath(target_par, os.getcwd()))
            link_compat(source_par, temp_par)
            shutil.rmtree(content_dir)
            return

    if fresh or par_dirty:
        del ledger[target_par]
        diff = ledger[target_par].snapshot.changed(content_dir)
    else:
        if not os.path.exists(target_par):  # Fallback to backup if source is missing.
            print(f'File "{target_par}" could not be found. Using backup as source.')
            link_compat(backup_par, target_par)

    remove_unchanged_files(content_dir, diff)

    max_size = 2**31  # 2GB
    parts = None
    if (dir_size := get_dir_size(content_dir)) > max_size:
        print(
            f"Update ({dir_size / 2**30 : 0.2f}GB) exceeds {max_size / 2**30 : 0.2f}GB. Splitting into parts..."
        )
        parts_dir, parts = partition(content_dir, max_size)

    def pack_save(content_dir: str):
        par_add(source_par, temp_par, content_dir)
        shutil.rmtree(content_dir)
        if os.path.exists(target_par):
            os.remove(target_par)
        link_compat(temp_par, target_par)
        ledger[target_par] = Ledger.Entry(
            md5_hash(target_par),
            Ledger.Snapshot({**prev_snap, **diff}),
        )
        ledger.save()

    temp_par = os.path.join(TEMP_DIR, os.path.relpath(target_par, os.getcwd()))
    if parts is None:
        prev_snap = ledger[target_par].snapshot
        pack_save(content_dir)
    else:
        for part in os.listdir(parts_dir):
            print()
            print(f"Processing part {part}...")
            part_dir = os.path.join(parts_dir, part)
            prev_snap = ledger[target_par].snapshot
            diff = prev_snap.changed(part_dir)
            if os.path.exists(temp_par):
                os.remove(temp_par)
            pack_save(part_dir)
            os.remove(temp_par)
            source_par = target_par


def remove_unchanged_files(content_dir: str, diff: dict):
    for file in scan_dir(content_dir, recurse=True):
        if os.path.relpath(file, content_dir) not in diff:
            os.remove(file)


def create_backup_if_missing(target_par: str, backup_par: str) -> str:
    if not os.path.isfile(backup_par):
        os.makedirs(os.path.dirname(backup_par), exist_ok=True)
        shutil.copy2(target_par, backup_par)


def par_add(source_par: str, dest_par: str, content_dir: str) -> None:
    command_string = (
        f'{PAR_TOOL} {PAR_TOOL_ARGS} "{source_par}" "{content_dir}" "{dest_par}"'
    )
    run_proc(command_string)


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
                skip=args.skip,
                strict=not args.ignore_mismatch,
            )
            print()

    finally:
        shutil.rmtree(TEMP_DIR)
        prevent_sleep(False)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
