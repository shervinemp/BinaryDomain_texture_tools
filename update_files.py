from __future__ import annotations
from collections import namedtuple
from copy import copy
from typing import Self
from utils.file_utils import (
    get_dir_size,
    get_par_dirs,
    hardlink_files,
    is_descendant_of,
    is_empty_iter,
    link_compat,
    md5_hash,
    partition,
    scan_dir,
    sizeof_fmt,
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
    help="Don't force a rebuild for targets modified since the last update. (according to the ledger)",
)


class Ledger(dict):
    Entry = namedtuple("Entry", ["hash", "snapshot"])

    class Snapshot(dict):
        def diff(self, content_dir: str) -> Self:
            return Ledger.Snapshot(self.changed(content_dir))

        def changed(self, content_dir: str) -> Self:
            return (
                (k, v)
                for file in scan_dir(content_dir, recurse=True)
                if self.get(k := os.path.relpath(file, content_dir), None)
                != (v := md5_hash(file))
            )

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


class Updater:
    ledger_path: str = os.path.join(BACKUP_DIR, "ledger.json.gz")
    payload_max_size = 2**31  # 2GB

    def __init__(
        self,
        force_rebuild: bool = False,
        skip_existing: bool = True,
        check_mismatch: bool = True,
    ) -> None:
        self._rebuild = force_rebuild
        self._skip = skip_existing
        self._check = check_mismatch
        self.ledger = Ledger(Updater.ledger_path)

    def apply(self, update: Update) -> Update:
        tp_ = update.target_path
        ledger_entry = self.ledger[tp_]

        backup_path = os.path.join(BACKUP_DIR, tp_)
        create_backup_if_missing(tp_, backup_path)

        mismatch_ = self._check and md5_hash(tp_) != ledger_entry.hash
        if not self._rebuild and mismatch_:
            print("File has been modified since last update. Forcing rebuild...")

        if self._rebuild or mismatch_:
            source_path = backup_path
            del self.ledger[tp_]
            ledger_entry = self.ledger[tp_]
        else:
            source_path = tp_

        dir_unchanged_ = is_empty_iter(
            ledger_entry.snapshot.changed(update.content_dir)
        )
        if self._skip and dir_unchanged_:
            print("No changes detected. Skipping update...")
            link_compat(source_path, update._temp_path)
            shutil.rmtree(update.content_dir)
            return

        parts = None
        if (dir_size := get_dir_size(update.content_dir)) > Updater.payload_max_size:
            parts_dir, parts = partition(update.content_dir, Updater.payload_max_size)
            print(
                f"Update size ({sizeof_fmt(dir_size)} exceeds {sizeof_fmt(Updater.payload_max_size)}."
            )
            print(f"Splitting into {len(parts)} parts...")
        else:
            print(f"Update size: {sizeof_fmt(dir_size)}")

        if parts is None:
            self.ledger[tp_] = update._pack(source_path, ledger_entry.snapshot)
            self.ledger.save()
        else:
            for part in os.listdir(parts_dir):
                print()
                print(f"Processing part {part}...")
                part_dir = os.path.join(parts_dir, part)
                upd_part = copy(update)
                upd_part.content_dir = part_dir
                t_ = upd_part._temp_path
                if os.path.exists(t_):
                    os.remove(t_)
                self.ledger[tp_] = upd_part._pack(source_path, ledger_entry.snapshot)
                self.ledger.save()
                os.remove(t_)
                source_path = tp_


class Update:
    def __init__(
        self,
        target_path: str,
        content_dir: str,
    ) -> None:
        assert is_descendant_of(target_path, os.getcwd())
        self.target_path = os.path.relpath(target_path, os.getcwd())
        self.content_dir = content_dir
        self._temp_path = os.path.join(
            TEMP_DIR, os.path.relpath(self.target_path, os.getcwd())
        )

    def _pack(self, source_par: str, curr_snapshot: Ledger.Snapshot) -> Ledger.Entry:
        diff = curr_snapshot.diff(self.content_dir)
        remove_unchanged_files(self.content_dir, diff)

        par_add(source_par, self._temp_path, self.content_dir)

        shutil.rmtree(self.content_dir)
        if os.path.exists(self.target_path):
            os.remove(self.target_path)
        link_compat(self._temp_path, self.target_path)

        return Ledger.Entry(
            md5_hash(self.target_path),
            Ledger.Snapshot(curr_snapshot | diff),
        )


def remove_unchanged_files(content_dir: str, diff: dict):
    for file in scan_dir(content_dir, recurse=True):
        if os.path.relpath(file, content_dir) not in diff:
            os.remove(file)


def create_backup_if_missing(target_path: str, backup_path: str) -> str:
    if not os.path.isfile(backup_path):
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(target_path, backup_path)


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

    updater = Updater(
        force_rebuild=args.fresh,
        skip_existing=args.skip,
        check_mismatch=not args.ignore_mismatch,
    )
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
            update = Update(target_path, dir_path)
            updater.apply(update)

    finally:
        shutil.rmtree(TEMP_DIR)
        prevent_sleep(False)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
