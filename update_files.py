import argparse
import os
import shutil
import subprocess
import hashlib
import json
import gzip

from utils import (
    get_par_dirs,
    hardlink_files,
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


class UpdateLedger:
    def __init__(self, ledger_file: str) -> None:
        self._path: str = ledger_file
        self.ledger: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with gzip.open(self._path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            return {}

    def _save(self) -> None:
        with gzip.open(self._path, "wt", encoding="utf-8") as f:
            json.dump(self.ledger, f)

    def _generate_hash(self, file_path: str) -> str:
        hash_func = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def diff(self, directory: str) -> dict:
        # Don't allow the user to update files from outside the staged directory.
        assert not os.path.relpath(directory, TEMP_DIR).startswith("..")

        # Removed files are not considered to allow clean up of the staged directory.
        diff = {
            k: v
            for file in scan_dir(directory, recurse=True)
            if self.ledger.get(k := os.path.relpath(file, TEMP_DIR), None)
            != (v := self._generate_hash(file))
        }
        return diff

    def push_diff(self, diff: dict) -> None:
        self.ledger.update(diff)
        self._save()

    def clear(self) -> None:
        self.ledger.clear()
        self._save()


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
    ledger = UpdateLedger(path_)
    if args.fresh:
        ledger.clear()

    try:
        prevent_sleep()
        hardlink_files(STAGED_DIR, TEMP_DIR)

        for dir_path in get_par_dirs(TEMP_DIR):
            dir_relpath = os.path.relpath(dir_path, TEMP_DIR)

            # remove the leading underscore of the directory name.
            file_relpath = os.path.join(
                (s_ := os.path.split(dir_relpath))[0], s_[1][1:]
            )
            if not os.path.isfile(file_relpath):
                raise ValueError(f'No file found at "{file_relpath}"')

            backup_path = os.path.join(BACKUP_DIR, file_relpath)
            dest_path = os.path.join(TEMP_DIR, file_relpath)

            print(f'Processing "{dir_relpath}"...')
            # execute the external program and capture the output
            source_path = backup_path if args.fresh else file_relpath
            command_string = (
                f'{PAR_TOOL} {PAR_TOOL_ARGS} "{source_path}" "{dir_path}" "{dest_path}"'
            )

            if not os.path.isfile(backup_path):
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.copy2(file_relpath, backup_path)

            diff = ledger.diff(dir_path)
            if len(diff) == 0:
                print("No changes detected. Skipping update...")
                continue

            run_proc(command_string)

            ledger.push_diff(diff)
            shutil.rmtree(dir_path)

            os.remove(file_relpath)
            link_compat(dest_path, file_relpath)
            print("")

    finally:
        shutil.rmtree(TEMP_DIR)
        prevent_sleep(False)

    print("All done!")


if __name__ == "__main__":
    main()


# Incomplete: ParTool.exe add -c 1 "C:\Program Files (x86)\Steam\steamapps\common\Binary Domain\__backup\p_auth\s0830_0000a_01.par" "C:\Program Files (x86)\Steam\steamapps\common\Binary Domain\.tmp\p_auth\_s0830_0000a_01.par" "C:\Program Files (x86)\Steam\steamapps\common\Binary Domain\.tmp\p_auth\s0830_0000a_01.par"
# Changed files: {'..\\.tmp\\p_auth\\_s0830_0000a_01.par\\TiEventAuth_Archive\\dummy_ref.dds': '10432276dfb9de9c5b043004d4bcb393', '..\\.tmp\\p_auth\\_s0830_0000a_01.par\\TiEventAuth_Archive\\o_a_kas_caine_wire_rope_d.dds': 'c331feb6c3c0ea6bea3860a51ed5abc3', '..\\.tmp\\p_auth\\_s0830_0000a_01.par\\TiEventAuth_Archive\\o_a_kas_caine_wire_rope_n.dds': '335ef07a7305e7fbd36cc7afadc3c020', '..\\.tmp\\p_auth\\_s0830_0000a_01.par\\TiEventAuth_Archive\\o_a_kas_caine_wire_rope_u.dds': 'c7b831b7ae6166ff728ec59ffdb9bd76'}
