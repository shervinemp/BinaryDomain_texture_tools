import os
import shutil
import subprocess

from utils import get_par_dirs, run_proc

PAR_TOOL = "ParTool.exe"
PAR_TOOL_ARGS = "add -c 1"
MODIFIED_DIR = os.path.abspath("__modified")
TEMP_DIR = os.path.abspath(".tmp")
BACKUP_DIR = os.path.abspath("__backup")


def backup(path):
    rel_path = os.path.relpath(path, ".")
    backup_path = os.path.join(BACKUP_DIR, rel_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if not os.path.isfile(backup_path):
        shutil.copy2(rel_path, backup_path)


if __name__ == "__main__":

    if os.path.exists(TEMP_DIR):
        c = input(
            f"The contents of {TEMP_DIR} must be cleared to avoid conflict. Would you like to proceed? (y/n) "
        )
        if c.lower() == "y":
            shutil.rmtree(TEMP_DIR)
        else:
            exit()

    print(f'Copying files to "{TEMP_DIR}" for processing...')
    shutil.copytree(MODIFIED_DIR, TEMP_DIR)

    for dir_path in get_par_dirs(TEMP_DIR):
        dir_relpath = os.path.relpath(dir_path, TEMP_DIR)

        # remove the leading underscore of the directory name and also the tag directories.
        s_ = os.path.split(dir_relpath)
        r_ = s_[0].split(os.sep)
        file_relpath = os.path.join(
            r_[0],
            *r_[2:],
            s_[1][1:],
        )

        dest_path = os.path.join(TEMP_DIR, file_relpath)
        backup_path = os.path.join(BACKUP_DIR, file_relpath)

        print(f'Processing "{dir_relpath}"...')
        try:
            if not os.path.isfile(file_relpath):
                raise ValueError(f'No file found at "{file_relpath}"')

            # execute the external program and capture the output
            command_string = (
                f'{PAR_TOOL} {PAR_TOOL_ARGS} "{backup_path}" "{dir_path}" "{dest_path}"'
            )

            backup(file_relpath)

            # remove destpath if it exists
            if os.path.exists(dest_path):
                os.remove(dest_path)

            run_proc(command_string)
        except ValueError as e:
            print(str(e))
        finally:
            shutil.rmtree(dir_path)
        print("")

    print("Overwriting game files...")
    subprocess.run(
        f'robocopy /s /move /njh /njs /ndl "{TEMP_DIR}" .', stdout=subprocess.DEVNULL
    )
    print("Done!")
