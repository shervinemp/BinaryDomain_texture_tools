import os
import shutil
import subprocess
from functools import partial

PAR_TOOL = os.path.join(os.path.dirname(__file__), "ParTool.exe")
PAR_TOOL_ARGS = "add -c 1"
MODIFIED_DIR = os.path.abspath("modified_")
TEMP_DIR = os.path.abspath(".modified_")
BACKUP_DIR = os.path.abspath("backup_")

echo = partial(print, end="")

def run_proc(command_string):
    with subprocess.Popen(command_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as prc:
        while prc.poll() is None:
            out = prc.stdout.readline()
            echo(out.decode())
        err = prc.stderr.read()
        echo(err.decode())

def update_file(dir_path, par_path, backup_path=None):
    original_path = os.path.relpath(par_path, TEMP_DIR)

    if not os.path.isfile(original_path):
        raise ValueError(f"No file found at \"{original_path}\"")

    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if not os.path.isfile(backup_path):
        shutil.copy2(original_path, backup_path)

    # execute the external program and capture the output
    command_string = f"{PAR_TOOL} {PAR_TOOL_ARGS} \"{backup_path}\" \"{dir_path}\" \"{par_path}\""
    run_proc(command_string)

def process_dir(dir_path):
    file_paths_heap = []
    has_files = set()
    for root, dirs, files in os.walk(dir_path, topdown=False):
        if files or root in has_files:
            has_files.add(os.path.dirname(root))
            has_files.discard(root)
            if root.endswith(".par_"):
                yield root


if __name__ == "__main__":
    print(f"Copying files to \"{TEMP_DIR}\"...")
    shutil.copytree(MODIFIED_DIR, TEMP_DIR, dirs_exist_ok=True)
    for dir_path in process_dir(TEMP_DIR):
        dest_path = dir_path[:-1]
        rel_path = os.path.relpath(dest_path, TEMP_DIR)
        backup_path = os.path.join(BACKUP_DIR, rel_path)
        print(f"Processing \"{rel_path}\"...")
        try:
            update_file(dir_path, dest_path, backup_path)
        except ValueError as e:
            print(str(e))
        finally:
            shutil.rmtree(dir_path)
        print("")
    print(f"Overwriting game files...")
    subprocess.run(f"robocopy /s /move /njh /njs /ndl \"{TEMP_DIR}\" .", stdout=subprocess.DEVNULL)
    print("Done!")
