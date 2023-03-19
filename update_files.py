import os
import shutil
import subprocess
from functools import partial

PAR_TOOL = os.path.join(os.path.dirname(__file__), "ParTool.exe")
PAR_TOOL_ARGS = "add -c 1"
MODIFIED_DIR = os.path.abspath("modified_")
BACKUP_DIR = os.path.abspath("backup_")

echo = partial(print, end="")

def update_file(dir_path, par_path, backup_path=None):
    if not os.path.isfile(par_path):
        raise ValueError(f"No file found at \"{par_path}\"")

    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if not os.path.isfile(backup_path):
        shutil.copy2(par_path, backup_path)

    # execute the external program and capture the output
    try:
        os.remove(par_path)
        command_string = f"{PAR_TOOL} {PAR_TOOL_ARGS} \"{backup_path}\" \"{dir_path}\" \"{par_path}\""
        with subprocess.Popen(command_string, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as prc:
            while prc.poll() is None:
                out = prc.stdout.readline()
                echo(out.decode())
            err = prc.stderr.read()
            echo(err.decode())
    except:
        shutil.copy2(backup_path, par_path)
        raise

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
    for dir_path in process_dir(MODIFIED_DIR):
        rel_path = os.path.relpath(dir_path, MODIFIED_DIR)
        par_path = os.path.join(os.getcwd(), rel_path[:-1])
        backup_path = os.path.join(BACKUP_DIR, rel_path[:-1])
        print(f"\"{dir_path}\" --> \"{par_path}\"")
        try:
            update_file(dir_path, par_path, backup_path)
        except ValueError as e:
            print(str(e))
        print("")
