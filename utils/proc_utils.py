from functools import partial
import multiprocessing
import os
import subprocess
import time
from typing import Any, Callable, Iterable, Set


echo = partial(print, end="")


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
                time.sleep(0.05)
            err = prc.stderr.read()
            echo(err.decode())
        except KeyboardInterrupt as e:
            if not silent:
                print(f"Incomplete: {command_string}")
            raise e


def multiproc(fn_: Callable, iter_: Iterable, p: int, chunksize: int = 8) -> Set[Any]:
    if p == 1:
        out = set(map(fn_, iter_))
    else:
        p_count = p if p > 0 else multiprocessing.cpu_count() - p
        with multiprocessing.Pool(p_count) as p:
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
