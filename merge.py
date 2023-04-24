import os
import sys
import shutil
import tarfile
import lzma
import numpy as np
from functools import reduce
from itertools import combinations
from PIL import Image
from scale import amd_cas

IMPROVED_MERGE = True
COMPRESS_DEST = False

if __name__ == "__main__":
    addr = sys.argv[1] if len(sys.argv) > 1 else os.path.curdir
    mdirs = os.listdir(addr)
    dest_basedir = os.path.join(addr, "merged")
    try:
        for root, _, files in os.walk(mdirs[0]):
            rel_root = os.path.relpath(root, mdirs[0])
            dest_dir = os.path.join(dest_basedir, rel_root)
            os.makedirs(dest_dir, exist_ok=True)
            for f in files:
                source_paths = [os.path.join(d, rel_root, f) for d in mdirs]
                dest_path = os.path.join(dest_dir, f)
                images = []
                for p in source_paths:
                    if os.path.exists(p):
                        images.append(Image.open(p))
                    else:
                        print(f"Warning: File \"{p}\" not found.")
                try:
                    num_images = len(images)
                    if num_images == 1:
                        out_im = images[0]
                    else:
                        if IMPROVED_MERGE:
                            images_np = list(map(np.array, images))
                            diffs = {(i, j): np.abs(images_np[j] - images_np[i]) for i, j in combinations(range(num_images), 2) if i < j}
                            dists = [sum(diffs[tuple(sorted((i, j)))] for j in range(num_images) if j != i) for i in range(num_images)]
                            weights = [1 / (d / num_images + 1) for d in dists]
                            weighted = np.round(sum(w * im for w, im in zip(weights, images)) / sum(weights))
                            out_im = amd_cas(weighted)
                        else:
                            fn_ = lambda a, b: (b[0], Image.blend(b[1], a[1], 1/b[0]))
                            _, out_im = reduce(fn_, enumerate(images, 1))
                    out_im.save(dest_path)
                finally:
                    for img in images:
                        img.close()
    except:
        shutil.rmtree(dest_basedir, ignore_errors=True)
    
    if COMPRESS_DEST:
        TAR_XZ_FILENAME = dest_basedir.rstrip(".").rstrip("/") + ".tar.xz"
        xz_file = lzma.LZMAFile(TAR_XZ_FILENAME, mode='w')
        try:
            with tarfile.open(mode='w', fileobj=xz_file) as tar_xz_file:
                tar_xz_file.add(dest_basedir)
        finally:
            xz_file.close()