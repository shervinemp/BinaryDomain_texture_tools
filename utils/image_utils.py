import os
import subprocess
from PIL import Image


def load_image(path: str) -> Image.Image:
    im = Image.open(path)
    if os.path.exists((p := os.path.splitext(path))[0] + "_alpha." + p[1]):
        alpha = Image.open(p[0] + "_alpha." + p[1]).convert("L")
        im.putalpha(alpha)
    return im


def get_format_tag(info: dict) -> str:
    flags = dict_get(info, "Pixel Format.Flags")[1]
    tag_list = []
    if "DDPF_FOURCC" in flags:
        fourcc = dict_get(info, "Pixel Format.FourCC")
        tag_list.append(fourcc.split()[0].strip("'"))
    else:
        formats = [f.split("DDPF_")[-1] for f in flags.keys()]
        tag_list.extend(formats)
    swizzle = dict_get(info, "Pixel Format.Swizzle")
    if swizzle is not None:
        tag_list.append(swizzle.split()[0].strip("'"))
    if dict_get(info, "Caps.Caps 2.DDSCAPS2_CUBEMAP_ALL_FACES"):
        tag_list.append("CUBEMAP")

    # Remove all ASCII control characters (from \x00 to \x1F)
    tag = "_".join(tag_list)
    tag = "".join(c for c in tag if ord(c) >= 32)

    return tag


def get_ddsinfo(path: str) -> dict:
    proc = subprocess.Popen(
        f'nvddsinfo "{path}"', stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    info = parse_ddsinfo(proc.stdout.read().decode().split("\n"))
    return info


def parse_ddsinfo(lines, *, return_count=False) -> dict:
    lines = list(filter(len, lines))
    info = {}
    lines_parsed = 0
    while len(lines):
        h = 1
        if ":" in lines[0]:
            item = list(filter(len, (x.strip() for x in lines[0].split(":"))))
            if len(lines) > 1 and any(lines[1].startswith(ws) for ws in ("\t", " ")):
                ws = lines[1][0]
                w = 1
                for c in lines[1][1:]:
                    if c == ws:
                        w += 1
                    else:
                        break
                indent = ws * w
                for ln in lines[1:]:
                    if ln.startswith(indent):
                        h += 1
                    else:
                        break
                parsed, _ = parse_ddsinfo(
                    [x[w:] for x in lines[1:h]], return_count=True
                )
                if len(item) == 1 and h:
                    info[item[0]] = parsed
                else:
                    info[item[0]] = (item[1], parsed)
            else:
                if len(item) == 1:
                    info[item[0]] = {}
                else:
                    v = item[1]
                    info[item[0]] = int(v) if v.isnumeric() else v
        else:
            info[lines[0].strip()] = True
        lines_parsed += h
        lines = lines[h:]
    return (info, lines_parsed) if return_count else info


def dict_get(d: dict, path: str) -> any:
    path = path.split(".")
    for i, k in enumerate(path):
        if k not in d:
            return None
        d = d[k]
        if isinstance(d, tuple) and len(d) == 2:
            if i + 1 != len(path):
                d = d[1]
    return d
