#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
import PIL.Image as Image
import click
from deeppcb.base import imsave

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('--levels', default=3)
@click.argument('src')
@click.argument('dst')
def main(levels, src, dst):
    files = sorted(
        glob(osp.join(src, '**', '*.jpg'), recursive=True)
        + glob(osp.join(src, '**', '*.png'), recursive=True))

    for f in files:
        img = Image.open(f)
        p = osp.relpath(f, src)
        for lvl in range(1, levels+1):
            scale = 2**lvl
            dst_dir = dst.format(scale=scale)
            dst_f = osp.join(dst_dir, p)
            os.makedirs(osp.dirname(dst_f), exist_ok=True)
            w, h = img.width // 2, img.height // 2
            img = img.resize((w, h))
            imsave(dst_f,img)


if __name__ == "__main__":
    main()
