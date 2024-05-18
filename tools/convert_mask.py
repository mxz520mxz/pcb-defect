#!/usr/bin/env python
import os
import os.path as osp
import sys

cur_dir = osp.dirname(__file__)
sys.path.insert(0, osp.join(cur_dir, '..'))

import click
from glob import glob
from deeppcb.base import Image
import numpy as np
from deeppcb.base import imsave

@click.command()
@click.argument('src')
@click.argument('dst')
def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    classes = os.listdir(src)
    for cls in classes:
        cls_src_dir = osp.join(src, cls)
        cls_dst_dir =  osp.join(dst, cls)

        os.makedirs(cls_dst_dir, exist_ok=True)
        for img_f in glob(osp.join(cls_src_dir, '*.png')):
            dst_f = osp.join(cls_dst_dir, osp.basename(img_f))

            pil = Image.open(img_f)
            alpha = pil.split()[-1]
            imsave(dst_f,alpha)


if __name__ == "__main__":
    main()
