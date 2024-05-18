#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import cv2
import numpy as np
from deeppcb.base import imsave
from deeppcb.stitch import stitch_images
import click
import json

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

@click.command()
@click.option('--tform', default='stitched/Ts.json')
@click.argument('src')
@click.argument('dst')
def main(tform, src, dst):
    Ts = json.load(open(tform))
    o_w, o_h = Ts['golden_size']
    Ts = Ts['golden_Ts_wc']

    for pattern in os.listdir(src):
        cur_src = osp.join(src, pattern)

        imgs = {
            file_name(i): cv2.imread(osp.join(cur_src, i),
                                     cv2.IMREAD_GRAYSCALE)
            for i in os.listdir(cur_src)
        }

        cur_Ts = {k: Ts[k] for k in imgs}

        ret = stitch_images(imgs, cur_Ts, (o_w, o_h, 1), order='nearest', fusion='overlap')

        dst_f = dst.format(pattern=pattern)
        os.makedirs(osp.dirname(dst_f), exist_ok=True)
        imsave(dst_f, ret.img)

if __name__ == "__main__":
    main()
