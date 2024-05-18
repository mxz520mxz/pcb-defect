#!/usr/bin/env python
import os
import os.path as osp
from glob import glob
from PIL import Image
import numpy as np
import json
import cv2
from tqdm import tqdm
import click

Image.MAX_IMAGE_PIXELS = None

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

@click.command()
@click.option('--gap', default=300)
@click.option('--name_mapping', default='name_mapping.txt')
@click.option('-r', '--radius', default=5)
@click.argument('bg')
@click.argument('dst')
def main(gap, name_mapping, radius, bg, dst):
    name_dic = {}
    if name_mapping:
        for l in open(name_mapping):
            k, v = l.split()
            name_dic[k] = v

    out = {}

    kernel = np.ones((radius, radius))

    for fname in tqdm(glob(osp.join(bg, '*.png'))):
        name = file_name(fname)
        name = name_dic.get(name, name)
        mask = np.asarray(Image.open(fname))
        cv2.erode(mask, kernel, dst=mask)
        cv2.dilate(mask, kernel, dst=mask)

        ys, xs = np.where(mask > 128)

        min_y = int(max(ys.min() - gap, 0))
        max_y = int(min(ys.max() + gap, mask.shape[0]))

        out[name] = {
            'vrange': [min_y, max_y]
        }

    json.dump(out, open(dst, 'w'), indent=2, sort_keys=True)

if __name__ == "__main__":
    main()
