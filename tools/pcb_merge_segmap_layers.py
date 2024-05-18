#!/usr/bin/env python
import os
import os.path as osp

base_dir = osp.join(osp.dirname(__file__), '..')

from easydict import EasyDict as edict
import numpy as np
import toml
from glob import glob
import cv2
import json
from tqdm import tqdm
from PIL import Image
import click

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--crop', default='vcrop.json')
@click.option('--name_mapping', default='name_mapping.txt')
@click.argument('src')
@click.argument('dst')
def main(cfg, crop, name_mapping, src, dst):
    os.makedirs(dst, exist_ok=True)

    name_dic = {}
    if name_mapping:
        for l in open(name_mapping):
            k, v = l.split()
            name_dic[k] = v

    crop = json.load(open(crop))

    C = edict(toml.load(cfg))
    img_files = list(set((osp.basename(i), Image.open(i).size) for i in glob(osp.join(src, '*', '*.png'))))

    for img_name, (imw, imh) in tqdm(img_files):
        name, ext = osp.splitext(img_name)
        name = name_dic.get(name, name)

        y0, y1 = crop[name]['vrange']
        imh = y1 - y0
        segmap = np.zeros((imh, imw), dtype='u1')

        for cls_name, v in C.classes.items():
            seg_f = osp.join(src, cls_name, img_name)
            if not osp.exists(seg_f):
                continue

            layer = cv2.imread(seg_f, cv2.IMREAD_GRAYSCALE)
            layer = layer[y0:y1, :]

            mask = layer > 128
            segmap[mask] |= v.label

        dst_f = osp.join(dst, name+ext)
        cv2.imwrite(dst_f, segmap)

if __name__ == "__main__":
    main()
