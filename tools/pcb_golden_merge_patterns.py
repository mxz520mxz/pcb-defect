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
from PIL import Image
from tqdm import tqdm
import pylab as plt
import click

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--crop', default='vcrop.json')
@click.option('--name_mapping', default='name_mapping.txt')
@click.argument('src')
@click.argument('dst')
def main(cfg, crop, name_mapping, src, dst):
    name_dic = {}
    if name_mapping:
        for l in open(name_mapping):
            k, v = l.split()
            name_dic[k] = v

    crop = json.load(open(crop))

    C = edict(toml.load(cfg))

    for pattern in  os.listdir(src):
        cur_src = osp.join(src, pattern)
        p = osp.join(src, pattern, '*', '*.png')
        files = list(set((osp.basename(i), Image.open(i).size) for i in glob(p)))
        for img_name, (imw, imh) in tqdm(files):
            name, ext = osp.splitext(img_name)
            name = name_dic.get(name, name)

            y0, y1 = crop[name]['vrange']
            imh = y1 - y0

            canvas = np.zeros((imh, imw), dtype='u1')

            for tp, v in C.pattern.items():
                mask_f = osp.join(cur_src, tp, img_name)
                if not osp.exists(mask_f):
                    continue

                layer = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
                layer = layer[y0:y1, :]

                mask = layer > 128
                canvas[mask] = np.bitwise_or(canvas[mask], v.label)

                if v.extend:
                    kernel = np.ones((v.extend*2+1, v.extend*2+1))
                    layer = cv2.dilate(layer, kernel)

                    mask = layer > 128
                    canvas[mask] = np.bitwise_or(canvas[mask], v.extend_label)

            name, ext = osp.splitext(img_name)
            cur_dst = osp.join(dst, pattern)
            os.makedirs(cur_dst, exist_ok=True)
            dst_f = osp.join(cur_dst, name_dic.get(name, name)+ext)
            cv2.imwrite(dst_f, canvas)

if __name__ == "__main__":
    main()
