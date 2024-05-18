#!/usr/bin/env python
import os
import os.path as osp
import sys
from tempfile import template

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

import math as M
import numpy as np
import numpy.linalg as npl
from skimage.transform import SimilarityTransform, AffineTransform, ProjectiveTransform
from glob import glob
from PIL import Image, ImageDraw
import networkx as nx
from deeppcb.align_menpo import menpofit_lk
from deeppcb.align_ocv import ocv_find_ecc
from deeppcb.align_edge import align_edge
from deeppcb.base import estimate_transform, transform_img, imsave
from deeppcb.stitch import stitch_images
from deeppcb.target import read_list, process_crop
from deeppcb.base import imread, transform_img, imsave, scale_H

import cv2
import toml
import re
import pickle
from easydict import EasyDict as edict
import pylab as plt
from tqdm import tqdm
import json
import click

Image.MAX_IMAGE_PIXELS = None

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def cut_expand_constant(x0, y0, x1, y1, k):
    x0, y0, x1, y1 = x0-k, y0-k, x1+k, y1+k

    if x0 <= 0:
        x0 = 0
    if y0 <= 0:
        y0 = 0
    if x1 >= 8320:
        x1 = 8320

    return x0, y0, x1, y1

def cut_expand_dynamic(x0, y0, x1, y1):
    w = x1 - x0
    h = y1 -y0
    # print('x0, y0, x1, y1', x0, y0, x1, y1)
    # print('w, h', w, h)
    xx, yy = x0 + w//2, y0 + h//2
    size_level = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    if max(w, h) < 32:
        x0 = xx - 32
        x1 = xx + 32
        y0 = yy - 32
        y1 = yy + 32

        if x0 <= 0:
            x0 = 0
        if y0 <= 0:
            y0 = 0
        if x1 >= 8320:
            x1 = 8320

    else:
        for i in range(len(size_level)-2):
           if max(w, h) > size_level[i] and max(w, h) < size_level[i+1]:
               size = size_level[i+2]
               x0 = xx - size // 2
               x1 = xx + size - size // 2
               y0 = yy - size // 2
               y1 = yy + size - size // 2

               if x0 <= 0:
                   x0 = 0
               if y0 <= 0:
                   y0 = 0
               if x1 >= 8320:
                   x1 = 8320
               break

    print('x0, y0, x1, y1', x0, y0, x1, y1)

    return x0, y0, x1, y1

@click.command()
@click.option('--foreigns_path', default='foreigns')
@click.option('--img_dir', default='trans_smooth_images_orig')
@click.option('--verify_dir', default='foreign_patches')
def main(foreigns_path, img_dir, verify_dir):

    foreign_files = glob(osp.join(foreigns_path,'*'))
    for foreign_f in foreign_files:

        img_foreign = pickle.load(open(foreign_f, 'rb'))
        img_name = file_name(foreign_f)
        print(img_name)
        img = imread(osp.join(img_dir, img_name+'.jpg'))

        detector_config = img_foreign['detector_config']
        objects = img_foreign['objects']
        segmap = img_foreign['segmap']
        groups = img_foreign['groups']
        # print('objects',objects)
        # print('groups',groups)
        # raise
        
        # S
        for k,v in objects.items():
            level = v['level']
            if k in groups['-1']['children'] and (level == 'black' or  level == 'gray'):
                x0, y0, x1, y1 = v['bbox']
                # centroid = v['centroid']
                w = x1 - x0
                h = y1 - y0
                print(x0, y0, x1, y1)
                # x0, y0, x1, y1 = cut_expand_constant(x0, y0, x1, y1,32)
                x0, y0, x1, y1 = cut_expand_dynamic(x0, y0, x1, y1)
                patch = img[y0:y1, x0:x1]
                img_save_path = osp.join(verify_dir, 'images', img_name, level)
                os.makedirs(img_save_path, exist_ok=True)
                imsave(osp.join(img_save_path,img_name+f'-S{k}_{x0, y0, x1, y1}.jpg'),patch)

        # G 
        for k,v in groups.items():
            if k != '-1':
                x0, y0, x1, y1 = v['bbox']
                # centroid = v['centroid']
                level = v['level']

                # x0, y0, x1, y1 = cut_expand_constant(x0, y0, x1, y1,32)
                x0, y0, x1, y1 = cut_expand_dynamic(x0, y0, x1, y1)
                patch = img[y0:y1, x0:x1]
                img_save_path = osp.join(verify_dir, 'images', img_name, level)
                os.makedirs(img_save_path, exist_ok=True)
                imsave(osp.join(img_save_path,img_name+f'-G{k}_{x0, y0, x1, y1}.jpg'),patch)


if __name__ == "__main__":
    main()


