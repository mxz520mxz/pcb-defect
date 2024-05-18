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
from easydict import EasyDict as edict
import pylab as plt
from tqdm import tqdm
import json
import click

Image.MAX_IMAGE_PIXELS = None

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def get_images_width(img_dir):
    widths = set(Image.open(img_f).size[0] for img_f in glob(img_dir+'/*.jpg'))
    assert len(widths) == 1, f"multiple widths {widths}"
    return next(iter(widths))

@click.command()
@click.option('-c', '--cfg', default=osp.join(base_dir, 'config/config.toml'))
@click.option('--img0_dir', default='images0')
@click.option('--ymz_defects0', default='ymz_defects0')
@click.option('--lst', default='list.txt')
@click.option('--verify_dir', default='verify_ymz_defects')
def main(cfg, img0_dir, ymz_defects0, lst, verify_dir):
    C = edict(toml.load(cfg))

    os.makedirs(verify_dir, exist_ok=True)
    
    # get tpls info
    tsks = read_list(lst, C.target.cam_mapping)
    tsks_list = []
    for i in tsks:
        tsks_dict = {}
        tsks_dict.update({
            'name':i[0],
            'tpl':i[1],
            'face':i[2],
            'cam':i[3]})
        tsks_list.append(tsks_dict)
    
    for i in os.listdir(ymz_defects0):
        print(i)
        ymz_defects = json.load(open(osp.join(ymz_defects0,i)))
        print('ymz_defects',ymz_defects)

        for pro_info in tsks_list:
            
            # get vcrop 
            if pro_info['name'] == ymz_defects['name']:
                tpl = pro_info['tpl']
                face = pro_info['face']
                cam = pro_info['cam']
                print(ymz_defects['name'],tpl,face,cam)

                # defect
                img = cv2.imread(osp.join(img0_dir,ymz_defects['name']))

                for defect in ymz_defects['objects']:
                    print('defect',defect)
                    centroid = defect['centroid']
                    bbox = defect['bbox']
                    x, y = centroid
                    x0, y0, x1, y1 = bbox
                
                    # verify image0
                    print(img.shape)
                    cv2.rectangle(img,(x0,y0),(x1,y1),(255,255,255),10)

                cv2.imwrite(osp.join(verify_dir,ymz_defects['name']),img)

        # raise
    # raise

if __name__ == "__main__":
    main()
