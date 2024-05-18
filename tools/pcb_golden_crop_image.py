#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from functools import partial
from PIL import Image
from glob import glob
import json
from multiprocessing import Pool
from deeppcb.base import resize, imsave, Image
from deeppcb.utils import run_parallel
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def process_one(img_name, src, dst, crop, name_dic):
    name, ext = osp.splitext(img_name)
    name = name_dic.get(name, name)
    dst_f = osp.join(dst, name+ext)
    if osp.exists(dst_f):
        return

    img = Image.open(osp.join(src, img_name))
    imw, imh = img.size

    y0, y1 = crop[name]['vrange']
    img_crop = img.crop((0, y0, img.width, y1))

    os.makedirs(osp.dirname(dst_f), exist_ok=True)
    imsave(dst_f, img_crop)

@click.command()
@click.option('--crop', default='vcrop.json')
@click.option('--name_mapping', default='name_mapping.txt')
@click.option('--jobs', default=5)
@click.option('--debug', is_flag=True)
@click.argument('src')
@click.argument('dst')
def main(crop, name_mapping, jobs, debug, src, dst):
    name_dic = {}
    if name_mapping:
        for l in open(name_mapping):
            k, v = l.split()
            name_dic[k] = v

    crop = json.load(open(crop))
    tsks = os.listdir(src)
    assert tsks
    worker = partial(process_one, src=src, dst=dst, crop=crop, name_dic=name_dic)
    run_parallel(worker, tsks, jobs, debug)

if __name__ == "__main__":
    main()
