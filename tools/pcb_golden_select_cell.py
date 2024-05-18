#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from easydict import EasyDict as edict
from glob import glob
from deeppcb.base import parse_name
import shutil
import toml
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

# check rx is contained in ry
def is_contained(rx, ry):
    return rx[0] >= ry[0] and rx[2] <= ry[2]

@click.command()
@click.option('--ann', default='annotations/annotation.toml')
@click.argument('src')
@click.argument('dst')
def main(ann, src, dst):
    C = edict(toml.load(open(ann)))
    defect_positions = C.get('defect_positions', [])

    pats = os.listdir(src)
    for pat in pats:
        pat_src = osp.join(src, pat)
        names = os.listdir(osp.join(pat_src, 'images'))
        d = [parse_name(i) for i in names]
        d = [i for i in d if i['pos'] not in defect_positions]
        d = sorted(d, key=lambda x: x['roi'][2] - x['roi'][0])

        strict_results = []
        for idx, i in enumerate(d):
            if any(is_contained(i['roi'], d[j]['roi']) for j in range(idx+1, len(d))):
                continue
            strict_results.append(i)

        print (f"{pat}:")
        for i in strict_results:
            print (f"  {i['name']}")

        for folder in ['images', 'segmaps']:
            for i in strict_results:
                src_dir = osp.join(pat_src, folder)
                if not osp.exists(src_dir):
                    continue

                src_f = osp.join(src_dir, i['name']+'.*')
                src_fs = glob(src_f)
                if not src_fs:
                    continue
                assert len(src_fs) == 1
                src_f = src_fs[0]
                print("copy", src_f)

                cur_dst = osp.join(dst.format(pattern=pat), 'cell_'+folder)
                os.makedirs(cur_dst, exist_ok=True)
                shutil.copy2(src_f, cur_dst)

if __name__ == "__main__":
    main()
