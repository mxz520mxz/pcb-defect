#!/usr/bin/env python
import os
import os.path as osp
import re
import glob
import click

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

@click.command()
@click.option('--pattern', default='(?P<ts>.*)_cam(?P<idx>.*)_(?P<board>.*).*')
@click.argument('src')
def main(pattern, src):
    img_dic = {
        int(re.match(pattern, f).groupdict()['idx']): file_name(f) for f in os.listdir(src)
    }
    img_list = sorted(img_dic.items())

    out = {}
    for idx, (k, v) in enumerate(img_list):
        print (v, idx)

if __name__ == "__main__":
    main()
