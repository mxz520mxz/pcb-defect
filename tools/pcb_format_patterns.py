#!/usr/bin/env python
import os
import os.path as osp

import numpy as np
from glob import glob
from PIL import Image
import cv2
import click

Image.MAX_IMAGE_PIXELS = None

@click.command()
@click.option('--roi_extend', default=32)
@click.argument('src')
def main(roi_extend, src):
    kernel = np.ones((roi_extend, roi_extend), np.uint8)

    for pattern in os.listdir(src):
        cur_src = osp.join(src, pattern)

        mr_dir = osp.join(cur_src, 'match_region')
        assert osp.exists(mr_dir)
        roi_dir = osp.join(cur_src, 'roi')
        if not osp.exists(roi_dir):
            os.makedirs(roi_dir, exist_ok=True)

            mr_files = glob(osp.join(mr_dir, '*.png'))
            for mr_f in mr_files:
                mask = np.asarray(Image.open(mr_f))

                roi_mask = cv2.dilate(mask, kernel, iterations=1)
                dst_f = osp.join(roi_dir, osp.basename(mr_f))
                cv2.imwrite(dst_f, roi_mask)

if __name__ == "__main__":
    main()
