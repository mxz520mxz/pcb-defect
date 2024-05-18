#!/usr/bin/env python
import os.path as osp
from gimpfu import *

def file_name(f):
    return osp.splitext(osp.basename(f))[0]

def is_same(f1, f2):
    return osp.exists(f1) and osp.exists(f2) and osp.samefile(f1, f2)

def pcb_load_layers(image, image_dir, segmap_dir, foreign_dir, deviation_dir):
    fname = image.filename
    name = file_name(fname)
    cur_dir = osp.dirname(fname)

    img_f = osp.join(cur_dir, image_dir).format(name)
    foreign_f = osp.join(cur_dir, foreign_dir).format(name)
    deviation_f = osp.join(cur_dir, deviation_dir).format(name)
    segmap_f = osp.join(cur_dir, segmap_dir).format(name)

    load_layers = {}
    if is_same(fname, img_f):
        if osp.exists(segmap_f):
            load_layers['segmap'] = segmap_f
        if osp.exists(foreign_f):
            load_layers['foreign'] = foreign_f

    elif is_same(fname, segmap_f):
        if osp.exists(img_f):
            load_layers['img'] = img_f
        if osp.exists(foreign_f):
            load_layers['foreign'] = foreign_f

    elif is_same(fname, deviation_f):
        if osp.exists(img_f):
            load_layers['img'] = img_f
        if osp.exists(deviation_f):
            load_layers['deviation'] = deviation_f

    elif is_same(fname, foreign_f):
        if osp.exists(img_f):
            load_layers['img'] = img_f
        if osp.exists(segmap_f):
            load_layers['segmap'] = segmap_f

    for k, v in load_layers.items():
        # pdb.gimp_message("load {} {}".format(k, v))
        layer = pdb.gimp_file_load_layer(image, v)
        pdb.gimp_layer_set_name(layer, k)
        pdb.gimp_image_insert_layer(image, layer, None, 0)
		# pdb.gimp_image_lower_item(image, layer)


register(
    "pcb_load_layers",
    "PCB load layers",
    "load different layers into one",
    "vision",
    "vision",
    "2022",
    "PCB Load Layers",
    "*",      # Alternately use RGB, RGB*, GRAY*, INDEXED etc.
    [
        (PF_IMAGE, "image", "current image", None),
        (PF_STRING, "image_dir", "image relative directory", '../images_1x/{}.jpg'),
        (PF_STRING, "segmap_dir", "segmap relative directory", '../ood_segmaps_1x_color/{}.jpg'),
        (PF_STRING, "foreign_dir", "foreign relative directory", '../verify_foreigns/{}.png'),
        (PF_STRING, "deviation_dir", "deviation relative directory", '../verify_deviations/{}.png'),
    ],
    [],
    pcb_load_layers, menu="<Image>/PCB" )

main()
