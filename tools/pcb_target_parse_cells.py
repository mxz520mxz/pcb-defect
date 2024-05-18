#!/usr/bin/env python
import os
import os.path as osp
import sys

base_dir = osp.join(osp.dirname(__file__), '..')
sys.path.insert(0, base_dir)

from deeppcb.stitch import inverse_project_cells
from easydict import EasyDict as edict
import json
import click

@click.command()
@click.option('--tpl_cells', default='template/stitched/cells.json')
@click.option('--tforms', default='stitched/Ts.json')
@click.option('-o', '--out_f', default='stitched/cells.json')
@click.option('--min_inv_proj_overlap', default=0.02)
def main(tpl_cells, tforms, out_f, min_inv_proj_overlap):
    tpl_C = edict(json.load(open(tpl_cells)))
    C_T = edict(json.load(open(tforms)))

    assert C_T.image_width == tpl_C.image_width
    ref_width = tpl_C.image_width

    Ts = C_T.golden_Ts_wc
    cells = tpl_C.cells

    tile_cells =  inverse_project_cells(cells, Ts, ref_width, min_inv_proj_overlap)

    out = tpl_C.copy()
    out['tile_cells'] = tile_cells

    json.dump(out, open(out_f, 'w'), indent=2)


if __name__ == "__main__":
    main()
