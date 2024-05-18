import os
import os.path as osp
import toml
from easydict import EasyDict as edict

enhance_tool = osp.expanduser('~/deep3d/repo/deepenhance/tools')

cfg_f = osp.join(osp.dirname(__file__), '../config/config.toml')
cfg = edict(toml.load(open(cfg_f))).base

WIDTH0 = cfg.width0
ALIGN_WIDTH = cfg.align_width
ALIGN_SCALE = cfg.align_scale
STITCH_WIDTH = cfg.stitch_width
ICON_WIDTH = cfg.icon_width

ENHANCE_ITERS = 4
