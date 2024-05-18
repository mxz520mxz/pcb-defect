#!/usr/bin/env python
import os
import os.path as osp

import sys
import rust_pipeline

cur_dir = osp.dirname(__file__)
base_dir = osp.join(cur_dir, '../')
sys.path.insert(0, base_dir)

os.environ['GREEN_BACKEND'] = 'gevent'
from greenthread.monkey import monkey_patch; monkey_patch()

import time
import click
import toml
from queue import Queue
from greenthread.green import *
from loguru import logger
from mqsrv.base import get_rpc_exchange
from mqsrv.server import MessageQueueServer, run_server, make_server
from mqsrv.client import make_client
from pydaemon import Daemon
from easydict import EasyDict as edict

# from deeppcb.base import Image
# from deeppcb.target import read_list, file_name

# from tools.pcb_cad_target_infer_2x import run_stage_crop, run_stage_filter, run_stage_resize_align, run_stage_estimate_camera_align, run_stage_align_camera, run_stage_seg_gmm, run_stage_seg_ood, run_stage_detect_foreigns, run_stage_detect_deviations, run_stage_detect_foreigns_deviations
# from src.pcb_sort_image_names import sort_image_names
# from src.pcb_load_origin_images import LoadImages

class PCB:
    rpc_prefix = 'pcb_a'
    def __init__(self, config):
        self.queue = Queue()
        self.config = config
        self.workspace = osp.join(base_dir,'../','data_bench_server')
        self.templates = 'templates'
        # self.li = LoadImages()
        # self.crop = run_stage_crop
        # self.filter = run_stage_filter
        # self.resize = run_stage_resize_align
        # self.transform = run_stage_estimate_camera_align
        # self.align = run_stage_align_camera
        # self.gmm = run_stage_seg_gmm
        # self.ood = run_stage_seg_ood
        # self.foreigns = run_stage_detect_foreigns
        # self.deviations = run_stage_detect_deviations
        # self.detectall = run_stage_detect_foreigns_deviations
        
        addr = "amqp://guest:guest@localhost:5672/"        
        client = make_client()
        caller = client.get_caller('pcb_alg_rpc_queue')
        self.pubber = client.get_pubber('pcb_controller_event_queue')

        self.is_processing = False
        
    def setup(self):
        while 1:
            if self.queue.qsize != 0 and not self.is_processing:
                self.infer(path=self.queue.get())
            time.sleep(0.1)
            # print("pop one image")
    
    def run(self, event_type, im_p):
        print(im_p)        
        self.queue.put(im_p)
        print(self.queue.qsize())
        
    # def infer(self, path, debug=True,
    #         save_crop='',
    #         save_filter='',
    #         save_resize_align='',
    #         save_transform='',
    #         save_aligned_images='',
    #         save_seg_gmm='',
    #         save_seg_ood='',
    #         save_foreigns='pkl_result/save_foreigns',
    #         save_deviations='pkl_result/save_deviations',
    #         save_foreigns_patches='defect_local/save_foreigns_patches',
    #         save_deviations_patches='defect_local/save_deviations_patches',
    #         verify_transform='',
    #         verify_seg_gmm='',
    #         verify_seg_ood='',
    #         verify_foreigns='defect_global/verify_foreigns',
    #         verify_deviations='defect_global/verify_deviations',
    #         stage='detect_foreigns_deviations', 
    #         zoom=1,
    #         ):
        
    #     self.is_processing = True

    #     C = self.config
    #     workspace = self.workspace
    #     templates = self.templates
    #     print('workspace',workspace)
                        
    #     # workspace = osp.join(osp.dirname(path),'../')
        
    #     img_name = osp.basename(path)
    #     img_n = img_name.split('.')[0]
    #     pcb_type = img_n.split('_')[2]
    #     pcb_cam = img_n.split('_')[1]
    #     print(pcb_type)
    #     print(pcb_cam)
    #     print(pcb_cam[3:])
    #     # raise
    #     templates = osp.join(workspace, templates)

    #     if pcb_cam[3:] == '1' or pcb_cam[3:] == '2'  or pcb_cam[3:] == '3' or pcb_cam[3:] == '4' or pcb_cam[3:] == '5':
    #         save_crop = osp.join(workspace, 'result', pcb_type, 'A', save_crop)
    #         save_filter = osp.join(workspace, 'result', pcb_type, 'A', save_filter)
    #         save_resize_align = osp.join(workspace, 'result', pcb_type, 'A', save_resize_align)
    #         save_transform = osp.join(workspace, 'result', pcb_type, 'A', save_transform)
    #         save_aligned_images = osp.join(workspace, 'result', pcb_type, 'A', save_aligned_images)
    #         save_seg_gmm = osp.join(workspace, 'result', pcb_type, 'A', save_seg_gmm)
    #         save_seg_ood = osp.join(workspace, 'result', pcb_type, 'A', save_seg_ood)
    #         save_foreigns = osp.join(workspace, 'result', pcb_type, 'A', save_foreigns)
    #         save_deviations = osp.join(workspace, 'result', pcb_type, 'A', save_deviations)
    #         save_foreigns_patches = osp.join(workspace, 'result', pcb_type, 'A', save_foreigns_patches)
    #         save_deviations_patches = osp.join(workspace, 'result', pcb_type, 'A', save_deviations_patches)
    #         verify_transform = osp.join(workspace, 'result', pcb_type, 'A', verify_transform)
    #         verify_seg_gmm = osp.join(workspace, 'result', pcb_type, 'A', verify_seg_gmm)
    #         verify_seg_ood = osp.join(workspace, 'result', pcb_type, 'A', verify_seg_ood)
    #         verify_foreigns = osp.join(workspace, 'result', pcb_type, 'A', verify_foreigns)
    #         verify_deviations = osp.join(workspace, 'result', pcb_type, 'A', verify_deviations)
    #         # verify_global = osp.join(workspace, 'result', pcb_type, 'A', verify_global)

    #     if pcb_cam[3:] == '6' or pcb_cam[3:] == '7'  or pcb_cam[3:] == '8' or pcb_cam[3:] == '9' or pcb_cam[3:] == '10':
    #         save_crop = osp.join(workspace, 'result', pcb_type, 'B', save_crop)
    #         save_filter = osp.join(workspace, 'result', pcb_type, 'B', save_filter)
    #         save_resize_align = osp.join(workspace, 'result', pcb_type, 'B', save_resize_align)
    #         save_transform = osp.join(workspace, 'result', pcb_type, 'B', save_transform)
    #         save_aligned_images = osp.join(workspace, 'result', pcb_type, 'B', save_aligned_images)
    #         save_seg_gmm = osp.join(workspace, 'result', pcb_type, 'B', save_seg_gmm)
    #         save_seg_ood = osp.join(workspace, 'result', pcb_type, 'B', save_seg_ood)
    #         save_foreigns = osp.join(workspace, 'result', pcb_type, 'B', save_foreigns)
    #         save_deviations = osp.join(workspace, 'result', pcb_type, 'B', save_deviations)
    #         save_foreigns_patches = osp.join(workspace, 'result', pcb_type, 'B', save_foreigns_patches)
    #         save_deviations_patches = osp.join(workspace, 'result', pcb_type, 'B', save_deviations_patches)
    #         verify_transform = osp.join(workspace, 'result', pcb_type, 'B', verify_transform)
    #         verify_seg_gmm = osp.join(workspace, 'result', pcb_type, 'B', verify_seg_gmm)
    #         verify_seg_ood = osp.join(workspace, 'result', pcb_type, 'B', verify_seg_ood)
    #         verify_foreigns = osp.join(workspace, 'result', pcb_type, 'B', verify_foreigns)
    #         verify_deviations = osp.join(workspace, 'result', pcb_type, 'B', verify_deviations)
    #         # verify_global = osp.join(workspace, 'result', pcb_type, 'B', verify_global)

    #     cls_info = read_list(osp.join(workspace, 'list.txt'), C.target.cam_mapping, return_dict=True)

    #     img_f = path
    #     name = file_name(img_name)
    #     info = cls_info[name]

    #     start_img = Image.open(img_f)
    #     start_img = start_img.resize((start_img.size[0] // zoom , start_img.size[1] // zoom),Image.ANTIALIAS)     

    #     ctx = edict({
    #         'C': C,
    #         'name': name,
    #         'img_name': osp.basename(img_f),
    #         'tpl_dir': osp.join(workspace, templates, info.board),
    #         'board': info.board,
    #         'cam_id': info.cam_id,

    #         'save_crop': save_crop,
    #         'save_filter': save_filter,
    #         'save_resize_align': save_resize_align,
    #         'save_transform': save_transform,
    #         'save_aligned_images': save_aligned_images,
    #         'save_seg_gmm': save_seg_gmm,
    #         'save_seg_ood': save_seg_ood,
    #         'save_foreigns': save_foreigns,
    #         'save_deviations': save_deviations,
    #         'save_foreigns_patches': save_foreigns_patches,
    #         'save_deviations_patches': save_deviations_patches,

    #         'verify_transform': verify_transform,
    #         'verify_seg_gmm': verify_seg_gmm,
    #         'verify_seg_ood': verify_seg_ood,
    #         'verify_foreigns': verify_foreigns,
    #         'verify_deviations': verify_deviations,
            
    #         'zoom': zoom,

    #         'start.img': start_img,
    #     })
                
    #     # # server mode
    #     # ctx, stage, img = self.crop(ctx)
    #     # print(ctx, stage, img)
        
    #     # ori mode 
    #     stage_processors = {
    #         'start': lambda x: x,
    #         'crop': self.crop,
    #         'filter': self.filter,
    #         'resize_align': self.resize,
    #         'estimate_camera_align': self.transform,
    #         'align_camera': self.align,
    #         'seg_gmm': self.gmm,
    #         'seg_ood': self.ood,
    #         'detect_foreigns': self.foreigns,
    #         'detect_deviations': self.deviations,
    #         'detect_foreigns_deviations': self.detectall,
    #     }

    #     stage = 'detect_foreigns_deviations'

    #     verify_deviations, verify_foreigns = stage_processors[stage](ctx)


    #     if pcb_cam[3:] == '1' or pcb_cam[3:] == '2'  or pcb_cam[3:] == '3' or pcb_cam[3:] == '4' or pcb_cam[3:] == '5':
    #         result={
    #             "path":path,
    #             "defect_global":osp.join(workspace, 'result', pcb_type, 'A', 'defect_global'),
    #             "defect_local":osp.join(workspace, 'result', pcb_type, 'A', 'defect_local'),
    #             "pkl_result":osp.join(workspace, 'result', pcb_type, 'A', 'pkl_result')
    #         }
    #     if pcb_cam[3:] == '6' or pcb_cam[3:] == '7'  or pcb_cam[3:] == '8' or pcb_cam[3:] == '9' or pcb_cam[3:] == '10':
    #         result={
    #             "path":path,
    #             "defect_global":osp.join(workspace, 'result', pcb_type, 'B', 'defect_global'),
    #             "defect_local":osp.join(workspace, 'result', pcb_type, 'B', 'defect_local'),
    #             "pkl_result":osp.join(workspace, 'result', pcb_type, 'B', 'pkl_result')
    #         }


    #     print('verify_deviations:',verify_deviations)
    #     print(f'publish {path}')
    #     self.pubber("result", result)
    #     self.is_processing = False

    #     return result
    
    
    def infer(self, path, debug=True):
        
        self.is_processing = True
        print(f'publish {path}')
        
        result = rust_pipeline.run(path,2)
        
        
        self.pubber("result", result)
        self.is_processing = False

        return result
        
    def set_rpc_prefix(self, rpc_prefix):
        PCB.rpc_prefix = rpc_prefix

    def teardown(self):
        print (f"{PCB.rpc_prefix} teared down")


def run(config_f):
        
    config = edict(toml.load(config_f))
    pcb = PCB(config)
    pcb.set_rpc_prefix('pcb_a')
    addr = "amqp://guest:guest@localhost:5672/"
    rpc_queue = 'pcb_alg_rpc_queue'         
    evt_queue = 'pcb_alg_event_queue'

    server = make_server(
        conn = addr,
        rpc_routing_key=rpc_queue,
        event_routing_keys=evt_queue,
    )

    server.register_context(pcb)
    # server.register_rpc(pcb.run)
    server.register_rpc(pcb.run, 'call_alg')
    server.register_event_handler('call_alg', pcb.run)

    run_server(server)
    
def run_pcb_server(config):
    try:
        run(config)
    except Exception as e:
        logger.exception(e)

@click.group()
@click.option('--pidfile', default = osp.join(base_dir, 'tmp/mqsrv/run_pcb_server.pid'))
@click.pass_context
def cli(ctx, pidfile):
    ctx.ensure_object(dict)
    print(pidfile)
    ctx.obj['pidfile'] = pidfile

@cli.command()
@click.option('--config', default = osp.join(base_dir, 'config/config.toml'))
@click.option('--logfile', default = osp.join(base_dir, 'tmp/mqsrv/run_pcb_server.log'))
@click.option('--fg', is_flag=True)
@click.option('--restart', is_flag=True)
@click.pass_context
def start(ctx, config, logfile, fg, restart):
    
    config = osp.abspath(config)
    logfile = osp.abspath(logfile)
    pidfile = ctx.obj['pidfile']
    name = "run_pcb_server"
    os.makedirs(osp.dirname(pidfile), exist_ok=True)
    
    if fg:
        run_pcb_server(config)
        exit()

    if logfile and not fg:
        fp = open(logfile, 'w')
        logger.add(fp, level='DEBUG')

    daemon = Daemon(pidfile, name=name)
    if restart:
        daemon.restart(run_pcb_server, config)
    else:
        if osp.exists(pidfile):
            os.remove(pidfile)
        daemon.start(run_pcb_server, config)

@cli.command()
@click.pass_context
def stop(ctx):
    pidfile = ctx.obj['pidfile']
    daemon = Daemon(pidfile)
    daemon.stop()


if __name__ == "__main__":
    cli()
    
    