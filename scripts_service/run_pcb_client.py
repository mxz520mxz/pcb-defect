#!/usr/bin/env python
import sys
import os.path as osp
cur_d = osp.dirname(__file__)
sys.path.insert(0, cur_d+'/../')

from greenthread.monkey import monkey_patch; monkey_patch()

from greenthread.green import *
from loguru import logger
import traceback
import sys

from mqsrv.client import make_client

def main(broker_url):
    client = make_client(conn=broker_url)

    caller = client.get_caller('pcb_alg_rpc_queue')
    pubber = client.get_pubber('pcb_alg_event_queue')

    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20211023T193027S004_cam1_10584.jpg')
    # exc, result = caller.pcb_a_run('/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20211024T013639S465_cam1_11417.jpg')
    
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202107S806_cam1_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202120S339_cam2_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202132S558_cam3_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202144S155_cam4_8530_concatenate_pillow.jpg')
    # # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202153S799_cam5_8530_concatenate_pillow.jpg')
    # # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202203S765_cam6_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202212S973_cam7_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202223S569_cam8_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202234S638_cam9_8530_concatenate_pillow.jpg')
    # exc, result = caller.pcb_a_run(event_type='path',im_p='/home/vision/users/dengsx/pcb_py/data_bench_server/images0/20220409T202245S494_cam10_8530_concatenate_pillow.jpg')
    
    exc, result = caller.pcb_a_run()
    print(result)

    if exc:
        print ("="*10)
        print ("Exception from Server, traceback on server:")
        traceback.print_exception(*exc)

        print ("="*10)
        print ("Reraising")
        # raise exc[1]
    
    client.release()

if __name__ == '__main__':
    main('pyamqp://guest:guest@10.20.20.111:5672/')
