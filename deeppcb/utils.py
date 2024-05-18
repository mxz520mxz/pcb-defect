from multiprocessing import Pool
from tqdm import tqdm

def run_parallel(worker, tsks, jobs, debug):
    assert len(tsks) > 0
    if debug:
        for tsk in tsks:
            worker(tsk)
    else:
        with Pool(min(len(tsks), jobs)) as p:
            max_ = len(tsks)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap_unordered(worker, tsks)):
                    pbar.update()

def get_zoom(zoom, hint):
    if not zoom:
        zoom = int(hint.split('_')[-1].rstrip('x'))
    return zoom

def get_zoomed_len(l, zoom, min_val=1):
    if isinstance(l, dict):
        return l[f'{zoom}x']
    # return max(l // zoom, min_val)
    import math
    return max(math.ceil(l / zoom), min_val)                                 #

def get_zoomed_area(v, zoom, min_val=1):
    if isinstance(v, dict):
        return v[f'{zoom}x']
    # return max(v // zoom**2, min_val)
    import math
    return max(math.ceil(v // zoom**2), min_val)                             #

def update_zoomed_len(cfg, key, zoom, **kws):
    keys = key.split('.')
    parent = cfg
    for i in keys[:-1]:
        parent = cfg[i]
    key = keys[-1]
    parent[key] = get_zoomed_len(parent[key], zoom, **kws)

def update_zoomed_area(cfg, key, zoom, **kws):
    keys = key.split('.')
    parent = cfg
    for i in keys[:-1]:
        parent = cfg[i]
    key = keys[-1]
    parent[key] = get_zoomed_area(parent[key], zoom, **kws)
