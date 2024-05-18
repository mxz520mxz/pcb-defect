import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid',grid)

    def forward(self,src,flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:,i,...] = 2 * (new_locs[:,i,...]/(shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0,2,3,1)
            new_locs = new_locs[...,[1,0]]
        else:
            # print('shape is not 2,please check it ')
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src,new_locs,align_corners=True,mode=self.mode)
    

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0,'nsteps should be >=0,found:%d' % nsteps

        self.nsteps = nsteps
        self.scale = 1.0/(2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self,vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec,vec)

        return vec
    


class ResizeTransform(nn.Module):
    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        return x

        


