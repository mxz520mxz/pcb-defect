import os
import random
import argparse
import time
import numpy as np
import utils
import torch
import sys

from networks import network
from networks import losses


parser = argparse.ArgumentParser()

parser.add_argument('--img-list')
parser.add_argument('--img-prefix')
parser.add_argument('--img-suffix')
parser.add_argument('--atlas')
parser.add_argument('--model-dir',default='models')
parser.add_argument('--multichannel',action='store_true')

parser.add_argument('--gpu',default='0')
parser.add_argument('--batch-size',type=int,default=1)
parser.add_argument('--epochs',type=int,default=100)
parser.add_argument('--steps-per-epoch',type=int,default=100)
parser.add_argument('--load-model')
parser.add_argument('--initial-epoch',type=int,default=0)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--cudnn-nondet',action='store_true')

parser.add_argument('--enc',type=int,nargs='+')
parser.add_argument('--dec',type=int,nargs='+')
parser.add_argument('--int-steps',type=int,default=7)
parser.add_argument('--int-downsize',type=int,default=2)
parser.add_argument('--bidir',action='store_true')

parser.add_argument('--image-loss',default='mse')
parser.add_argument('--lambda',type=float,dest='weight',default=0.01)

args = parser.parse_args()

bidir = args.bidir

add_feat_axis = not args.multichannel

[fixed_img_train,moving_img_train,fixed_img_val,moving_img_val,fixed_img_test,moving_img_test] = utils.load_data('../DeepPCB/data')

generator = utils.data_generator(fixed_img_train,moving_img_train)

inshape = next(generator)[0][0].shape[1:-1]

model_dir = args.model_dir
os.makedirs(model_dir,exist_ok=True)

gpus = args.gpu.split(',')
nb_gpus = len(gpus)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
assert np.mod(args.batch_size,nb_gpus) == 0,'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size,nb_gpus)

enc_nf = args.enc if args.enc else [16,32,32,32]
dec_nf = args.dec if args.dec else [32,32,32,32,32,16,16]

if args.load_model:
    model = network.VxmDense.load(args.load_model,device)
else:
    model = network.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf,dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

if nb_gpus > 1:
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)

if args.image_loss == 'ncc':
    image_loss_func = losses.NCC().loss
elif args.image_loss == 'mse':
    image_loss_func = losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc"')

if bidir:
    losses = [image_loss_func,image_loss_func]
    weights = [0.5,0.5]
else:
    losses = [image_loss_func]
    weights = [1]

for epoch in range(args.initial_epoch,args.epochs):

    if epoch % 20 == 0:
        model.save(os.path.join(model_dir,'%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):
        step_start_time = time.time()
        inputs,y_true = next(generator)

        inputs = [torch.from_numpy(d).to(device).float().permute(3,0,1,2) for d in inputs]
        y_true = [torch.from_numpy(d).to(device).float().permute(3,0,1,2) for d in y_true]

        y_pred = model(*inputs)


        loss = 0
        loss_list = []
        for n,loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n],y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_step_time.append(time.time() - step_start_time)

    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
    print(' - '.join((epoch_info, time_info, loss_info)), flush=True)

model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))     
