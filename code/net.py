#### @Chao Huang(huangchao09@zju.edu.cn).
### Accustomed from https://github.com/huangmozhilv/u2net_torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# u2net3d(3D U-squared Net): universally applicable unet3d.
# This code was partially inspired by nnU_Net and residual adapter paper(https://github.com/srebuffi/residual_adapters/).

# series_adapter and parallel_adapter are from https://github.com/srebuffi/residual_adapters/.
# separable_adapter is the proposed adapter in our U2Net.


def norm_act(nchan, only='both'):
    norm = nn.InstanceNorm3d(nchan, affine=True)
    # act = nn.ReLU() # activation
    act = nn.LeakyReLU(negative_slope=1e-2)
    if only=='norm':
        return norm
    elif only=='act':
        return act
    else:
        return nn.Sequential(norm, act)


class conv1x1(nn.Module):
    def __init__(self, inChans, outChans=None, stride=1, padding=0):
        super(conv1x1, self).__init__()
        self.op1 = nn.Conv3d(inChans, inChans, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.op1(x)
        return out

class dwise(nn.Module):
    def __init__(self, inChans, kernel_size=3, stride=1, padding=1):
        super(dwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, inChans, kernel_size=kernel_size, stride=stride, padding=padding, groups=inChans)
        self.op1 = norm_act(inChans,only='both')

    def forward(self, x):
        out = self.conv1(x)
        out = self.op1(out)
        return out

class pwise(nn.Module):
    def __init__(self, inChans, outChans, kernel_size=1, stride=1, padding=0):
        super(pwise, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        return out

class conv_unit(nn.Module):
    '''
    variants of conv3d+norm by applying adapter or not.
    '''
    def __init__(self, args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1, second=0):
        super(conv_unit, self).__init__()
        self.stride = stride

        if self.stride != 1:
            self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        elif self.stride == 1:
            if args.trainMode != 'universal':
                self.conv = nn.Conv3d(inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.dwise = nn.ModuleList([dwise(inChans) for i in range(nb_tasks)])
                self.pwise = pwise(inChans, outChans)

        self.op = nn.ModuleList([norm_act(outChans, only='norm') for i in range(nb_tasks)])

    def forward(self, args, task_idx, x):
        if self.stride != 1:
            out = self.conv(x)
            out = self.op[task_idx](out)
            return out
        elif self.stride == 1:
            if args.trainMode != 'universal'
                out = self.conv(x)
                out = self.op[task_idx](out)
                return out
            else:
                out = self.dwise[task_idx](x)
                out = self.pwise(out)
                out = self.op[task_idx](out)
                return out

class InputTransition(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, base_outChans):
        super(InputTransition, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, base_outChans, kernel_size=3, stride=1, padding=1),
            norm_act(base_outChans)
        )

    def forward(self, x):
        out = self.op1(x)
        return out

class DownSample(nn.Module):
    def __init__(self, args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(DownSample, self).__init__()
        self.op1 = conv_unit(args, nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")

    def forward(self, args, task_idx, x):
        out = self.op1(args, task_idx, x)
        out = self.act1(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(DownBlock, self).__init__()
        self.op1 = conv_unit(args, nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")
        self.op2 = conv_unit(args, nb_tasks, outChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act2 = norm_act(outChans, only="act")

    def forward(self, args, task_idx, x):
        out = self.op1(args, task_idx, x)
        out = self.act1(out)
        out = self.op2(args, task_idx, out)
        out = self.act2(x + out) 
        return out


def Upsample3D(scale_factor=(2,2,2)):
    '''
    task specific
    '''
    upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest') 
    return upsample

class UnetUpsample(nn.Module):
    def __init__(self, args, nb_tasks, inChans, outChans, up_stride=(2,2,2)):
        super(UnetUpsample, self).__init__()
        self.upsamples = nn.ModuleList(
            [Upsample3D(scale_factor=up_stride) for i in range(nb_tasks)]
        )
        self.op = conv_unit(args, nb_tasks, inChans, outChans, kernel_size=3,stride=1, padding=1)
        self.act = norm_act(outChans, only='act')

    def forward(self, args, task_idx, x):
        out = self.upsamples[task_idx](x)
        out = self.op(args, task_idx, out)
        out = self.act(out)
        return out

class UpBlock(nn.Module):
    def __init__(self, args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1):
        super(UpBlock, self).__init__()
        self.op1 = conv_unit(args, nb_tasks, inChans, outChans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act1 = norm_act(outChans, only="act")
        self.op2 = conv_unit(args, nb_tasks, outChans, outChans, kernel_size=1, stride=1, padding=0)
        self.act2 = norm_act(outChans, only="act")

    def forward(self, args, task_idx, x, up_x):
        out = self.op1(args, task_idx, x)
        out = self.act1(out)
        out = self.op2(args, task_idx, out)
        out = self.act2(out)
        return out

class DeepSupervision(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, num_class, up_stride=(2,2,2)):
        super(DeepSupervision, self).__init__()
        self.op1 = nn.Sequential(
            nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0),
            norm_act(num_class)
        ) 
        self.op2 = Upsample3D(scale_factor=up_stride)

    def forward(self, x, deep_supervision):
        if deep_supervision is None:
            out = self.op1(x)
        else:
            out = torch.add(self.op1(x), deep_supervision)
        out = self.op2(out)
        return out

class OutputTransition(nn.Module):
    '''
    task specific
    '''
    def __init__(self, inChans, num_class):
        super(OutputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, num_class, kernel_size=1, stride=1, padding=0)
       
    def forward(self, x, deep_supervision=None):
        out = self.conv1(x)
        if deep_supervision is None:
            return out
        else:
            out = torch.add(out, deep_supervision)
            return out

def num_pool2stride_size(num_pool_per_axis):
    max_num = max(num_pool_per_axis)
    stride_size_per_pool = list()
    for i in range(max_num):
        unit = [1,2]
        stride_size_per_pool.append((unit[i<num_pool_per_axis[0]], unit[i<num_pool_per_axis[1]], unit[i<num_pool_per_axis[2]]))
    return stride_size_per_pool

class segmentor(nn.Module):
    def __init__(self, args, inChans_list=[2], base_outChans=16, num_class_list=[4]):
        '''
        Args:
        One or more tasks could be input at once. So lists of inital model settings are passed.
            inChans_list: a list of num_modality for each input task.
            base_outChans: outChans of the inputTransition, i.e. inChans of the first layer of the shared backbone of the universal model.
            depth: depth of the shared backbone.
        '''
        super(segmentor, self).__init__()
        
        nb_tasks = len(num_class_list)

        self.depth = max(args.stride) + 1
        stride_sizes = num_pool2stride_size(args.stride)

        self.in_tr_list = nn.ModuleList(
            [InputTransition(inChans_list[j], base_outChans) for j in range(nb_tasks)]
        )

        outChans_list = list()
        self.down_blocks = nn.ModuleList() 
        self.down_samps = nn.ModuleList()
        self.down_pads = list()

        inChans = base_outChans
        for i in range(self.depth):
            outChans = base_outChans * (2**i)
            outChans_list.append(outChans)
            self.down_blocks.append(DownBlock(args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1))
        
            if i != self.depth-1:
                pads = list()
                for j in stride_sizes[i][::-1]:
                    if j == 2:
                        pads.extend([0,1])
                    elif j == 1:
                        pads.extend([1,1])
                self.down_pads.append(pads) 
                self.down_samps.append(DownSample(args, nb_tasks, outChans, outChans*2, kernel_size=3, stride=tuple(stride_sizes[i]), padding=0))
                inChans = outChans*2
            else:
                inChans = outChans

        self.up_samps = nn.ModuleList([None] * (self.depth-1))
        self.up_blocks = nn.ModuleList([None] * (self.depth-1))
        self.dSupers = nn.ModuleList() 
        for i in range(self.depth-2, -1, -1):
            self.up_samps[i] = UnetUpsample(args, nb_tasks, inChans, outChans_list[i], up_stride=stride_sizes[i])

            self.up_blocks[i] = UpBlock(args, nb_tasks, outChans_list[i]*2, outChans_list[i], kernel_size=3,stride=1, padding=1)

            if i < 3 and i > 0:
                self.dSupers.append(nn.ModuleList(
                    [DeepSupervision(outChans_list[i], num_class_list[j], up_stride=tuple(stride_sizes[i-1])) for j in range(nb_tasks)]
                ))

            inChans = outChans_list[i]

        self.out_tr_list = nn.ModuleList(
            [OutputTransition(inChans, num_class_list[j]) for j in range(nb_tasks)]
        )
        

    def forward(self, args, task_idx, x):
        deep_supervision = None

        with autocast(enabled=False):
            out = self.in_tr_list[task_idx](x)

        down_list = list()
        for i in range(self.depth):
            out = self.down_blocks[i](args, task_idx, out)
            if i != self.depth-1:
                down_list.append(out)
                out = F.pad(out,tuple(self.down_pads[i]), mode="constant", value=0)
                out = self.down_samps[i](args, task_idx, out)
        
        idx = 0
        for i in range(self.depth-2, -1, -1):
            out = self.up_samps[i](args, task_idx, out)
            up_x = out
            out = torch.cat((out, down_list[i]), dim=1)
            out = self.up_blocks[i](args, task_idx, out, up_x)

            if i < 3 and i > 0:
                deep_supervision = self.dSupers[idx][task_idx](out, deep_supervision)
                idx += 1
        out = self.out_tr_list[task_idx](out, deep_supervision)
        
        return out

class critic(nn.Module):
    def __init__(self, args, inChans_numClass, base_outChans, num_class_list):
        super(critic, self).__init__()

        nb_tasks = len(num_class_list)

        self.depth = max(args.stride) + 1
        stride_sizes = num_pool2stride_size(args.stride)

        self.in_tr_list = nn.ModuleList(
            [InputTransition(inChans_numClass[j], base_outChans) for j in range(nb_tasks)]
        ) 

        outChans_list = list()
        self.down_blocks = nn.ModuleList()
        self.down_samps = nn.ModuleList()
        self.down_pads = list()

        inChans = base_outChans
        for i in range(self.depth):
            outChans = base_outChans * (2**i)
            outChans_list.append(outChans)
            self.down_blocks.append(DownBlock(args, nb_tasks, inChans, outChans, kernel_size=3, stride=1, padding=1))
        
            if i != self.depth-1:
                pads = list()
                for j in stride_sizes[i][::-1]:
                    if j == 2:
                        pads.extend([0,1])
                    elif j == 1:
                        pads.extend([1,1])
                self.down_pads.append(pads) 
                self.down_samps.append(DownSample(args, nb_tasks, outChans, outChans*2, kernel_size=3, stride=tuple(stride_sizes[i]), padding=0))
                inChans = outChans*2
            else:
                inChans = outChans

    def forward(self, args, task_idx, x):
        with autocast(enabled=False):
            out = self.in_tr_list[task_idx](x)

        output = out.view(1, -1)

        for i in range(self.depth):
            out = self.down_blocks[i](args, task_idx, out)
            if i != self.depth-1:
                out = F.pad(out, tuple(self.down_pads[i]), mode='constant', value=0)
                out = self.down_samps[i](args, task_idx, out)
            output = torch.cat([output, out.view(1, -1)], -1)

        del out

        return output