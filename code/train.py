import torch
import os
import json

from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch import autograd

import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import data
import loss
import net

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def weights_init(m):
    #Author https://github.com/huangmozhilv/u2net_torch
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def no_improv(args, losses):
    print('NO IMPROV')
    print(losses)
    num = 0
    for i in range(1,len(losses)):
        if abs(losses[i]) > abs(losses[0]):
            num += 1
    if num == args.window-1:
        return True
    else:
        return False

def train_segmentor(args, task_archive, segmentor, checkpoint=None):

    #cuda model
    if args.cuda:
        segmentor.cuda()
        print('segmentor on cuda', next(segmentor.parameters()).is_cuda)

    #set optimizer
    optimizer = torch.optim.Adam(segmentor.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume_ckp != '':
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #set losses
    dice_loss = loss.MulticlassDiceLoss()

    #last x epoch losses placeholder
    if args.resume_ckp != '':
        losses = checkpoint['losses']
        print(losses)
    else:
        losses = []

    #set epoch and iteration
    if args.resume_ckp != '':
        start_epoch = checkpoint['epoch'] + 1
        task_idx = checkpoint['task_idx']
        iterations = checkpoint['iterations']
    else:
        start_epoch = 0
        task_idx = 0
        iterations = {'Task01_BrainTumour': 0, 'Task02_Heart': 0, 'Task03_Liver': 0, 'Task04_Hippocampus': 0, 'Task05_Prostate': 0, 'Task06_Lung': 0, 'Task07_Pancreas': 0, 'Task08_HepaticVessel': 0, 'Task09_Spleen': 0, 'Task10_Colon': 0}

    for epoch in range(start_epoch, args.max_epoch):
        start.record()
        #import time
        
        #set train mode
        segmentor.train()

        #epoch losses and metrics placeholders
        num_epoch = 0
        num_epoch_list = [0] * len(args.tasks)
        #dloss
        dloss_epoch = 0
        mae_epoch = 0
        dloss_epoch_list = [0] * len(args.tasks)
        mae_epoch_list = [0] * len(args.tasks)
        dcoef_epoch = 0
        dcoef_epoch_list = [0] * len(args.tasks)
        dcoefs_epoch = defaultdict(list)
        maes_epoch = defaultdict(list)

        for step in range(args.step_per_epoch):

            #set task_idx, task and iterations
            if task_idx == len(args.tasks):
                task_idx = 0

            task = args.tasks[task_idx]

            if len(task_archive[task]['fold0']['train']) == iterations[task]:
                iterations[task] = 0

            #get data
            img, lab = data.get_data(args, task_archive[task]['fold0']['train'][iterations[task]])

            #fuse cancer to organ
            if task in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                print('FUSE CANCER TO ORGAN', np.unique(lab))
                lab[lab == 2] = 1
                print('FUSE CANCER TO ORGAN', np.unique(lab))

            print(epoch, step, iterations[task], task, task_archive[task]['fold0']['train'][iterations[task]], args.lr)

            #reshape img
            img = np.moveaxis(img, 0, -1)

            #pad data
            img, lab = data.pad_data(img, lab, size=args.input_size)

            img, lab = data.randCrop_data(img, lab, size=args.input_size)

            img, lab = data.trans_data(img, lab)
            print('NUM CLASS IN PATCH', np.unique(lab))

            print(img.shape, lab.shape)

            #cuda data
            if args.cuda:
                img = img.cuda()
                lab = lab.cuda()

            #set gradients to zero (before or after segmentor?)
            optimizer.zero_grad()

            #train model
            #with autograd.detect_anomaly():
            with autocast():
                output = segmentor(args, task_idx, img)
                print('output', torch.isnan(output).sum())
                output = F.softmax(output, dim=1)
                print('output', torch.isnan(output).sum())

                #set num_class
                num_class = output.shape[1]
                print('NUM CLASS', num_class)
                dloss, dcoef, all_dcoefs = dice_loss(args, output, lab, num_class=num_class)
                label = lab.clone()
                maes = []
                for cl in range(num_class):
                    label[lab == cl] = 1
                    label[lab != cl] = 0
                    maes.append(torch.mean(torch.abs(output[:,cl,:,:,:]-label)))
                mae = np.mean([mae.item() for mae in maes])
        
            print(dloss.item(), dcoef.item(), [dcoef.item() for dcoef in all_dcoefs])
            print(mae, [mae.item() for mae in maes])

            #save losses and metrics
            data.save_results(args, 'dloss', epoch, step, task, 'all', dloss.item())
            data.save_results(args, 'dcoef', epoch, step, task, 'all', dcoef.item())
            data.save_results(args, 'mae', epoch, step, task, 'all', mae)
            for i in range(len(all_dcoefs)):
                data.save_results(args, 'dcoefs', epoch, step, task, i, all_dcoefs[i].item())
                data.save_results(args, 'maes', epoch, step, task, i, maes[i])

            scale = 65536.0
            (dloss*scale).backward()
            #end.record()
            #torch.cuda.synchronize()
            #print('backward', start.elapsed_time(end))
            #start.record()
            optimizer.step()
            #end.record()
            #torch.cuda.synchronize()
            #print('step', start.elapsed_time(end))

        #increment epoch losses and metrics
        #start.record()
        num_epoch += 1
        num_epoch_list[task_idx] += 1
        #dice
        dloss_epoch += dloss.item()
        mae_epoch += mae
        dloss_epoch_list[task_idx] += dloss.item()
        mae_epoch_list[task_idx] += mae
        dcoef_epoch += dcoef.item()
        dcoef_epoch_list[task_idx] += dcoef.item()
        if task not in dcoefs_epoch.keys():
            dcoefs_epoch[task] = [0 for i in range(num_class)]
            maes_epoch[task] = [0 for i in range(num_class)]
        for i in range(len(all_dcoefs)):
            dcoefs_epoch[task][i] += all_dcoefs[i].item()
            maes_epoch[task][i] += maes[i]

        #increment task_idx and iterations
        iterations[task] += 1
        task_idx += 1

        #save epoch losses and metrics
        #dice
        dloss_epoch /= num_epoch
        data.save_results(args, 'epoch_dloss', epoch, None, 'all', 'all', dloss_epoch)
        dcoef_epoch /= num_epoch
        data.save_results(args, 'epoch_dcoef', epoch, None, 'all', 'all', dcoef_epoch)
        mae_epoch /= num_epoch
        data.save_results(args, 'epoch_mae', epoch, None, 'all', 'all', mae_epoch)
        for idx in range(len(args.tasks)):
            task = args.tasks[idx]
            dloss_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_dloss', epoch, None, task, 'all', dloss_epoch_list[idx])
            dcoef_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_dcoef', epoch, None, task, 'all', dcoef_epoch_list[idx])
            mae_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_mae', epoch, None, task, 'all', mae_epoch_list[idx])
            for i in range(len(dcoefs_epoch[task])):
                dcoefs_epoch[task][i] /= num_epoch_list[idx]
                data.save_results(args, 'epoch_dcoefs', epoch, None, task, i, dcoefs_epoch[task][i])
                maes_epoch[task][i] /= num_epoch_list[idx]
                data.save_results(args, 'epoch_maes', epoch, None, task, i, maes_epoch[task][i])

        #update losses
        if len(losses) < args.window:
            losses.append(dloss_epoch)
        else:
            losses.pop(0)
            losses.append(dloss_epoch)

        #save model
        ckp_dir = args.base_dir + 'models/{}/{}.pth.tar'.format(args.modelName, 'epoch_{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'segmentor': segmentor,
            'segmentor_state_dict': segmentor.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'iterations': iterations,
            'task_idx': task_idx,
            'args': args
        }, ckp_dir)

        #lr decay and early stopping
        if len(losses) == args.window:
            if no_improv(args,losses):
                if args.lr < 10**(-8) and epoch:
                    print('EARLY STOPPING')
                    print(losses)
                    break
                else:
                    args.lr /= 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                    print('NEW LEARNING RATE')
                    print(losses)
                    print(args.lr)

        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print('epoch time', time)

        with open(args.base_dir + 'models/{}/train_info.txt'.format(args.modelName), 'a') as f:
            f.write('epoch: {} lr: {} scale: {} time_ms: {} time_min: {}\n'.format(epoch, args.lr, scale, time, (time/60)/1000))

def train_segan(args, task_archive, segmentor, critic, checkpoint=None, checkpoint_critic=None):

    #cuda model
    if args.cuda:
        segmentor.cuda()
        print('segmentor on cuda', next(segmentor.parameters()).is_cuda)
        critic.cuda()
        print('critic on cuda', next(critic.parameters()).is_cuda)

    #set optimizer
    optimizer = torch.optim.Adam(segmentor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_critic = torch.optim.Adam(segmentor.parameters(), lr=args.lr_critic, weight_decay=args.weight_decay)

    if args.resume_ckp != '':
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_critic.load_state_dict(checkpoint_critic['optimizer_state_dict'])

    #set losses
    dice_loss = loss.MulticlassDiceLoss()

    #last x epoch losses placeholder
    if args.resume_ckp != '':
        losses = checkpoint['losses']
        losses_critic = checkpoint_critic['losses_critic']
        print(losses)
        print(losses_critic)
    else:
        losses = []
        losses_critic = []

    #set epoch and iteration
    if args.resume_ckp != '':
        start_epoch = checkpoint['epoch'] + 1
        task_idx = checkpoint['task_idx']
        iterations = checkpoint['iterations']
    else:
        start_epoch = 0
        task_idx = 0
        iterations = {'Task01_BrainTumour': 0, 'Task02_Heart': 0, 'Task03_Liver': 0, 'Task04_Hippocampus': 0, 'Task05_Prostate': 0, 'Task06_Lung': 0, 'Task07_Pancreas': 0, 'Task08_HepaticVessel': 0, 'Task09_Spleen': 0, 'Task10_Colon': 0} #FINISH

    for epoch in range(start_epoch, args.max_epoch):
        start.record()
        
        #set segmentor train mode
        segmentor.train()

        #epoch losses and metrics placeholders
        num_epoch = 0
        num_epoch_list = [0] * len(args.tasks)
        #dice
        dloss_epoch = 0
        dloss_critic_epoch = 0
        mae_epoch = 0
        dloss_epoch_list = [0] * len(args.tasks)
        dloss_critic_epoch_list = [0] * len(args.tasks)
        mae_epoch_list = [0] * len(args.tasks)
        dcoef_epoch = 0
        dcoef_critic_epoch = 0
        dcoef_epoch_list = [0] * len(args.tasks)
        dcoef_critic_epoch_list = [0] * len(args.tasks)
        dcoefs_epoch = defaultdict(list)
        dcoefs_critic_epoch = defaultdict(list)
        maes_epoch = defaultdict(list)
        #joint
        critic_minus_epoch = 0
        critic_minus_epoch_list = [0] * len(args.tasks)
        critic_plus_epoch = 0
        critic_plus_epoch_list = [0] * len(args.tasks)
        joint_epoch = 0
        joint_epoch_list = [0] * len(args.tasks)
        joint_critic_epoch = 0
        joint_critic_epoch_list = [0] * len(args.tasks)

        for step in range(args.step_per_epoch):

            #set task_idx, task and iterations
            if task_idx == len(args.tasks):
                task_idx = 0

            task = args.tasks[task_idx]

            if len(task_archive[task]['fold0']['train']) == iterations[task]:
                iterations[task] = 0

            #get data
            img, lab = data.get_data(args, task_archive[task]['fold0']['train'][iterations[task]])

            #fuse cancer to organ
            if task in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                print('FUSE CANCER TO ORGAN', np.unique(lab))
                lab[lab == 2] = 1
                print('FUSE CANCER TO ORGAN', np.unique(lab))

            print(epoch, step, iterations[task], task, task_archive[task]['fold0']['train'][iterations[task]], args.lr, args.lr_critic)

            #reshape img
            img = np.moveaxis(img, 0, -1)

            #pad data
            img, lab = data.pad_data(img, lab, size=args.input_size)

            #crop data
            img, lab = data.randCrop_data(img, lab, size=args.input_size)

            #transform data to torch + batch and channel
            img, lab = data.trans_data(img, lab)
            print('NUM CLASS IN PATCH', np.unique(lab))
            print(img.shape, lab.shape)

            #cuda data
            if args.cuda:
                img = img.cuda()
                lab = lab.cuda()

            ####################################################################

            #set critic gradients to zero
            critic.zero_grad()

            #train segmentor
            #with autograd.detect_anomaly():
            with autocast():
                output = segmentor(args, task_idx, img)
                print('output', torch.isnan(output).sum())

                #get segmentor softmax
                output = F.softmax(output, dim=1)
                print('output', torch.isnan(output).sum())

                #set num_class and mod
                num_class = output.shape[1]
                num_mod = img.shape[1]

                dloss, dcoef, all_dcoefs = dice_loss(args, output, lab, num_class=num_class)
            
            print(dloss.item(), dcoef.item(), [dcoef.item() for dcoef in all_dcoefs])
            output = output.detach()

            #save losses and metrics
            #dice
            data.save_results(args, 'dloss_critic', epoch, step, task, 'all', dloss.item())
            data.save_results(args, 'dcoef_critic', epoch, step, task, 'all', dcoef.item())
            for i in range(len(all_dcoefs)):
                data.save_results(args, 'dcoefs_critic', epoch, step, task, i, all_dcoefs[i].item())

            #increment epoch losses and metrics
            #dice
            dloss_critic_epoch += dloss.item()
            dloss_critic_epoch_list[task_idx] += dloss.item()
            dcoef_critic_epoch += dcoef.item()
            dcoef_critic_epoch_list[task_idx] += dcoef.item()
            if task not in dcoefs_critic_epoch.keys():
                dcoefs_critic_epoch[task] = [0 for i in range(num_class)]
            for i in range(len(all_dcoefs)):
                dcoefs_critic_epoch[task][i] += all_dcoefs[i].item()

            ####################################################################

            #get output_masked
            output_masked = torch.empty(1, num_mod*num_class, args.input_size[0], args.input_size[1], args.input_size[2])
            input_mask = img.clone()

            i = 0
            for mod in range(num_mod):
                for cl in range(num_class):
                    output_masked[:,i,:,:,:] = input_mask[:,mod,:,:,:] * output[:,cl,:,:,:]
                    i += 1
            del output

            #cuda data
            if args.cuda:
                output_masked = output_masked.cuda()

            #get target_masked
            target_masked = torch.empty(1, num_mod*num_class, args.input_size[0], args.input_size[1], args.input_size[2])
            output_mask = lab.clone()

            i = 0
            for mod in range(num_mod):
                for cl in range(num_class):
                    output_mask[lab == cl] = 1
                    output_mask[lab != cl] = 0
                    target_masked[:,i,:,:,:] = input_mask[:,mod,:,:,:] * output_mask
                    i += 1
            del output_mask, input_mask

            #cuda data
            if args.cuda:
                target_masked = target_masked.cuda()

            #train critic and get critic loss for each class
            #with autograd.detect_anomaly():
            with autocast():
                output_masked = critic(args, task_idx, output_masked)
                print('output masked', torch.isnan(output_masked).sum())
            #with autograd.detect_anomaly():
            with autocast():
                target_masked = critic(args, task_idx, target_masked)
                print('target masked', torch.isnan(target_masked).sum())
                loss_critic = torch.mean(torch.abs(output_masked-target_masked))
                del output_masked, target_masked

                #calculate joint loss
                loss_joint = - (loss_critic + dloss)

            #save losses and metrics
            data.save_results(args, 'critic_minus', epoch, step, task, 'all', loss_critic.item())
            data.save_results(args, 'joint_critic', epoch, step, task, 'all', loss_joint)

            scale_critic = 65536.0
            #(-loss_critic*scale_critic).backward()
            (loss_joint*scale_critic).backward()
            optimizer_critic.step()
            #scaler_critic.scale(loss_critic).backward()
            #scaler_critic.unscale_(optimizer_critic)
            #scaler_critic.step(optimizer_critic)
            #scaler_critic.update()

            #increment epoch losses and metrics
            critic_minus_epoch += loss_critic.item()
            critic_minus_epoch_list[task_idx] += loss_critic.item()
            joint_critic_epoch += loss_joint.item()
            joint_critic_epoch_list[task_idx] += loss_joint.item()

            #clip parameters in critic
            for p in critic.parameters():
                p.data.clamp_(-0.05,0.05)

            ####################################################################

            #set segmentor gradients to zero
            segmentor.zero_grad()

            #train segmentor
            #with autograd.detect_anomaly():
            with autocast():
                output = segmentor(args, task_idx, img)
                print('output', torch.isnan(output).sum())

                #get softmax
                output = F.softmax(output, dim=1)
                print('output', torch.isnan(output).sum())

                #calculate losses and metrics
                dloss, dcoef, all_dcoefs = dice_loss(args, output, lab, num_class=num_class)
            
            label = lab.clone()
            maes = []
            for cl in range(num_class):
                label[lab == cl] = 1
                label[lab != cl] = 0
                maes.append(torch.mean(torch.abs(output[:,cl,:,:,:]-label)))
            mae = np.mean([mae.item() for mae in maes])
            
            print(dloss.item(), dcoef.item(), [dcoef.item() for dcoef in all_dcoefs])
            print(mae, [mae.item() for mae in maes])

            #save losses and metrics
            #dice
            data.save_results(args, 'dloss', epoch, step, task, 'all', dloss.item())
            data.save_results(args, 'dcoef', epoch, step, task, 'all', dcoef.item())
            data.save_results(args, 'mae', epoch, step, task, 'all', mae)
            for i in range(len(all_dcoefs)):
                data.save_results(args, 'dcoefs', epoch, step, task, i, all_dcoefs[i].item())
                data.save_results(args, 'maes', epoch, step, task, i, maes[i])

            #increment epoch losses and metrics
            num_epoch += 1
            num_epoch_list[task_idx] += 1
            #dice
            dloss_epoch += dloss.item()
            mae_epoch += mae
            dloss_epoch_list[task_idx] += dloss.item()
            mae_epoch_list[task_idx] += mae
            dcoef_epoch += dcoef.item()
            dcoef_epoch_list[task_idx] += dcoef.item()
            if task not in dcoefs_epoch.keys():
                dcoefs_epoch[task] = [0 for i in range(num_class)]
                maes_epoch[task] = [0 for i in range(num_class)]
            for i in range(len(all_dcoefs)):
                dcoefs_epoch[task][i] += all_dcoefs[i].item()
                maes_epoch[task][i] += maes[i]

            ####################################################################

            #get output_masked
            output_masked = torch.empty(1, num_mod*num_class, args.input_size[0], args.input_size[1], args.input_size[2])
            input_mask = img.clone()

            i = 0
            for mod in range(num_mod):
                for cl in range(num_class):
                    output_masked[:,i,:,:,:] = input_mask[:,mod,:,:,:] * output[:,cl,:,:,:]
                    i += 1
            del output

            #cuda data
            if args.cuda:
                output_masked = output_masked.cuda()

            #get target_masked
            target_masked = torch.empty(1, num_mod*num_class, args.input_size[0], args.input_size[1], args.input_size[2])
            output_mask = lab.clone()

            i = 0
            for mod in range(num_mod):
                for cl in range(num_class):
                    output_mask[lab == cl] = 1
                    output_mask[lab != cl] = 0
                    target_masked[:,i,:,:,:] = input_mask[:,mod,:,:,:] * output_mask
                    i += 1
            del output_mask, input_mask

            #cuda data
            if args.cuda:
                target_masked = target_masked.cuda()

            #train critic and get critic loss for each class
            #with autograd.detect_anomaly():
            with autocast():
                output_masked = critic(args, task_idx, output_masked)
                print('output masked', torch.isnan(output_masked).sum())
            #with autograd.detect_anomaly():
            with autocast():
                target_masked = critic(args, task_idx, target_masked)
                print('target masked', torch.isnan(target_masked).sum())
                loss_critic = torch.mean(torch.abs(output_masked-target_masked))
                del output_masked, target_masked

                #calculate joint loss
                loss_joint = loss_critic + dloss

            #save losses and metrics
            data.save_results(args, 'critic_plus', epoch, step, task, 'all', loss_critic.item())
            data.save_results(args, 'joint', epoch, step, task, 'all', loss_joint)

            scale = 65536.0
            (loss_joint*scale).backward() #loss_joint=segan, dloss=segmentor
            optimizer.step()
            #scaler.scale(loss_joint).backward()
            #scaler.unscale_(optimizer)
            #scaler.step(optimizer)
            #scalet.update()

            #increment epoch losses and metrics
            critic_plus_epoch += loss_critic.item()
            critic_plus_epoch_list[task_idx] += loss_critic.item()
            joint_epoch += loss_joint.item()
            joint_epoch_list[task_idx] += loss_joint.item()

            ####################################################################

            #increment task_idx and iterations
            iterations[task] += 1
            task_idx += 1

        #save epoch losses and metrics
        #dice
        dloss_epoch /= num_epoch
        data.save_results(args, 'epoch_dloss', epoch, None, 'all', 'all', dloss_epoch)
        dloss_critic_epoch /= num_epoch
        data.save_results(args, 'epoch_critic_dloss', epoch, None, 'all', 'all', dloss_critic_epoch)
        dcoef_epoch /= num_epoch
        data.save_results(args, 'epoch_dcoef', epoch, None, 'all', 'all', dcoef_epoch)
        dcoef_critic_epoch /= num_epoch
        data.save_results(args, 'epoch_critic_dcoef', epoch, None, 'all', 'all', dcoef_critic_epoch)
        mae_epoch /= num_epoch
        data.save_results(args, 'epoch_mae', epoch, None, 'all', 'all', mae_epoch)
        for idx in range(len(args.tasks)):
            task = args.tasks[idx]
            dloss_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_dloss', epoch, None, task, 'all', dloss_epoch_list[idx])
            dloss_critic_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_critic_dloss', epoch, None, task, 'all', dloss_critic_epoch_list[idx])
            dcoef_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_dcoef', epoch, None, task, 'all', dcoef_epoch_list[idx])
            dcoef_critic_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_critic_dcoef', epoch, None, task, 'all', dcoef_critic_epoch_list[idx])
            mae_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_mae', epoch, None, task, 'all', mae_epoch_list[idx])
            for i in range(len(dcoefs_epoch[task])):
                dcoefs_epoch[task][i] /= num_epoch_list[idx]
                data.save_results(args, 'epoch_dcoefs', epoch, None, task, i, dcoefs_epoch[task][i])
                dcoefs_critic_epoch[task][i] /= num_epoch_list[idx]
                data.save_results(args, 'epoch_critic_dcoefs', epoch, None, task, i, dcoefs_critic_epoch[task][i])
                maes_epoch[task][i] /= num_epoch_list[idx]
                data.save_results(args, 'epoch_maes', epoch, None, task, i, maes_epoch[task][i])
        #joint
        critic_minus_epoch /= num_epoch
        data.save_results(args, 'epoch_critic_minus', epoch, None, 'all', 'all', critic_minus_epoch)
        critic_plus_epoch /= num_epoch
        data.save_results(args, 'epoch_critic_plus', epoch, None, 'all', 'all', critic_plus_epoch)
        joint_epoch /= num_epoch
        data.save_results(args, 'epoch_joint', epoch, None, 'all', 'all', joint_epoch)
        joint_critic_epoch /= num_epoch
        data.save_results(args, 'epoch_critic_joint', epoch, None, 'all', 'all', joint_critic_epoch)
        for idx in range(len(args.tasks)):
            task = args.tasks[idx]
            critic_minus_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_critic_minus', epoch, None, task, 'all', critic_minus_epoch_list[idx])
            critic_plus_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_critic_plus', epoch, None, task, 'all', critic_plus_epoch_list[idx])
            joint_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_joint', epoch, None, task, 'all', joint_epoch_list[idx])
            joint_critic_epoch_list[idx] /= num_epoch_list[idx]
            data.save_results(args, 'epoch_critic_joint', epoch, None, task, 'all', joint_critic_epoch_list[idx])

        #update losses
        if len(losses) < args.window:
            losses.append(joint_epoch)
            losses_critic.append(critic_minus_epoch)
        else:
            losses.pop(0)
            losses.append(joint_epoch)
            losses_critic.pop(0)
            losses_critic.append(critic_minus_epoch)

        #save segmentor
        ckp_dir = args.base_dir + 'models/{}/{}.pth.tar'.format(args.modelName, 'epoch_{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'segmentor': segmentor,
            'segmentor_state_dict': segmentor.state_dict(),
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'iterations': iterations,
            'task_idx': task_idx,
            'args': args
        }, ckp_dir)

        #save critic
        ckp_dir = args.base_dir + 'models/{}/{}.pth.tar'.format(args.modelName, 'critic_epoch_{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'critic': critic,
            'critic_state_dict': critic.state_dict(),
            'optimizer': optimizer_critic,
            'optimizer_state_dict': optimizer_critic.state_dict(),
            'losses_critic': losses_critic,
            'iterations': iterations,
            'task_idx': task_idx,
            'args': args
        }, ckp_dir)

        #lr decay and early stopping
        if len(losses) == args.window:
            if no_improv(args,losses):
                if args.lr < 10**(-8):
                    print('EARLY STOPPING')
                    print(losses)
                    break
                else:
                    args.lr /= 2
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = args.lr
                    print('NEW LEARNING RATE')
                    print(losses)
                    print(args.lr)
            if no_improv(args,losses_critic):
                args.lr_critic /= 2
                for param_group in optimizer_critic.param_groups:
                    param_group['lr'] = args.lr_critic
                print('NEW LEARNING RATE')
                print(losses_critic)
                print(args.lr_critic)

        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print('epoch time', time)

        with open(args.base_dir + 'models/{}/train_info.txt'.format(args.modelName), 'a') as f:
            f.write('epoch: {} lr: {} lr_critic: {} scale: {} scale_critic: {} time_ms: {} time_min: {}\n'.format(epoch, args.lr, args.lr_critic, scale, scale_critic, time, (time/60)/1000))