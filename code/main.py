import argparse
import json
import torch
import os
import shutil
import random
import time

from collections import defaultdict

import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import nibabel as nib

import train
import net
import eval
import plot
import data
import time

#set parser
parser = argparse.ArgumentParser()
#all
parser.add_argument('--mode', default='', type=str, help='train, eval, plot or preproc')
parser.add_argument('--base_dir', default='/content/gdrive/My Drive/bachelor/', type=str, help='base dir (../ or /content/gdrive/My Drive/bachelor/)')
#train
parser.add_argument('--trainMode', default='universal', type=str, help='training mode (independent or universal)')
parser.add_argument('--tasks', default='Task04_Hippocampus', nargs='+', help='tasks to be trained')
parser.add_argument('--modelType', default='segan', type=str, help='model type (segmentor and segan)')
parser.add_argument('--max_epoch', default=500, type=int, help='number of epochs')
parser.add_argument('--step_per_epoch', default=250, type=int, help='number of steps per epoch')
parser.add_argument('--base_outChans', default=16, type=int, help='')
parser.add_argument('--lr', default=3*10**(-4), type=float, help='learning rate used for optimization')
parser.add_argument('--lr_critic', default=3*10**(-4), type=float, help='learning rate used for critic optimization')
parser.add_argument('--weight_decay', default=10**(-5), type=float, help='weight devay used for optimization')
parser.add_argument('--input_size', nargs='+', type=int, help='model input size of image and label')
parser.add_argument('--depth', default=5, type=int, help='number of down and up sampling layers')
parser.add_argument('--modelName', default='test', type=str, help='name of model')
parser.add_argument('--window', default=30, type=int, help='early stopping window size')
parser.add_argument('--stride', nargs='+', type=int, help='used to get stride for model')
parser.add_argument('--object_seg', default=False, type=bool, help='to fuse cancer to organ or not (True or False)')
#resume train
parser.add_argument('--resume_ckp', default='', type=str, help='modelName to load')
parser.add_argument('--resume_epoch', default='', type=str, help='epoch to load')
#transfer train
parser.add_argument('--ckp', default='', type=str, help='modelName to transfer from')
parser.add_argument('--ckp_epoch', default='', type=str, help='epoch to transfer from')
#eval
parser.add_argument('--model_to_eval', default='test', type=str, help='modelName to evaluate')
parser.add_argument('--epoch_to_eval', default='', type=str, help='epoch to evaluate')
parser.add_argument('--eval_epochs', default=0, nargs='+', help='list of epochs to evaluate')
parser.add_argument('--task_idxs', default=0, nargs='+', help='')
#plot
parser.add_argument('--model_to_plot', default='test', type=str, help='modelName to plot')
parser.add_argument('--results', default='epoch_dloss', nargs='+', help='results to be plotted')
#preproc
parser.add_argument('--domains', default='Task04_Hippocampus', nargs='+', help='domains to be preprocessed')
parser.add_argument('--resume_preproc', default='True', type=str, help='False if new preproc, True if resume old preproc')
#set args
args = parser.parse_args()

def transfer_weights(model, old_model):
    #Accustomed from https://github.com/huangmozhilv/u2net_torch
    shared_modules = ['down_blocks', 'up_blocks']
    store_weight3x3 = []
    store_weight1x1 = []
    store_weightNorm = []
    name3x3 = []
    name1x1 = []
    nameNorm = []
    for name, m in old_model.named_modules():
        if any(i in name for i in shared_modules):
            if '.dwise' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==3):
                store_weight3x3.append(m.weight.data)
                name3x3.append(name)
            elif '.dwise' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==1):
                store_weight1x1.append(m.weight.data)
                name1x1.append(name)
            elif '.dwise' not in name and isinstance(m, nn.InstanceNorm3d):
                store_weightNorm.append(m.weight.data)
                nameNorm.append(name)
    element3x3 = 0
    element1x1 = 0
    elementNorm = 0
    for name, m in model.named_modules():
        if any(i in name for i in shared_modules):
            if '.dwise' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==3):
                m.weight.data = store_weight3x3[element3x3]
                m.weight.requires_grad = False # # Freeze shared weights
                element3x3 += 1
            elif '.dwise' not in name and isinstance(m, nn.Conv3d) and (m.kernel_size[0]==1):
                m.weight.data = store_weight1x1[element1x1]
                m.weight.requires_grad = False # # Freeze shared weights
                element1x1 += 1
    del old_model
    return model

if args.mode == 'train':

    #set checkpoint and args
    if args.resume_ckp != '':
        resume_ckp = args.resume_ckp
        resume_epoch = args.resume_epoch
        ckp_dir = args.base_dir + 'models/{}/'.format(args.resume_ckp)
        checkpoint = torch.load(ckp_dir + 'epoch_{}.pth.tar'.format(args.resume_epoch))
        args = checkpoint['args']
        args.resume_ckp = resume_ckp
        args.resume_epoch = resume_epoch
        print('WINDOW', args.window)

    #set checkpoint and args for critic
    if args.resume_ckp != '' and args.modelType == 'segan':
        ckp_dir = args.base_dir + 'models/{}/'.format(args.resume_ckp)
        checkpoint_critic = torch.load(ckp_dir + 'critic_epoch_{}.pth.tar'.format(args.resume_epoch))

    #check if cuda is available
    if torch.cuda.is_available():
        print('CUDA is available')
        args.cuda = True
    else:
        args.cuda = False

    #set tasks when only one task
    if type(args.tasks) is str:
        args.tasks = [args.tasks]

    #make dirs and results ready for curr model
    if args.resume_ckp != '':
        results_dir = args.base_dir + 'results/{}/'.format(args.resume_ckp)
        results = os.listdir(results_dir)
        for result in results:
            lines = open(results_dir + result).readlines()
            for line in lines[::-1]:
                if int(line.split()[1]) > int(args.resume_epoch):
                    del lines[-1]
                else:
                    break
            open(results_dir + result, 'w').writelines(lines)
    else:
        dirs = [args.base_dir + 'results/{}'.format(args.modelName), args.base_dir + 'models/{}'.format(args.modelName)]
        for dir in dirs:
            shutil.rmtree(dir, ignore_errors=True)
            os.makedirs(dir)

    #open task archive
    with open(args.base_dir + '/dataset/fold_splits.json', mode='r') as f:
        task_archive = json.load(f)

    #open dataset info
    dataset_info = defaultdict(dict)
    for task in args.tasks:
        with open(args.base_dir + 'dataset/{}/dataset.json'.format(task), mode='r') as f:
            info_dict = json.load(f)
            try:
                if task in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                    keys = list(info_dict['labels'].keys())
                    for key in keys:
                        if info_dict['labels'][key] in ['cancer', 'tumour']:
                            info_dict['labels'].pop(key)
                    print(info_dict)
            except:
                print('HEY', task)
            dataset_info[task] = info_dict

    #set seed for initializing weights
    torch.manual_seed(102)
    if args.cuda:
        torch.cuda.manual_seed(102)

    #set inChans_list and num_class_list
    inChans_list = [len(dataset_info[task]['modality']) for task in args.tasks]
    num_class_list = [len(dataset_info[task]['labels']) for task in args.tasks]
    inChans_numClass = [inChans_list[i]*num_class_list[i] for i in range(len(args.tasks))]

    #instantialize segmentor
    if args.modelType == 'segmentor':
        segmentor = net.segmentor(args, inChans_list, args.base_outChans, num_class_list)
        segmentor.apply(train.weights_init)
        if args.resume_ckp != '':
            segmentor.load_state_dict(checkpoint['segmentor_state_dict'])
            train.train_segmentor(args, task_archive, segmentor, checkpoint)
        elif args.ckp != '':
            ckp_dir = args.base_dir + 'models/{}/'.format(args.ckp)
            checkpoint = torch.load(ckp_dir + 'epoch_{}.pth.tar'.format(args.ckp_epoch))
            #segmentor_old = segmentor
            #segmentor_old.load_state_dict(checkpoint['segmentor_state_dict'])
            segmentor_old = checkpoint['segmentor']
            segmentor = transfer_weights(segmentor, segmentor_old)
            train.train_segmentor(args, task_archive, segmentor)
        else:
            train.train_segmentor(args, task_archive, segmentor)

    #instantialize segan
    elif args.modelType == 'segan':
        segmentor = net.segmentor(args, inChans_list, args.base_outChans, num_class_list)
        segmentor.apply(train.weights_init)
        critic = net.critic(args, inChans_numClass, args.base_outChans, num_class_list)
        critic.apply(train.weights_init)
        if args.resume_ckp != '':
            segmentor.load_state_dict(checkpoint['segmentor_state_dict'])
            critic.load_state_dict(checkpoint_critic['critic_state_dict'])
            train.train_segan(args, task_archive, segmentor, critic, checkpoint, checkpoint_critic)
        elif args.ckp != '':
            ckp_dir = args.base_dir + 'models/{}/'.format(args.ckp)
            checkpoint = torch.load(ckp_dir + 'epoch_{}.pth.tar'.format(args.ckp_epoch))
            #segmentor_old = segmentor
            #segmentor_old.load_state_dict(checkpoint['segmentor_state_dict'])
            segmentor_old = checkpoint['segmentor']
            segmentor = transfer_weights(segmentor, segmentor_old)
            checkpoint_critic = torch.load(ckp_dir + 'critic_epoch_{}.pth.tar'.format(args.ckp_epoch))
            #critic_old = critic
            #critic_old.load_state_dict(checkpoint_critic['critic_state_dict'])
            critic_old = checkpoint_critic['critic']
            critic = transfer_weights(critic, critic_old)
            train.train_segan(args, task_archive, segmentor, critic)
        else:
            train.train_segan(args, task_archive, segmentor, critic)

elif args.mode == 'eval':

    for epoch in args.eval_epochs:
    
        #set checkpoint and args
        base_dir = args.base_dir
        model_to_eval = args.model_to_eval
        epoch_to_eval = epoch
        task_idxs = args.task_idxs
        ckp_dir = args.base_dir + 'models/{}/'.format(args.model_to_eval)
        checkpoint = torch.load(ckp_dir + 'epoch_{}.pth.tar'.format(epoch))
        args = checkpoint['args']
        args.base_dir = base_dir
        args.model_to_eval = model_to_eval
        args.epoch_to_eval = epoch_to_eval
        args.task_idxs = task_idxs

        #set tasks when only one task
        if type(args.tasks) is str:
            args.tasks = [args.tasks]

        #check if cuda is available
        if torch.cuda.is_available():
            print('CUDA is available')
            args.cuda = True
        else:
            args.cuda = False

        #make dirs and results ready for curr model
        results_dir = args.base_dir + 'results_eval/{}/'.format(args.model_to_eval)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        #open task archive
        with open(args.base_dir + '/dataset/fold_splits.json', mode='r') as f:
            task_archive = json.load(f)

        #open dataset info
        dataset_info = defaultdict(dict)
        for task in args.tasks:
            print(task)
            with open(args.base_dir + 'dataset/{}/dataset.json'.format(task), mode='r') as f:
                info_dict = json.load(f)
                try:
                    if task in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                        keys = list(info_dict['labels'].keys())
                        for key in keys:
                            if info_dict['labels'][key] in ['cancer', 'tumour']:
                                info_dict['labels'].pop(key)
                        print(info_dict)
                except:
                    print('HEY', task)
                dataset_info[task] = info_dict

        #set inChans_list and num_class_list
        inChans_list = [len(dataset_info[task]['modality']) for task in args.tasks]
        num_class_list = [len(dataset_info[task]['labels']) for task in args.tasks]

        #instantialize segmentor
        segmentor = net.segmentor(args, inChans_list, args.base_outChans, num_class_list)
        segmentor.apply(train.weights_init) #DOES THIS MAKE A DIFFERENCE?
        segmentor.load_state_dict(checkpoint['segmentor_state_dict'])

        #evaluate
        start = time.time()
        eval.evaluate(args, task_archive, dataset_info, segmentor, epoch)
        end = time.time()
        print('EVAL TIME', end-start)

elif args.mode == 'plot':
    
    #set results when only one result
    if type(args.results) is str:
        args.results = [args.results]

    for result in args.results:
        plot.plot_results(args.base_dir + 'results/{}/{}.txt'.format(args.model_to_plot, result))


elif args.mode == 'preproc':

    #set domain when only one domain
    if type(args.domains) is str:
        args.domains = [args.domains]

    for domain in args.domains:
        ids = [id for id in os.listdir(args.base_dir + 'dataset/{}/imagesTr'.format(domain)) if id[0] != '.']
        print(ids)
        print(len(ids))
        #open dataset info
        dataset_info = defaultdict(dict)
        with open(args.base_dir + 'dataset/{}/dataset.json'.format(domain), mode='r') as f:
            dataset_info[domain] = json.load(f)
        num_class = len(dataset_info[domain]['labels'])
        for id in ids:
            if id not in done:
                print(id)
                img, lab, orig_stats = data.preproc(args, id[:-7], num_class)
                print(img.shape, lab.shape)
                img_dir = args.base_dir + 'data2/{}/imagesTr'.format(domain)
                lab_dir = args.base_dir + 'data2/{}/labelsTr'.format(domain)
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                if not os.path.exists(lab_dir):
                    os.makedirs(lab_dir)
                np.save(args.base_dir + 'data2/{}/imagesTr/{}'.format(domain, id[:-7]), img, allow_pickle=False)
                np.save(args.base_dir + 'data2/{}/labelsTr/{}'.format(domain, id[:-7]), lab, allow_pickle=False)