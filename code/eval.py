import torch

from collections import defaultdict
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, adjusted_rand_score, adjusted_mutual_info_score, cohen_kappa_score, roc_auc_score
from scipy.spatial.distance import directed_hausdorff

import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
import numpy as np

import data
import loss
import json
import time

def evaluate(args, task_archive, dataset_info, segmentor, epoch):

    #cuda model
    if args.cuda:
        segmentor.cuda()
        print('segmentor on cuda', next(segmentor.parameters()).is_cuda)
    
    #set eval mode
    segmentor.eval()

    #set losses
    dice_loss = loss.MulticlassDiceLoss()

    for task_idx in args.task_idxs:

        #get curr task
        task_idx = int(task_idx)
        print(args.tasks)
        print(task_idx)
        task = args.tasks[task_idx]

        #open dataset info
        dataset_info = defaultdict(dict)
        for t in args.tasks:
            with open(args.base_dir + 'dataset/{}/dataset.json'.format(t), mode='r') as f:
                info_dict = json.load(f)
                try:
                    if t in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                        keys = list(info_dict['labels'].keys())
                        for key in keys:
                            if info_dict['labels'][key] in ['cancer', 'tumour']:
                                info_dict['labels'].pop(key)
                        print(info_dict)
                except:
                    print('HEY', t)
                dataset_info[t] = info_dict
        num_class = len(dataset_info[task]['labels'])

        #eval losses and metrics placeholders
        num_id = 0
        dloss_all = 0
        dcoef_all = 0
        dcoefs_all = [0 for i in range(num_class)]
        done_ids = []
        accuracy_all = 0
        precision_all = [0 for i in range(num_class)]
        recall_all = [0 for i in range(num_class)]
        f1_all = [0 for i in range(num_class)]
        support_all = [0 for i in range(num_class)]
        aucs_all = [0 for i in range(num_class)]
        macro_precision_all = 0
        macro_recall_all = 0
        macro_f1_all = 0
        w_precision_all = 0
        w_recall_all = 0
        w_f1_all = 0

        #for id in ids:
        for id in task_archive[task]['fold0']['val']:

            print(id)

            img, lab = data.get_data(args, id)

            #fuse cancer to organ
            try:
                if task in ['Task03_Liver', 'Task07_Pancreas'] and args.object_seg == True:
                    print('FUSE CANCER TO ORGAN', np.unique(lab))
                    lab[lab == 2] = 1
                    print('FUSE CANCER TO ORGAN', np.unique(lab))
            except:
                print('HEY', task)

            print(img.shape, lab.shape)

            #reshape img
            img = np.moveaxis(img, 0, -1)

            #get pad sizes
            pad_size = []
            for i in range(3):
                if args.input_size[i]-img.shape[i] >= 0:
                    pad_size.append(args.input_size[i])
                else:
                    curr_pad_size = args.input_size[i]
                    while curr_pad_size-img.shape[i] < 0:
                        curr_pad_size += args.input_size[i]/2
                    pad_size.append(int(curr_pad_size))

            #pad data
            img, lab = data.pad_data(img, lab, pad_size)

            #transform data to torch + batch and channel
            img, lab = data.trans_data(img, lab)
            print(img.shape, lab.shape)

            #cuda data
            if args.cuda:
                img = img.cuda()
                lab = lab

            #set num class, probs and nums
            probs = torch.zeros(1,num_class,img.shape[2],img.shape[3],img.shape[4])
            nums = torch.zeros(1,num_class,img.shape[2],img.shape[3],img.shape[4])

            #cuda props
            if args.cuda:
                probs = probs.cuda()
                nums = nums

            num = 0
            coords = []
            for x in range(0, img.shape[2]-int(args.input_size[0]/2), int(args.input_size[0]/2)):
                for y in range(0, img.shape[3]-int(args.input_size[1]/2), int(args.input_size[1]/2)):
                    for z in range(0, img.shape[4]-int(args.input_size[2]/2), int(args.input_size[2]/2)):
                        
                        print(x,y,z)
                        coords.append((x,y,z))

                        #get curr patch
                        curr_img = img[:, :, x:x+args.input_size[0], y:y+args.input_size[1], z:z+args.input_size[2]]
                        if x != num:
                            for coord in coords:
                                img[:, :, coord[0]:coord[0]+int(args.input_size[0]/2), coord[1]:coord[1]+int(args.input_size[1]/2), coord[2]:coord[2]+int(args.input_size[2]/2)] = 0
                            print(num)
                            print(coords)
                            num = x
                            coords = []

                        #get probs
                        probs[:, :, x:x+args.input_size[0], y:y+args.input_size[1], z:z+args.input_size[2]] += segmentor(args, task_idx, curr_img).detach()
                        nums[:, :, x:x+args.input_size[0], y:y+args.input_size[1], z:z+args.input_size[2]] += 1

            print(torch.unique(nums))

            #get softmax
            softmax = F.softmax(probs.cpu()/nums, dim=1)
            print(softmax.shape)

            #get argmax
            argmax = torch.argmax(softmax, dim=1)
            print(argmax.shape, torch.unique(argmax))

            #calculate losses and metrics
            #args.cuda = False
            dloss, dcoef, all_dcoefs = dice_loss(args, softmax, lab, num_class=num_class)
            print(dloss.item(), dcoef.item(), [dcoef.item() for dcoef in all_dcoefs])
            #f1 = f1_score(torch.flatten(lab), torch.flatten(argmax), average=None)
            #print(f1)
            class_report = classification_report(torch.flatten(lab), torch.flatten(argmax), output_dict=True)
            print(class_report)
            all_aucs = []
            for i in range(num_class):
                curr_lab = lab.clone()
                curr_lab[lab == i] = 1
                curr_lab[lab != i] = 0
                auc_score = roc_auc_score(torch.flatten(curr_lab), torch.flatten(softmax[:,i,:,:,:]))
                all_aucs.append(auc_score)
            print(all_aucs)

            #save losses and metrics
            #dice
            data.save_results(args, 'eval_dloss', args.epoch_to_eval, id, task, 'all', dloss.item(), type='results_eval')
            dloss_all += dloss.item()
            data.save_results(args, 'eval_dcoef', args.epoch_to_eval, id, task, 'all', dcoef.item(), type='results_eval')
            dcoef_all += dcoef.item()
            for i in range(len(all_dcoefs)):
                data.save_results(args, 'eval_dcoefs', args.epoch_to_eval, id, task, i, all_dcoefs[i].item(), type='results_eval')
                dcoefs_all[i] += all_dcoefs[i].item()
            #precision, recall, f1-score, support, accuracy
            for i in class_report.keys():
                if i == 'accuracy':
                    data.save_results(args, 'eval_accuracy', args.epoch_to_eval, id, task, i, class_report[i], type='results_eval')
                    accuracy_all += float(class_report[i])
                elif i == 'macro avg':
                    data.save_results(args, 'eval_precision', args.epoch_to_eval, id, task, 'all_macro', class_report[i]['precision'], type='results_eval')
                    macro_precision_all += float(class_report[i]['precision'])
                    data.save_results(args, 'eval_recall', args.epoch_to_eval, id, task, 'all_macro', class_report[i]['recall'], type='results_eval')
                    macro_recall_all += float(class_report[i]['recall'])
                    data.save_results(args, 'eval_f1', args.epoch_to_eval, id, task, 'all_macro', class_report[i]['f1-score'], type='results_eval')
                    macro_f1_all += float(class_report[i]['f1-score'])
                elif i == 'weighted avg':
                    data.save_results(args, 'eval_precision', args.epoch_to_eval, id, task, 'all_w', class_report[i]['precision'], type='results_eval')
                    w_precision_all += float(class_report[i]['precision'])
                    data.save_results(args, 'eval_recall', args.epoch_to_eval, id, task, 'all_w', class_report[i]['recall'], type='results_eval')
                    w_recall_all += float(class_report[i]['recall'])
                    data.save_results(args, 'eval_f1', args.epoch_to_eval, id, task, 'all_w', class_report[i]['f1-score'], type='results_eval')
                    w_f1_all += float(class_report[i]['f1-score'])
                else:
                    data.save_results(args, 'eval_precision', args.epoch_to_eval, id, task, i, class_report[i]['precision'], type='results_eval')
                    precision_all[int(float(i))] += float(class_report[i]['precision'])
                    data.save_results(args, 'eval_recall', args.epoch_to_eval, id, task, i, class_report[i]['recall'], type='results_eval')
                    recall_all[int(float(i))] += float(class_report[i]['recall'])
                    data.save_results(args, 'eval_f1', args.epoch_to_eval, id, task, i, class_report[i]['f1-score'], type='results_eval')
                    f1_all[int(float(i))] += float(class_report[i]['f1-score'])
                    data.save_results(args, 'eval_support', args.epoch_to_eval, id, task, i, class_report[i]['support'], type='results_eval')
                    support_all[int(float(i))] += float(class_report[i]['support'])
            for i in range(len(all_aucs)):
                data.save_results(args, 'eval_aucs', args.epoch_to_eval, id, task, i, all_aucs[i], type='results_eval')
                aucs_all[i] += float(all_aucs[i])

            #increment epoch losses and metrics
            num_id += 1
            
            #args.cuda = True

        #save mean losses and metrics
        #dice
        dloss_all /= num_id
        data.save_results(args, 'mean_eval_dloss', args.epoch_to_eval, None, task, 'all', dloss_all, type='results_eval')
        dcoef_all /= num_id
        data.save_results(args, 'mean_eval_dcoef', args.epoch_to_eval, None, task, 'all', dcoef_all, type='results_eval')
        for i in range(len(dcoefs_all)):
            dcoefs_all[i] /= num_id
            data.save_results(args, 'mean_eval_dcoefs', args.epoch_to_eval, None, task, i, dcoefs_all[i], type='results_eval')
        #precision, recall, f1-score, support, accuracy
        accuracy_all /= num_id
        data.save_results(args, 'mean_eval_accuracy', args.epoch_to_eval, None, task, 'all', accuracy_all, type='results_eval')
        macro_precision_all /= num_id
        data.save_results(args, 'mean_eval_precision', args.epoch_to_eval, None, task, 'all_macro', macro_precision_all, type='results_eval')
        macro_recall_all /= num_id
        data.save_results(args, 'mean_eval_recall', args.epoch_to_eval, None, task, 'all_macro', macro_recall_all, type='results_eval')
        macro_f1_all /= num_id
        data.save_results(args, 'mean_eval_f1', args.epoch_to_eval, None, task, 'all_macro', macro_f1_all, type='results_eval')
        w_precision_all /= num_id
        data.save_results(args, 'mean_eval_precision', args.epoch_to_eval, None, task, 'all_w', w_precision_all, type='results_eval')
        w_recall_all /= num_id
        data.save_results(args, 'mean_eval_recall', args.epoch_to_eval, None, task, 'all_w', w_recall_all, type='results_eval')
        w_f1_all /= num_id
        data.save_results(args, 'mean_eval_f1', args.epoch_to_eval, None, task, 'all_w', w_f1_all, type='results_eval')
        for i in range(len(precision_all)):
            precision_all[i] /= num_id
            data.save_results(args, 'mean_eval_precision', args.epoch_to_eval, None, task, i, precision_all[i], type='results_eval')
            recall_all[i] /= num_id
            data.save_results(args, 'mean_eval_recall', args.epoch_to_eval, None, task, i, recall_all[i], type='results_eval')
            f1_all[i] /= num_id
            data.save_results(args, 'mean_eval_f1', args.epoch_to_eval, None, task, i, f1_all[i], type='results_eval')
            support_all[i] /= num_id
            data.save_results(args, 'mean_eval_support', args.epoch_to_eval, None, task, i, support_all[i], type='results_eval')
        for i in range(len(aucs_all)):
            aucs_all[i] /= num_id
            data.save_results(args, 'mean_eval_aucs', args.epoch_to_eval, None, task, i, aucs_all[i], type='results_eval')
        