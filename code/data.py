import torch
import random
import json

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from collections import defaultdict

def reorint(img, start_axcode, end_axcode):
    start_orint = nib.orientations.axcodes2ornt(start_axcode)
    end_orint = nib.orientations.axcodes2ornt(end_axcode)
    ornt_transf = nib.orientations.ornt_transform(start_orint, end_orint)
    data_reoriented = nib.orientations.apply_orientation(
        img, ornt_transf)
    return data_reoriented

def preproc(args, id, num_class):
    """
    Accustomed from https://github.com/huangmozhilv/u2net_torch
    """

    task = id.split('_')[0]
    task_taskId = {'BRATS': 'Task01_BrainTumour', 'la': 'Task02_Heart', 'liver': 'Task03_Liver', 'hippocampus': 'Task04_Hippocampus', 'prostate': 'Task05_Prostate', 'lung': 'Task06_Lung', 'pancreas': 'Task07_Pancreas', 'hepaticvessel': 'Task08_HepaticVessel', 'spleen': 'Task09_Spleen', 'colon': 'Task10_Colon'}

    sitk_image = sitk.ReadImage(args.base_dir + 'dataset/{}/imagesTr/{}.nii.gz'.format(task_taskId[task], id))
    #orig_volume = sitk.GetArrayFromImage(sitk_image) # mod, z, y, x
    
    img = nib.load(args.base_dir + 'dataset/{}/imagesTr/{}.nii.gz'.format(task_taskId[task], id))
    if nib.aff2axcodes(img.affine) != ('R','A','S'):
        print(nib.aff2axcodes(img.affine))
        img = reorint(img.get_fdata(), nib.aff2axcodes(img.affine), ('R','A','S'))
        orig_volume = np.transpose(img)
    else:
        orig_volume = sitk.GetArrayFromImage(sitk_image) # mod, z, y, x

    if sitk_image.GetDimension() == 3:
        mod_num = 1
    elif sitk_image.GetDimension() == 4:
        mod_num = sitk_image.GetSize()[3]

    if mod_num == 1:
        orig_volume = orig_volume[np.newaxis,...]

    #background_vals = []
    min_vals = []
    max_vals = []
    mean_vals = []
    std_vals = []
    orig_stats = (min_vals, max_vals, mean_vals, std_vals)

    volume_list = []
    for mod_idx in range(mod_num):
        volume = orig_volume[mod_idx,...]
        #background_val = np.bincount(volume).argmax()
        #background_vals.append(background_val)
        min_val = np.amin(volume)
        min_vals.append(min_val)
        max_val = np.amax(volume)
        max_vals.append(max_val)
        mean_val = np.mean(volume)
        mean_vals.append(mean_val)
        std_val = np.std(volume)
        std_vals.append(std_val)
        original_shape = volume.shape
        # 155 244 244
        if mod_idx == 0:
            # contain whole tumor
            margin = 5 # small padding value
            bbmin, bbmax = get_none_zero_region(volume, margin) 
        volume = crop_ND_volume_with_bounding_box(volume, bbmin, bbmax)
        pixel_spacings = {'Task01_BrainTumour': [1.0,1.0,1.0], 'Task02_Heart':[1.37,1.25,1.25], 'Task03_Liver':[1.0,0.77,0.77], 'Task04_Hippocampus':[1.0,1.0,1.0], 'Task05_Prostate':[3.6,0.625,0.625], 'Task06_Lung': [1.245,0.782,0.782], 'Task07_Pancreas':[2.5,0.8,0.8], 'Task08_HepaticVessel': [5.0,0.799,0.799], 'Task09_Spleen':[5.0,0.794,0.794], 'Task10_Colon': [5.0,0.781,0.781]}
        volume = resample2fixedSpacing(args, id, volume, pixel_spacings[task_taskId[task]], interpolate_method=sitk.sitkBSpline) # sitk.sitkLinear # cautions! remember to inversely resample the label map to cropped volume size.
        # intensity clipping
        volume[volume<-1024] = -1024 # works for CT

        if mod_idx == 0:
            weight = np.asarray(volume > 0, np.float32)

        p_l,p_u = np.percentile(volume, (2.0, 98.0))
        volume = np.clip(volume, p_l,p_u)
        volume = itensity_normalize_one_volume(volume)
        
        volume_list.append(volume)

    #label = sitk.GetArrayFromImage(sitk.ReadImage(args.base_dir + 'dataset/{}/labelsTr/{}.nii.gz'.format(task_taskId[task], id))) # mod, d, h, w
    lab = nib.load(args.base_dir + 'dataset/{}/labelsTr/{}.nii.gz'.format(task_taskId[task], id))
    if nib.aff2axcodes(lab.affine) != ('R','A','S'):
        print(nib.aff2axcodes(lab.affine))
        lab = reorint(lab.get_fdata(), nib.aff2axcodes(lab.affine), ('R','A','S'))
        label = np.transpose(lab)
    else:
        label = sitk.GetArrayFromImage(sitk.ReadImage(args.base_dir + 'dataset/{}/labelsTr/{}.nii.gz'.format(task_taskId[task], id))) # mod, d, h, w
    label[label > num_class-1] = 0 # Task04_Hippocampus 003 and 243 have one wrong gt pixel assigned 254. Here arbitrarily set to 0 （background).
    label = crop_ND_volume_with_bounding_box(label, bbmin, bbmax) 
    # resampling.
    label = resample2fixedSpacing(args, id, label, pixel_spacings[task_taskId[task]], interpolate_method=sitk.sitkNearestNeighbor)

    volumes = np.asarray(volume_list)

    return volumes, label, orig_stats

def get_none_zero_region(im, margin):
    """
    Author https://github.com/huangmozhilv/u2net_torch
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    if len(indxes[0]):
        idx_min = []
        idx_max = []
        # logger.info('indxes:{}'.format(indxes))
        for i in range(len(input_shape)):
            idx_min.append(indxes[i].min())
            idx_max.append(indxes[i].max())

        for i in range(len(input_shape)):
            idx_min[i] = max(idx_min[i] - margin[i], 0)
            idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
        return idx_min, idx_max
    else:
        # some tasks, e.g. Task03_Liver, some cases have no tumor/cancer, so no small_center_bbox to cal
        return

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    Author https://github.com/huangmozhilv/u2net_torch
    crop/extract a subregion form an nd image.
    """
    output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                            range(min_idx[1], max_idx[1] + 1),
                            range(min_idx[2], max_idx[2] + 1))]
    return output

def resample2fixedSpacing(args, id, volume, newSpacing, interpolate_method=sitk.sitkBSpline): 
    # sitk.sitkLinear
    '''
    Author https://github.com/huangmozhilv/u2net_torch
    also works for 2-D?
    Resample dat(i.e. one 3-D sitk image/GT) to destine resolution. but keep the origin, direction be the same.
    Volume: 3D numpy array, z, y, x.
    oldSpacing: z,y,x
    newSpacing: z,y,x
    refer_file_path: source to get origin, direction, and oldSpacing. Here we use the image_file path.
    ''' 
    # in the project, oldSpacing, origin, direction will be extracted from gt_file as the refer_file_path
    task = id.split('_')[0]
    task_taskId = {'BRATS': 'Task01_BrainTumour', 'la': 'Task02_Heart', 'liver': 'Task03_Liver', 'hippocampus': 'Task04_Hippocampus', 'prostate': 'Task05_Prostate', 'lung': 'Task06_Lung', 'pancreas': 'Task07_Pancreas', 'hepaticvessel': 'Task08_HepaticVessel', 'spleen': 'Task09_Spleen', 'colon': 'Task10_Colon'}
    sitk_refer = sitk.ReadImage(args.base_dir + 'dataset/{}/imagesTr/{}.nii.gz'.format(task_taskId[task], id))
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    origin = sitk_refer.GetOrigin()
    oldSpacing =  sitk_refer.GetSpacing()
    direction = sitk_refer.GetDirection()
    

    # prepare oldSize, oldSpacing, newSpacing, newSize in order of [x,y,z]
    oldSize = np.asarray(volume.shape, dtype=float)[::-1]
    oldSpacing = np.asarray([round(i, 3) for i in oldSpacing], dtype=float)
    newSpacing = np.asarray([round(i, 3) for i in newSpacing], dtype=float)[::-1]
    # compute new size, assuming same volume of tissue (not number of total pixels) before and after resampled 
    newSize = np.asarray(oldSize * oldSpacing/newSpacing, dtype=int)

    # create sitk_old from array and set appropriate meta-data
    sitk_old = sitk.GetImageFromArray(volume)
    

    sitk_old.SetOrigin(origin)
    sitk_old.SetSpacing(oldSpacing)
    sitk_old.SetDirection(direction)
    sitk_new = sitk.Resample(sitk_old, newSize.tolist(), sitk.Transform(), interpolate_method, origin, newSpacing, direction)

    newVolume = sitk.GetArrayFromImage(sitk_new)

    
    return newVolume

def itensity_normalize_one_volume(volume):
    """
    Author https://github.com/huangmozhilv/u2net_torch
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    pixels = volume[volume > 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/(std + 1e-20)
    # random normal too slow
    #out_random = np.random.normal(0, 1, size = volume.shape)
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]
    return out

def pad_data(img, lab, size): #MAKE FOR LOOP
    #get pad value
    if img.shape[0] < size[0]:
        if ((size[0]-img.shape[0])) % 2 == 0:
            x_pad = (int((size[0]-img.shape[0])/2), int((size[0]-img.shape[0])/2))
        else:
            x_pad = (int((size[0]-img.shape[0])/2), int((size[0]-img.shape[0])/2)+1)
    else:
        x_pad = (0,0)
    if img.shape[1] < size[1]:
        if ((size[1]-img.shape[1])) % 2 == 0:
            y_pad = (int((size[1]-img.shape[1])/2), int((size[1]-img.shape[1])/2))
        else:
            y_pad = (int((size[1]-img.shape[1])/2), int((size[1]-img.shape[1])/2)+1)
    else:
        y_pad = (0,0)
    if img.shape[2] < size[2]:
        if ((size[2]-img.shape[2])) % 2 == 0:
            z_pad = (int((size[2]-img.shape[2])/2), int((size[2]-img.shape[2])/2))
        else:
            z_pad = (int((size[2]-img.shape[2])/2), int((size[2]-img.shape[2])/2)+1)
    else:
        z_pad = (0,0)
    #pad img and lab
    if len(img.shape) == 4:
        img = np.pad(img, pad_width=[x_pad, y_pad, z_pad, (0,0)]) #default mode constant pad with value 0
    else:
        img = np.pad(img, pad_width=[x_pad, y_pad, z_pad])
    lab = np.pad(lab, pad_width=[x_pad, y_pad, z_pad])
    return img, lab

def randCrop_data(img, lab, size):
    #num_class = len(np.unique(lab))
    done = False
    #get start value
    x_start = random.randint(0, img.shape[0]-size[0])
    y_start = random.randint(0, img.shape[1]-size[1])
    z_start = random.randint(0, img.shape[2]-size[2])
    if len(img.shape) == 4:
        img = img[x_start:x_start+size[0], y_start:y_start+size[1], z_start:z_start+size[2], :]
    else:
        img = img[x_start:x_start+size[0], y_start:y_start+size[1], z_start:z_start+size[2]]
    lab = lab[x_start:x_start+size[0], y_start:y_start+size[1], z_start:z_start+size[2]]
    #done = True
    return img, lab

def trans_data(img, lab):
    #from numpy to torch
    img = torch.from_numpy(img)
    lab = torch.from_numpy(lab)
    #reshape from (H,W,D,C) to (C,H,W,D) + add batch and channel
    if len(img.shape) == 4:
        img = img.permute(-1,0,1,2).unsqueeze(0)
        lab = lab.unsqueeze(0) #.unsqueeze(0) <- no batch added to lab
    else:
        img = img.unsqueeze(0).unsqueeze(0)
        lab = lab.unsqueeze(0) #.unsqueeze(0)
    return img.float(), lab.float()

def get_patch(args, id):
    task = id.split('_')[0]
    task_taskId = {'BRATS': 'Task01_BrainTumour', 'la': 'Task02_Heart', 'liver': 'Task03_Liver', 'hippocampus': 'Task04_Hippocampus', 'prostate': 'Task05_Prostate', 'lung': 'Task06_Lung', 'pancreas': 'Task07_Pancreas', 'hepaticvessel': 'Task08_HepaticVessel', 'spleen': 'Task09_Spleen', 'colon': 'Task10_Colon'}
    img = torch.load(args.base_dir + 'patch_data/{}/imagesTr/{}.pt'.format(task_taskId[task], id))
    lab = torch.load(args.base_dir + 'patch_data/{}/labelsTr/{}.pt'.format(task_taskId[task], id))
    return img, lab

#get data function
def get_nifty(args, id):
    #set task and task_taskID
    task = id.split('_')[0]
    task_taskId = {'BRATS': 'Task01_BrainTumour', 'la': 'Task02_Heart', 'liver': 'Task03_Liver', 'hippocampus': 'Task04_Hippocampus', 'prostate': 'Task05_Prostate', 'lung': 'Task06_Lung', 'pancreas': 'Task07_Pancreas', 'hepaticvessel': 'Task08_HepaticVessel', 'spleen': 'Task09_Spleen', 'colon': 'Task10_Colon'}
    #load img and lab
    img = nib.load(args.base_dir + 'dataset/{}/imagesTr/{}.nii.gz'.format(task_taskId[task], id))
    lab = nib.load(args.base_dir + 'dataset/{}/labelsTr/{}.nii.gz'.format(task_taskId[task], id))
    #from nib to numpy
    img = img.get_fdata()
    lab = lab.get_fdata()
    return img, lab

def get_data(args, id):
    task = id.split('_')[0]
    task_taskId = {'BRATS': 'Task01_BrainTumour', 'la': 'Task02_Heart', 'liver': 'Task03_Liver', 'hippocampus': 'Task04_Hippocampus', 'prostate': 'Task05_Prostate', 'lung': 'Task06_Lung', 'pancreas': 'Task07_Pancreas', 'hepaticvessel': 'Task08_HepaticVessel', 'spleen': 'Task09_Spleen', 'colon': 'Task10_Colon'}
    img = np.load(args.base_dir + 'data2/{}/imagesTr/{}.npy'.format(task_taskId[task], id))
    lab = np.load(args.base_dir + 'data2/{}/labelsTr/{}.npy'.format(task_taskId[task], id))
    return img, lab

def save_results(args, result_name, epoch, step_id, task, cl, value, type='results'):
    if type == 'results':
        with open(args.base_dir + '{}/{}/{}.txt'.format(type, args.modelName, result_name), 'a') as f:
            f.write('epoch: {} step/id: {} task: {} class: {} value: {}\n'.format(epoch, step_id, task, cl, value))
    elif type == 'results_eval':
        with open(args.base_dir + '{}/{}/{}.txt'.format(type, args.model_to_eval, result_name), 'a') as f:
            f.write('epoch: {} step/id: {} task: {} class: {} value: {}\n'.format(epoch, step_id, task, cl, value))