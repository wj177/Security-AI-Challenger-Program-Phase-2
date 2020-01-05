import pandas as pd
import os
import numpy as np
import shutil
import torch
import cv2
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image


# csv_path = '/home/ly/1project/安全AI第二期/data/XPZ/18号/4full/tar_start.csv'
# csv_path = '/home/ly/1project/安全AI第二期/data/tar_start.csv'
# csv_path = '/home/ly/1project/安全AI第二期/data/XPZ/22号/单个最好的/tar_start.csv'
csv_path = 'path to csv文件'#该文件是使用target_atk_loop得到的csv文件
dev_df = pd.read_csv(csv_path)
img_root_dir = 'path to 1216张图像的地址' #需要攻击的1216张图像的地址
mask_root_dir = 'path to imagenet数据集的地址' #imagenet数据集的地址
save_img_path = 'path to 保存的图像的地址' 

def my_trans(mask_path_i,scope_min,scope,size):
    mask_i = cv2.imread(mask_path_i)
    mask_YCB = cv2.cvtColor(mask_i, cv2.COLOR_BGR2YCrCb)
    Y, C, B = cv2.split(mask_YCB)
    calsh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    # Y = cv2.equalizeHist(Y)
    Y = calsh.apply(Y)
    channels = cv2.merge((Y, C, B))
    mask_i = Image.fromarray(cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)).convert('RGB')

    img_0 = mask_i
    new_img = Image.new(mode='RGB', size=(299 * 2, 299 * 2))
    new_img.paste(img_0, (0, 0))
    new_img.paste(img_0, (299, 0))
    new_img.paste(img_0, (0, 299))
    new_img.paste(img_0, (299, 299))
    new_img = new_img.resize((299, 299))

    mask_itc = torch_transforms(new_img)

    mask_itc = torch.clamp(mask_itc, scope_min, scope + scope_min) - scope_min
    input_mask = ori_img_ibu + (mask_itc * (size / (scope * 255)) - (0.5 * size / 255))
    eta = input_mask - ori_img_i
    eta = torch.clamp(eta, -32 / 255, 32 / 255)
    input_mask = ori_img_i + eta
    input_mask = torch.clamp(input_mask, 0., 1.)

    npimg = (input_mask.detach().numpy() * 255).astype(np.uint8)
    npimg = np.transpose(npimg, (1, 2, 0))
    save_img_mask = Image.fromarray(npimg)
    if not os.path.exists(save_img_path): os.makedirs(save_img_path)
    ss = os.path.join(save_img_path, ori_img_name)
    save_img_mask.save(ss, quality=100)

def my_trans_4(mask_path_i,scope_min,scope,size):
    mask_i = cv2.imread(mask_path_i)
    mask_YCB = cv2.cvtColor(mask_i, cv2.COLOR_BGR2YCrCb)
    Y, C, B = cv2.split(mask_YCB)
    calsh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    # Y = cv2.equalizeHist(Y)
    Y = calsh.apply(Y)
    channels = cv2.merge((Y, C, B))
    mask_i = Image.fromarray(cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)).convert('RGB')

    img_0 = mask_i
    new_img = Image.new(mode='RGB', size=(299 * 2, 299 * 2))
    new_img.paste(img_0, (0, 0))
    new_img.paste(img_0, (299, 0))
    new_img.paste(img_0, (0, 299))
    new_img.paste(img_0, (299, 299))
    new_img = new_img.resize((299, 299))

    mask_itc = torch_transforms(new_img)
    mask_itc = torch.clamp(mask_itc, scope_min, scope + scope_min) - scope_min
    input_mask = ori_img_ibu + (mask_itc * (size / (scope * 255)) - (0.5 * size / 255))
    eta = input_mask - ori_img_i
    eta = torch.clamp(eta, -32 / 255, 32 / 255)
    input_mask = ori_img_i + eta
    input_mask = torch.clamp(input_mask, 0., 1.)

    npimg = (input_mask.detach().numpy() * 255).astype(np.uint8)
    npimg = np.transpose(npimg, (1, 2, 0))
    save_img_mask = Image.fromarray(npimg)
    if not os.path.exists(save_img_path): os.makedirs(save_img_path)
    ss = os.path.join(save_img_path, ori_img_name)
    save_img_mask.save(ss, quality=100)

def my_trans1(mask_path_i,scope_min,scope,size):
    mask_i = cv2.imread(mask_path_i)
    mask_YCB = cv2.cvtColor(mask_i, cv2.COLOR_BGR2YCrCb)
    Y, C, B = cv2.split(mask_YCB)
    calsh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    # Y = cv2.equalizeHist(Y)
    Y = calsh.apply(Y)
    channels = cv2.merge((Y, C, B))
    mask_i = Image.fromarray(cv2.cvtColor(channels, cv2.COLOR_YCrCb2RGB)).convert('RGB')

    img_0 = mask_i
    mask_itc = torch_transforms(img_0)

    mask_itc = torch.clamp(mask_itc, scope_min, scope + scope_min) - scope_min
    input_mask = ori_img_ibu + (mask_itc * (size / (scope * 255)) - (0.5 * size / 255))
    eta = input_mask - ori_img_i
    eta = torch.clamp(eta, -32 / 255, 32 / 255)
    input_mask = ori_img_i + eta
    input_mask = torch.clamp(input_mask, 0., 1.)

    npimg = (input_mask.detach().numpy() * 255).astype(np.uint8)
    npimg = np.transpose(npimg, (1, 2, 0))
    save_img_mask = Image.fromarray(npimg)
    if not os.path.exists(save_img_path): os.makedirs(save_img_path)
    ss = os.path.join(save_img_path, ori_img_name)
    save_img_mask.save(ss, quality=100)

torch_transforms = transforms.Compose([transforms.ToTensor()])
for i in range(len(dev_df)):
    ori_img_name = dev_df.iloc[i]['ImageId']
    Ttogap = dev_df.iloc[i]['Ttogap']
    Uuogap = dev_df.iloc[i]['Uuogap']
    Ttrans = dev_df.iloc[i]['Ttrans']
    Utrans = dev_df.iloc[i]['Utrans']
    TmaskId = dev_df.iloc[i]['TmaskId']
    UmaskId = dev_df.iloc[i]['UmaskId']
    TmaskClass = str(dev_df.iloc[i]['TmaskClass'])
    UmaskClass = str(dev_df.iloc[i]['UmaskClass'])
    #得到原始图像的名字
    ori_1216_img_path_i =os.path.join(img_root_dir,ori_img_name)
    ori_img_i = Image.open(ori_1216_img_path_i)
    img_gs = cv2.imread(ori_1216_img_path_i)
    img_gs = cv2.GaussianBlur(img_gs,(21,21),0.0)
    img_gs = Image.fromarray(cv2.cvtColor(img_gs,cv2.COLOR_BGR2RGB))
    ori_img_ibu = torch_transforms(img_gs)
    ori_img_i = torch_transforms(ori_img_i)
    #要判断是使用T还是使用U
    if Ttogap >2.0:
        mask_path_i = os.path.join(mask_root_dir,TmaskClass,TmaskId)
        scope_min = int(Ttrans.split('_')[0]) / 255
        scope = int(Ttrans.split('_')[1]) / 255
        size = int(Ttrans.split('_')[2])
        my_trans1(mask_path_i,scope_min,scope,size)
    else:
        mask_path_i = os.path.join(mask_root_dir,UmaskClass,UmaskId)
        scope_min = int(Utrans.split('_')[0]) / 255
        scope = int(Utrans.split('_')[1]) / 255
        size = int(Utrans.split('_')[2])
        my_trans1(mask_path_i,scope_min,scope,size)

