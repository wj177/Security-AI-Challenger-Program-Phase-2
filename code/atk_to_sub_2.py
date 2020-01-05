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

csv_path = 'path to csv文件' #该文件是使用的target_atk_loopv2得到的csv文件
dev_df = pd.read_csv(csv_path)
img_root_dir = 'path to 1216张图像的地址' #需要攻击的1216张图像的地址
mask_root_dir = 'path to imagenet数据集的地址' #imagenet数据集的地址
save_img_path = 'path to 保存的图像的地址' 

def my_trans(mask_path_i,scope_min,scope,size):
    mask_i = cv2.imread(mask_path_i)
    mask_YCB = cv2.cvtColor(mask_i, cv2.COLOR_BGR2YCrCb)
    Y, C, B = cv2.split(mask_YCB)
    calsh = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
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


def my_equalhist(img_o_0):
    #输入图片的地址，返回经过直方图均衡化的图
    img_YCB_0 = cv2.cvtColor(np.asarray(img_o_0),cv2.COLOR_RGB2YCrCb)
    Y_0, C_0, B_0 = cv2.split(img_YCB_0)
    equalclash =cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
    Y_0 = equalclash.apply(Y_0)
    channels_0 = cv2.merge((Y_0, C_0, B_0))
    img_0 = Image.fromarray(cv2.cvtColor(channels_0, cv2.COLOR_YCrCb2RGB)).convert('RGB')
    return img_0

def my_trans_4(mask_path_1,mask_path_2,mask_path_3,mask_path_4,scope_min,scope,size):
    mask_1 = Image.open(mask_path_1)
    mask_2 = Image.open(mask_path_2)
    mask_3 = Image.open(mask_path_3)
    mask_4 = Image.open(mask_path_4)

    new_img = Image.new(mode='RGB', size=(299 * 2, 299 * 2))
    new_img.paste(mask_1, (0, 0))
    new_img.paste(mask_2, (299, 0))
    new_img.paste(mask_3, (0, 299))
    new_img.paste(mask_4, (299, 299))
    new_img = new_img.resize((299, 299))
    new_img = my_equalhist(new_img)
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

    TmaskId_0 = dev_df.iloc[i]['TmaskId_0']
    TmaskId_1 = dev_df.iloc[i]['TmaskId_1']
    TmaskId_2 = dev_df.iloc[i]['TmaskId_2']
    TmaskId_3 = dev_df.iloc[i]['TmaskId_3']
    TmaskId_4 = dev_df.iloc[i]['TmaskId_4']

    UmaskId_0 = dev_df.iloc[i]['UmaskId_0']
    UmaskId_1 = dev_df.iloc[i]['UmaskId_1']
    UmaskId_2 = dev_df.iloc[i]['UmaskId_2']
    UmaskId_3 = dev_df.iloc[i]['UmaskId_3']
    UmaskId_4 = dev_df.iloc[i]['UmaskId_4']

    TmaskClass_0 = str(dev_df.iloc[i]['TmaskClass_0'])
    TmaskClass_1 = str(dev_df.iloc[i]['TmaskClass_1'])
    TmaskClass_2 = str(dev_df.iloc[i]['TmaskClass_2'])
    TmaskClass_3 = str(dev_df.iloc[i]['TmaskClass_3'])
    TmaskClass_4 = str(dev_df.iloc[i]['TmaskClass_4'])

    UmaskClass_0 = str(dev_df.iloc[i]['UmaskClass_0'])
    UmaskClass_1 = str(dev_df.iloc[i]['UmaskClass_1'])
    UmaskClass_2 = str(dev_df.iloc[i]['UmaskClass_2'])
    UmaskClass_3 = str(dev_df.iloc[i]['UmaskClass_3'])
    UmaskClass_4 = str(dev_df.iloc[i]['UmaskClass_4'])

    #得到原始图像的名字
    ori_1216_img_path_i =os.path.join(img_root_dir,ori_img_name)
    ori_img_i = Image.open(ori_1216_img_path_i)
    img_gs = cv2.imread(ori_1216_img_path_i)
    img_gs = cv2.GaussianBlur(img_gs,(21,21),0.0)
    img_gs = Image.fromarray(cv2.cvtColor(img_gs,cv2.COLOR_BGR2RGB))
    ori_img_ibu = torch_transforms(img_gs)
    ori_img_i = torch_transforms(ori_img_i)

    #要判断是使用T还是使用U
    if Ttogap >4.0:
        #这里会使用4个mask,所以要更新4个
        mask_path_1 = os.path.join(mask_root_dir,TmaskClass_1,TmaskId_1)
        mask_path_2 = os.path.join(mask_root_dir,TmaskClass_2,TmaskId_2)
        mask_path_3 = os.path.join(mask_root_dir,TmaskClass_3,TmaskId_3)
        mask_path_4 = os.path.join(mask_root_dir,TmaskClass_4,TmaskId_4)

        scope_min = int(Ttrans.split('_')[0]) / 255
        scope = int(Ttrans.split('_')[1]) / 255
        size = int(Ttrans.split('_')[2])
        my_trans_4(mask_path_1,mask_path_2,mask_path_3,mask_path_4,scope_min,scope,size)



