
import torch
import argparse
from tqdm import tqdm
import numpy as np
import  torchvision.models as models
import efficientnet_pytorch
from PIL import Image
import os
import random
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
import cv2
import pandas as pd
import glob

parser = argparse.ArgumentParser(description='攻击的参数设置')
parser.add_argument('--models',default=['b1','inc3','des121','res50','vgg'],
                    help='使用的model是什么')

parser.add_argument('--update_dev_csv',default=False,
                    help='不更新mask_csv文件,差不多就是不使用增量迭代')
parser.add_argument('--update_mask_csv',default=False)

parser.add_argument('--transzl',default=False,help='对trans的改变搜索空间变大')
parser.add_argument('--datazl',default=True,help='对data的改变搜索空间变大')
parser.add_argument('--sample_num',default=0,help='随机采样的个数,个数越多,相对效果越好,但是时间也越多')

parser.add_argument('--pred_min',default=-5,help='在dev文件中如果使用了增量迭代,则最小的pred为该值')
parser.add_argument('--pred_max',default=5,help='在dev文件中如果使用了增量迭代,则最大的pred为该值')

parser.add_argument('--mask_tar_csv',default='path to mask_tar_csv',required=True,
                    help='在imagenet上目标攻击的每个图片的csv文件，保存了1216的每个图片的对于该对应的类的图片的重要信息')
parser.add_argument('--mask_img_dir',default='path to imageNet数据集',required=True,
                    help='imagenet的数据集的地址')
parser.add_argument('--in_bs',default=90,
                    help='内循环的btachsize')

parser.add_argument('--scope_min',default=[50/255,],help='clamp的下界[45/255,50/255,55/255]')
parser.add_argument('--scope',default=[150/255,],help='clamp的范围[140/255,150/255,160/255]')
parser.add_argument('--size',default=[100,],help='resize的大小 [80,90,100]')

parser.add_argument('--input_dir',type=str,default='path to 1216张图像',required=True,
                    help='1216张输入的图片的地址')
parser.add_argument('--dev_dir',type=str,default='path to dev.csv',required=True,
                    help='这个地址已经不仅仅是dev.csv的地址,但是第一次可以是dev.csv文件')
parser.add_argument('--sdcp',default='path to save_dev_csv_path',required=True,
                    help='保存新的csv的地址用于,save_dev_csv_path')
parser.add_argument('--mean',default=[0.485,0.456,0.406],
                    help='ImageNet的均值')
parser.add_argument('--std',default=[0.229,0.224,0.225],
                    help='ImageNet的标准差')
parser.add_argument('--yes_save_img_path',type=str,default='path to 合格的图像的保存地址',required=True,
                    help='合格的图片的保存路径')
parser.add_argument('--no_save_img_path',default='path to 不合格的图像的保存地址',required=True,
                    help='不合格图片的保存位置')
args = parser.parse_args()

#计算有多少个点是满足最大的epsilon的
def compute_e_mask(eta,epsilon):
    '''
    :param eta:纯扰动
    :param epsilon: 最大的规定扰动
    :return: 有多少个点满足做大的扰动
    '''
    condition1 = torch.le(eta,-epsilon)
    condition2 = torch.ge(eta,epsilon)
    condition = condition1+condition2
    num_sm = (1 / (args.batch_size * 3)) * len(torch.nonzero(condition))
    return num_sm

#用于外层原始数据导入
class DEVSet(Dataset):
    def __init__(self,df,transforms,data_dir,umi):
        self.df = df
        self.transform = transforms
        self.data_dir = data_dir
        self.umi = umi
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        item_i = item
        image_name = self.df.iloc[item_i]['ImageId']
        image_path = os.path.join(self.data_dir,image_name)
        img = Image.open(image_path)

        img_gs = cv2.imread(image_path)
        img_gs = cv2.GaussianBlur(img_gs,(21,21),0.0)
        # img_gs = cv2.blur(img_gs,(21,21))
        img_gs = Image.fromarray(cv2.cvtColor(img_gs,cv2.COLOR_BGR2RGB))

        mask_path = '...' #需要自己修改路径
        mask = cv2.imread(os.path.join(mask_path,image_name))
        mask = mask[:,:,0]/255

        img = self.transform(img) #原始图片
        img_gs = self.transform(img_gs)#gs铝箔的图片
        label = self.df.iloc[item_i]['TrueLabel'] #真实标签
        tar_label = self.df.iloc[item_i]['TargetClass'] #目标标签
        if self.umi !=0:
            TmaskId_0 = self.df.iloc[item_i]['TmaskId_0']
            TmaskId_1 = self.df.iloc[item_i]['TmaskId_1']
            TmaskId_2 = self.df.iloc[item_i]['TmaskId_2']
            TmaskId_3 = self.df.iloc[item_i]['TmaskId_3']
            TmaskId_4 = self.df.iloc[item_i]['TmaskId_4']
            TmaskClass_0 = self.df.iloc[item_i]['TmaskClass_0']
            TmaskClass_1 = self.df.iloc[item_i]['TmaskClass_1']
            TmaskClass_2 = self.df.iloc[item_i]['TmaskClass_2']
            TmaskClass_3 = self.df.iloc[item_i]['TmaskClass_3']
            TmaskClass_4 = self.df.iloc[item_i]['TmaskClass_4']
            TmaskId = list()
            TmaskId.extend([TmaskId_0,TmaskId_1,TmaskId_2,TmaskId_3,TmaskId_4])
            TmaskClass = list()
            TmaskClass.extend([TmaskClass_0,TmaskClass_1,TmaskClass_2,TmaskClass_3,TmaskClass_4])

            UmaskId_0 = self.df.iloc[item_i]['UmaskId_0']
            UmaskId_1 = self.df.iloc[item_i]['UmaskId_1']
            UmaskId_2 = self.df.iloc[item_i]['UmaskId_2']
            UmaskId_3 = self.df.iloc[item_i]['UmaskId_3']
            UmaskId_4 = self.df.iloc[item_i]['UmaskId_4']
            UmaskClass_0 = self.df.iloc[item_i]['UmaskClass_0']
            UmaskClass_1 = self.df.iloc[item_i]['UmaskClass_1']
            UmaskClass_2 = self.df.iloc[item_i]['UmaskClass_2']
            UmaskClass_3 = self.df.iloc[item_i]['UmaskClass_3']
            UmaskClass_4 = self.df.iloc[item_i]['UmaskClass_4']
            UmaskId = list()
            UmaskId.extend([UmaskId_0, UmaskId_1, UmaskId_2, UmaskId_3, UmaskId_4])
            UmaskClass = list()
            UmaskClass.extend([UmaskClass_0, UmaskClass_1, UmaskClass_2, UmaskClass_3, UmaskClass_4])
        else:
            TmaskId=[]
            TmaskClass=[]
            UmaskClass=[]
            UmaskId= []

        sample = {
            'dataset_idx':item,
            'img_name':image_name,
            'image':img,
            'TmaskId_list':TmaskId,
            'TmaskClass_list':TmaskClass,
            'UmaskId_list':UmaskId,
            'UmaskClass_list':UmaskClass,
            'img_blur':img_gs,
            'mask':mask,#得到的mask是[299,299]且是[0,1]
            'label':label,
            'tar_label':tar_label,
        }
        return sample
#用于1216张图像的数据导入
def test_dataloader_01(data_dir,df_dir,batch_size=32,pred_min=2.,pred_max=4.,umi=0):
    transform_test = transforms.Compose([transforms.ToTensor()])
    df = pd.read_csv(df_dir)
    # 增量迭代
    if args.update_dev_csv:
        df_zl = df[(df['Uuogap'] < pred_max) & (df['Uuogap']>pred_min)]
        df = pd.DataFrame(np.asarray(df_zl),
        columns=['ImageId','Ttogap','Uuogap','TmaskId_0','TmaskId_1','TmaskId_2','TmaskId_3','TmaskId_4','Tpred','Tpredori',
                  'Ttrans','TargetClass','TrueLabel','TmaskClass_0','TmaskClass_1','TmaskClass_2','TmaskClass_3','TmaskClass_4',
                 'UmaskId_0','UmaskId_1','UmaskId_2','UmaskId_3','UmaskId_4','UmaskClass_0','UmaskClass_1','UmaskClass_2',
                 'UmaskClass_3','UmaskClass_4','Upred','Upredori','Utrans','UpredClass'])
    test_datasets = DEVSet(df=df,transforms=transform_test,data_dir=data_dir,umi=umi)
    test_data_loaders = DataLoader(test_datasets,batch_size=batch_size,num_workers=1,shuffle=False,pin_memory=True)
    return test_data_loaders

#用于内循环的数据导入
class MASKSet(Dataset):
    def __init__(self,df,transforms,data_dir,umi,TmaskId_list,TmaskClass_list):
        self.df = df
        self.transform = transforms
        self.data_dir = data_dir
        self.umi = umi
        self.TmaskId_list = TmaskId_list
        self.TmaskClass_list = TmaskClass_list
    def __len__(self):
        if args.update_mask_csv and args.transzl:
            #由于这里我们需要得到的是4个合成一个的，所以len需要修改一下
            return len(self.df)
        else:
            return len(self.df)

    def __getitem__(self, item):
        #对imagenet_的图片做对比度增强,也就是直方图均衡化
        # img_path = os.path.join(self.data_dir,image_name)
        #直方图均衡化

        if args.update_mask_csv and args.transzl:
            #对4张图片做处理,得到4张图片的
            image_name_0 = self.df.iloc[item]['TmaskId']  # 图片的名字
            img_class_0 = self.df.iloc[item]['TmaskClass']
            img_path_0 = os.path.join(self.data_dir,str(img_class_0),image_name_0)
            img_0 = self.my_equalhist(img_path_0)
            new_img = Image.new(mode='RGB',size=(299*3,299*3))
            for i in range(1,4):
                for j in range(1,4):
                    new_img.paste(img_0,(299*(i-1),299*(j-1)))
            new_img = new_img.resize((299,299))
            img = self.transform(new_img)

            sample = {
                'img_name':image_name_0,
                'image':img,
            }
        else:
            image_name = self.df.iloc[item]['TmaskId']  # 图片的名字
            img_path = os.path.join(self.data_dir,image_name)
            img_s = Image.open(img_path)
            # img_s = self.my_equalhist(img_path)
            #其他三个保持不变的在这里处理
            if self.umi !=0:
                Tmask_name_1 = self.TmaskId_list[1][0]
                Tmask_name_2 = self.TmaskId_list[2][0]
                Tmask_name_3 = self.TmaskId_list[3][0]
                Tmask_name_4 = self.TmaskId_list[4][0]
                Tmask_Class_1 = str(self.TmaskClass_list[1].item())
                Tmask_Class_2 = str(self.TmaskClass_list[2].item())
                Tmask_Class_3 = str(self.TmaskClass_list[3].item())
                Tmask_Class_4 = str(self.TmaskClass_list[4].item())
                root_dir = '...' #需要自己修改这个路径
                Tmask_path_1 = os.path.join(root_dir,Tmask_Class_1,Tmask_name_1)
                Tmask_path_2 = os.path.join(root_dir,Tmask_Class_2,Tmask_name_2)
                Tmask_path_3 = os.path.join(root_dir,Tmask_Class_3,Tmask_name_3)
                Tmask_path_4 = os.path.join(root_dir,Tmask_Class_4,Tmask_name_4)
                Tmask_1 = Image.open(Tmask_path_1)
                Tmask_2 = Image.open(Tmask_path_2)
                Tmask_3 = Image.open(Tmask_path_3)
                Tmask_4 = Image.open(Tmask_path_4)


            new_img = Image.new(mode='RGB', size=(299 * 2, 299 * 2))
            if self.umi==0:
                for i in range(1, 3):
                    for j in range(1, 3):
                        new_img.paste(img_s, (299 * (i - 1), 299 * (j - 1)))
            if self.umi ==1:
                new_img.paste(img_s,(0,0))
                new_img.paste(Tmask_2,(299,0))
                new_img.paste(Tmask_3,(0,299))
                new_img.paste(Tmask_4,(299,299))
            if self.umi ==2:
                new_img.paste(Tmask_1, (0, 0))
                new_img.paste(img_s, (299, 0))
                new_img.paste(Tmask_3, (0, 299))
                new_img.paste(Tmask_4, (299, 299))
            if self.umi ==3:
                new_img.paste(Tmask_1, (0, 0))
                new_img.paste(Tmask_2, (299, 0))
                new_img.paste(img_s, (0, 299))
                new_img.paste(Tmask_4, (299, 299))
            if self.umi ==4:
                new_img.paste(Tmask_1, (0, 0))
                new_img.paste(Tmask_2, (299, 0))
                new_img.paste(Tmask_3, (0, 299))
                new_img.paste(img_s, (299, 299))

            new_img = new_img.resize((299, 299))
            new_img = self.my_equalhist(new_img)
            img = self.transform(new_img)
            # s_image = img_s.resize((150,150))
            # s_image = self.my_equalhist(s_image)
            # s_image = self.transform(s_image)
            sample = {
                'dataset_idx':item,
                'img_name':image_name,
                # 's_image':s_image,
                'image':img,
            }
        return sample
    #还需要思考
    def my_transform(self,img,mask):
        #注意输入的要是PIL的图片
        #随机水平翻转
        if np.random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        #随机缩放
        #如果要使用mask的话，还需要对mask做处理，而且感觉这样子的效率不好，使用连通域外接矩形？
        if np.random.random() > 0.5:
            random_size = random.randint(int(299*0.5),int(299*1.5))
            img = img.resize((random_size,random_size),Image.NEAREST)
            mask = mask.resize((random_size,random_size),Image.NEAREST)

    def my_equalhist(self,img_o_0):
        #输入图片的地址，返回经过直方图均衡化的图
        img_YCB_0 = cv2.cvtColor(np.asarray(img_o_0),cv2.COLOR_RGB2YCrCb)
        Y_0, C_0, B_0 = cv2.split(img_YCB_0)
        equalclash =cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
        Y_0 = equalclash.apply(Y_0)
        channels_0 = cv2.merge((Y_0, C_0, B_0))
        img_0 = Image.fromarray(cv2.cvtColor(channels_0, cv2.COLOR_YCrCb2RGB)).convert('RGB')
        return img_0
#用于每张图像对应的目标类的图像的数据导入
def mask_dataloader_01(data_dir,df_dir,batch_size,umi,TmaskId_list,TmaskClass_list):
    transform_test = transforms.Compose([transforms.ToTensor()])

    #做增量只是为了减少计算量
    if args.transzl:
        ori_df = pd.read_csv(df_dir)
        #如果是transzl的话就相当于说在原有的基础上做
        zl_75_num = ori_df['Ttogap'].describe()['50%'] #获取前50%的最小值
        #可以做增量迭代了
        df_zl = ori_df[(ori_df['Ttogap']> zl_75_num)] #选取前0.5的图片
        df = pd.DataFrame(np.asarray(df_zl),
        columns=['TmaskId','TmaskClass','Ttrans','Tpred','TargetClass','TrueLabel','Tpredori','Ttogap','Utrans','Uuogap','UpredClass'])
    else:
        #如果是数据增量的的话就是探索新的东西
        all_imgs = sorted(glob.glob(os.path.join(data_dir, '*.*')))
        all_names = [os.path.basename(all_img) for all_img in all_imgs]
        len_all_labels = len(all_imgs)
        index = []
        for i in range(len_all_labels):
            index.append(i)
        df = pd.DataFrame({'ImagePath': all_imgs, 'TmaskId': all_names, 'index': index})

    test_datasets = MASKSet(df=df, transforms=transform_test, data_dir=data_dir,umi=umi,
                            TmaskId_list=TmaskId_list,TmaskClass_list=TmaskClass_list)
    test_data_loaders = DataLoader(test_datasets, batch_size=batch_size, num_workers=16,
                                   shuffle=True, pin_memory=True,drop_last=False)
    return test_data_loaders

#PGD白盒攻击 虽然叫PGD白盒攻击，其实已经不是了
def PGD_white(input,models,tar_lab,true_label,mean,std,img_name=None,yes=0,img_blur=None,T_update=0,U_update=0,mask_dev=None,
              TmaskId_list=None,TmaskClass_list = None,UmaskClass_list =None,UmaskId_list = None,umi=None):

    # 设置一个imagenet的csv文件，每个原始图片的，或者是1216的正常的以后的
    #  需要自己修改下面这个路径
    target_list_dir = os.listdir('....') #得到所有的tar的类
    bs =args.in_bs #mask的batch_size是多少
    mn = 5 # 使用的模型的个数
    Tmax_gap = -10
    Umax_gap = -10
    int_tar = int(tar_lab.detach().cpu().numpy()+1) #用于建立目录 需要加1
    int_tru = int(true_label.detach().cpu().numpy()+1) #用于建立csv文件的,需要加1
    mask_csv_path =args.mask_tar_csv #攻击中的masktar的信息地址
    if not os.path.exists(mask_csv_path):os.makedirs(mask_csv_path)
    csv_path_i = os.path.join(mask_csv_path,str(img_name[0][:-4])+'.csv')

    #用于迭代更新的最终的,所以需要先读进去
    if umi !=0:
        dev_df = pd.read_csv(args.sdcp)

    # 循环迭代多少次呢?
    for scope_min in args.scope_min:
        for scope in args.scope:
            for size in args.size:
                # random_class = random.sample(target_list_dir,3)
                if args.transzl:
                    #transzl是自己的地址,不用传给他,但是这里值相当于一个循环
                    random_class=random.sample(target_list_dir,0)
                    random_class.insert(0,str(int_tar))
                elif args.datazl:
                    #datazl是从有的列表中随机采样一些出来
                    random_class = random.sample(target_list_dir,args.sample_num)
                    # random_class.insert(0,str(int_tar))
                    random_class.extend(['600','490'])
                else:
                    #其他的就是普通的,就是开始的那中
                    random_class = random.sample(target_list_dir,2)
                    random_class.insert(0,str(int_tar))
                for r_c in random_class:
                    if args.transzl:
                        data_dir_i = args.mask_img_dir
                    else:
                        data_dir_i = os.path.join(args.mask_img_dir, r_c)
                    mask_dataloader = mask_dataloader_01(data_dir=data_dir_i,df_dir=csv_path_i, batch_size=bs,umi=umi,
                                    TmaskId_list=TmaskId_list,TmaskClass_list=TmaskClass_list)
                    for masks in mask_dataloader:
                        mask = masks['image'].to('cuda') #得到mask对应的原图
                        bs_size = mask.size()[0] #为什么不直接用bs呢?因为最后一个周期会少,那直接drop_last多可惜啊
                        mask_name = masks['img_name']
                        # s_mask = masks['s_image'].to('cuda') #小的图
                        input_mask = torch.zeros((bs_size, 3, 299, 299), device='cuda')  # 初始化input_mask的
                        output = torch.zeros((mn, bs_size, 1000))  # 初始化output
                        pred_t = torch.zeros((bs_size))  # 初始化预测的概率
                        pred_o = torch.zeros((bs_size))  # 初始化原始预测的概率
                        pred_u = torch.zeros((bs_size))  # 无目标的预测的概率
                        trans = f'{int(scope_min*255):2}_{int(scope*255):3}_{int(size):2}'
                        mask = torch.clamp(mask,scope_min,scope+scope_min) -scope_min

                        for i in range(bs_size):
                            #这个参数无目标是越大越好，有目标是相对较小也比较好
                            input_mask[i] = img_blur[0]+((mask[i]*(size/(scope*255))-(0.5*size/255)))

                        eta = input_mask - input
                        eta = torch.clamp(eta,-32/255,32/255)
                        input_mask = input+eta
                        input_mask = torch.clamp(input_mask,0.,1.)

                        input_mask_m = input_mask.data.sub(mean[:,None,None]).div(std[:,None,None])
                        output[0] = models['b1'](input_mask_m)
                        output[1] = models['res50'](input_mask_m)
                        output[2] = models['des121'](input_mask_m)
                        output[3] = models['inc3'](input_mask_m)
                        output[4] = models['vgg'](input_mask_m)

                        output_all = output[0]+output[1]+output[2]+output[3]+output[4]
                        pred_class = torch.max(output_all,dim=1)[1]

                        for i in range(bs_size):
                            for j in range(mn):
                                pred_t[i] += torch.softmax(output[j][i],dim=0)[tar_lab].item()
                                pred_o[i] += torch.softmax(output[j][i],dim=0)[true_label].item()
                                if pred_class[i] !=true_label.cpu():
                                    pred_u[i] += torch.softmax(output[j][i],dim=0)[pred_class[i]].item()
                                else:
                                    pred_u[i] =0.
                        #目标类的概率于原始类的概率
                        pred_tgap = pred_t-pred_o
                        pred_ugap = pred_u-pred_o

                        for i in range(bs_size):
                            #这个是有目标的
                            if pred_tgap[i] > Tmax_gap:
                                Tmax_gap = pred_tgap[i] #最大有目标的gap
                                Tpred = pred_t[i]
                                Tpredori = pred_o[i]
                                TmaskId = mask_name[i]
                                Ttrans = trans
                                TmaskClass = r_c
                                Tmax_input_mask = input_mask[i]
                            #这个是无目标的
                            if pred_ugap[i] > Umax_gap:
                                Umax_gap = pred_ugap[i]
                                Upred = pred_u[i]
                                UmaskId = mask_name[i]
                                Upredori = pred_o[i]
                                Upredclass = pred_class[i].item()
                                UmaskClass = r_c
                                Utrans = trans
                                Umax_input_mask = input_mask[i]

                    if Tmax_gap >4.5:
                        yes+=1
                        npimg = (Tmax_input_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        npimg = np.transpose(npimg, (1, 2, 0))
                        save_img_mask = Image.fromarray(npimg)
                        save_yes_img_path = args.yes_save_img_path
                        if not os.path.exists(save_yes_img_path): os.makedirs(save_yes_img_path)
                        save_img_mask.save(save_yes_img_path+str(img_name[0]),quality=100)
                        #原始的开始
                        if umi ==0:
                            with open(file=args.sdcp, mode='a') as f:
                                print(f'{img_name[0]},{Tmax_gap:.5f},{Umax_gap:.5f},{TmaskId},{TmaskId},{TmaskId},{TmaskId},{TmaskId},'
                                      f'{Tpred:.5f},{Tpredori:.5f},{Ttrans},{int_tar},{int_tru},'
                                      f'{TmaskClass},{TmaskClass},{TmaskClass},{TmaskClass},{TmaskClass},'
                                      f'{UmaskId},{UmaskId},{UmaskId},{UmaskId},{UmaskId},'
                                      f'{UmaskClass},{UmaskClass},{UmaskClass},{UmaskClass},{UmaskClass},'
                                      f'{Upred:.5f},{Upredori:.5f},{Utrans},{Upredclass + 1}',
                                      file=f)
                        #增量更新
                        else:
                            ori_Tmax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Ttogap']
                            ori_Umax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Uuogap']
                            if ori_Tmax_gap.item() < Tmax_gap.detach().cpu().numpy().item():
                                m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
                                dev_df.loc[m_index, 'Ttogap'] = round(Tmax_gap.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Tpred'] = round(Tpred.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Tpredori'] = round(Tpredori.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Ttrans'] = Ttrans
                                dev_df.loc[m_index, f'TmaskClass_{umi}'] = TmaskClass
                                dev_df.loc[m_index, f'TmaskId_{umi}'] = TmaskId
                                T_update += 1
                                print(f'T增加{T_update} T:{ori_Tmax_gap.item():.5f}--{Tmax_gap:.5f} umi:{umi} id:{TmaskId}')
                            if ori_Umax_gap.item() < Umax_gap.detach().cpu().numpy().item():
                                m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
                                dev_df.loc[m_index, 'Uuogap'] = round(Umax_gap.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Upred'] = round(Upred.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Upredori'] = round(Upredori.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Utrans'] = Utrans
                                dev_df.loc[m_index, f'UmaskClass_{umi}'] = UmaskClass
                                dev_df.loc[m_index, 'UpredClass'] = Upredclass+1
                                dev_df.loc[m_index, f'UmaskId_{umi}'] = UmaskId
                                U_update += 1
                                print(f'U增加{U_update} U:{ori_Umax_gap.item():.5f}--{Umax_gap:.5f} umi:{umi} id:{UmaskId}')
                            dev_df.to_csv(args.sdcp,index=False)
                        return yes,T_update,U_update

    if Umax_gap - Tmax_gap >2.2:
        save_TorU = Umax_input_mask
    else:
        save_TorU = Tmax_input_mask
    npimg = (save_TorU.detach().cpu().numpy() * 255).astype(np.uint8)
    npimg = np.transpose(npimg, (1, 2, 0))
    save_img_mask = Image.fromarray(npimg)
    save_no_img_path = args.no_save_img_path
    if not os.path.exists(save_no_img_path):os.makedirs(save_no_img_path)
    save_img_mask.save(save_no_img_path + str(img_name[0]), quality=100)

    # 原始的开始
    if umi==0:
        with open(file=args.sdcp, mode='a') as f:
            print(f'{img_name[0]},{Tmax_gap:.5f},{Umax_gap:.5f},{TmaskId},{TmaskId},{TmaskId},{TmaskId},{TmaskId},'
                  f'{Tpred:.5f},{Tpredori:.5f},{Ttrans},{int_tar},{int_tru},'
                  f'{TmaskClass},{TmaskClass},{TmaskClass},{TmaskClass},{TmaskClass},'
                  f'{UmaskId},{UmaskId},{UmaskId},{UmaskId},{UmaskId},'
                  f'{UmaskClass},{UmaskClass},{UmaskClass},{UmaskClass},{UmaskClass},'
                  f'{Upred:.5f},{Upredori:.5f},{Utrans},{Upredclass + 1}',
                  file=f)

    # 增量更新
    else:
        ori_Tmax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Ttogap']
        ori_Umax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Uuogap']
        if ori_Tmax_gap.item() < Tmax_gap.detach().cpu().numpy().item():
            m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
            dev_df.loc[m_index, 'Ttogap'] = round(Tmax_gap.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Tpred'] = round(Tpred.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Tpredori'] = round(Tpredori.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Ttrans'] = Ttrans
            dev_df.loc[m_index, f'TmaskClass_{umi}'] = TmaskClass
            dev_df.loc[m_index, f'TmaskId_{umi}'] = TmaskId
            T_update += 1
            print(f'T增加{T_update} T:{ori_Tmax_gap.item():.5f}--{Tmax_gap:.5f} umi:{umi} id:{TmaskId}')
        if ori_Umax_gap.item() < Umax_gap.detach().cpu().numpy().item():
            m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
            dev_df.loc[m_index, 'Uuogap'] = round(Umax_gap.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Upred'] = round(Upred.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Upredori'] = round(Upredori.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Utrans'] = Utrans
            dev_df.loc[m_index, f'UmaskClass_{umi}'] = UmaskClass
            dev_df.loc[m_index, 'UpredClass'] = Upredclass + 1
            dev_df.loc[m_index, f'UmaskId_{umi}'] = UmaskId
            U_update += 1
            print(f'U增加{U_update} U:{ori_Umax_gap.item():.5f}--{Umax_gap:.5f} umi:{umi} id:{UmaskId}')
        dev_df.to_csv(args.sdcp,index=False)
    return yes,T_update,U_update

def main(args):
    yes = 0 #记录有多少个达到了设定的阈值以上的
    T_update =0 #记录哟多少个是更新了有目标的
    U_update =0 #记录有多少个是更新了无目标的
    # tool_1.seed_torch() # 随机种子
    device = torch.device('cuda:0')
    #dataloader
    # test_dataloader = test_dataloader_01(data_dir=args.input_dir,
    #                         pred_min=args.pred_min,pred_max=args.pred_max,df_dir=args.dev_dir,batch_size=1,umi=umi)
    #model的列表
    model1 = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b1').eval().to(device)
    model2 = models.resnet50(pretrained=True).eval().to(device)
    model3 = models.densenet121(pretrained=True).eval().to(device)
    model4 = models.inception_v3(pretrained=True).eval().to(device)
    model5 = models.vgg16(pretrained=True).eval().to(device)
    model_dict={'b1':model1,'res50':model2,'des121':model3,'inc3':model4,'vgg':model5}
    mean = torch.tensor(args.mean, device=device)
    std = torch.tensor(args.std, device=device)

    for umi in [0,1,2,3,4]:#我要跟新那个图片的id
        if umi ==0:
            with open(file=args.sdcp, mode='w') as f:
                print('ImageId,Ttogap,Uuogap,TmaskId_0,TmaskId_1,TmaskId_2,TmaskId_3,TmaskId_4,Tpred,Tpredori,'
                      'Ttrans,TargetClass,TrueLabel,TmaskClass_0,TmaskClass_1,TmaskClass_2,TmaskClass_3,TmaskClass_4,'
                      'UmaskId_0,UmaskId_1,UmaskId_2,UmaskId_3,UmaskId_4,UmaskClass_0,UmaskClass_1,UmaskClass_2,'
                      'UmaskClass_3,UmaskClass_4,Upred,Upredori,Utrans,UpredClass', file=f)
        if umi ==0:
            test_dataloader = test_dataloader_01(data_dir=args.input_dir,
                                                 pred_min=args.pred_min, pred_max=args.pred_max, df_dir=args.dev_dir,
                                                 batch_size=1,umi=umi)
        else:
            test_dataloader = test_dataloader_01(data_dir=args.input_dir,
                                                 pred_min=args.pred_min, pred_max=args.pred_max, df_dir=args.sdcp,
                                                 batch_size=1, umi=umi)

        for sample in tqdm(test_dataloader):
            img_name = sample['img_name'] #1216张图的名字
            data = sample['image'].to(device) #input的数据为[0-1]
            img_blur = sample['img_blur'].to(device) #经过gaussian bulr 得到的图片,但是实际上没有更好,所以kenerl设为0
            mask = sample['mask'].to(device)
            tar_label = sample['tar_label'].to(device)-1 #由于dev_csv中label是多了1的所以这里减1
            true_label = sample['label'].to(device)-1
            TmaskId_list = sample['TmaskId_list']
            TmaskClass_list = sample['TmaskClass_list']
            UmaskId_list = sample['UmaskId_list']
            UmaskClass_list = sample['UmaskClass_list']

            with torch.no_grad():
                yes,T_update,U_update = PGD_white(input=data,models=model_dict,true_label=true_label,
                tar_lab=tar_label,mean=mean,std=std,img_name=img_name,yes=yes,TmaskId_list = TmaskId_list,
                img_blur=img_blur,T_update=T_update,U_update=U_update,mask_dev = mask,umi=umi,
                TmaskClass_list = TmaskClass_list,UmaskId_list=UmaskId_list,UmaskClass_list=UmaskClass_list)

if __name__=='__main__':
    main(args)
