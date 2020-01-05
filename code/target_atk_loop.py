
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

parser.add_argument('--update_mask_csv',default=True,
                    help='不更新mask_csv文件,差不多就是不使用增量迭代')
parser.add_argument('--transzl',default=False,help='对trans的改变搜索空间变大')
parser.add_argument('--datazl',default=True,help='对data的改变搜索空间变大')
parser.add_argument('--sample_num',default=10,help='随机采样的个数,个数越多,相对效果越好,但是时间也越多')

parser.add_argument('--pred_min',default=-5,help='在dev文件中如果使用了增量迭代,则最小的pred为该值')
parser.add_argument('--pred_max',default=5,help='在dev文件中如果使用了增量迭代,则最大的pred为该值')

parser.add_argument('--mask_tar_csv',default='path to mask_csv',required=True,
                    help='在imagenet上目标攻击的每个图片的csv文件，保存了1216的每个图片的对于该对应的类的图片的重要信息')
parser.add_argument('--mask_img_dir',default='path to imagenet的数据集',required=True,
                    help='当做mask的图片的地址')
parser.add_argument('--in_bs',default=100,
                    help='内循环的btachsize')

parser.add_argument('--scope_min',default=[50/255,],help='clamp的下界[45/255,50/255,55/255]')
parser.add_argument('--scope',default=[150/255,],help='clamp的范围[140/255,150/255,160/255]')
parser.add_argument('--size',default=[100,],help='resize的大小 [80,90,100]')

parser.add_argument('--input_dir',type=str,default='path to 1216张images',required=True,
                    help='1216张输入的图片的地址')
parser.add_argument('--dev_dir',type=str,default='path to dev.csv',required=True,
                    help='这个地址已经不是dev.csv的地址了，是我自己的地址，要有判断的')
parser.add_argument('--sdcp',default='path to save_another_dev_csv_path',required=True,
                    help='保存上面的这个所谓的dev.csv的地址,save_dev_csv_path')
parser.add_argument('--mean',default=[0.485,0.456,0.406],
                    help='ImageNet的均值')
parser.add_argument('--std',default=[0.229,0.224,0.225],
                    help='ImageNet的标准差')
parser.add_argument('--yes_save_img_path',type=str,default='path to 合格的图像的地址',required=True,
                    help='合格的图片的保存路径')
parser.add_argument('--no_save_img_path',default='path to 不合格的图像的地址',required=True,
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
    def __init__(self,df,transforms,data_dir):
        self.df = df
        self.transform = transforms
        self.data_dir = data_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        image_name = self.df.iloc[item]['ImageId']
        image_path = os.path.join(self.data_dir,image_name)
        img = Image.open(image_path)

        img_gs = cv2.imread(image_path)
        img_gs = cv2.GaussianBlur(img_gs,(21,21),0.0)
        img_gs = Image.fromarray(cv2.cvtColor(img_gs,cv2.COLOR_BGR2RGB))

        img = self.transform(img)
        img_gs = self.transform(img_gs)
        label = self.df.iloc[item]['TrueLabel']
        tar_label = self.df.iloc[item]['TargetClass']
        sample = {
            'dataset_idx':item,
            'img_name':image_name,
            'image':img,
            'img_blur':img_gs,
            'label':label,
            'tar_label':tar_label,
        }
        return sample

def test_dataloader_01(data_dir,df_dir,batch_size=32,pred_min=2.,pred_max=4.):
    transform_test = transforms.Compose([transforms.ToTensor()])
    df = pd.read_csv(df_dir)
    # 增量迭代
    if args.update_mask_csv:
        df_zl = df[(df['Uuogap'] < pred_max) & (df['Uuogap']>pred_min)]
        df = pd.DataFrame(np.asarray(df_zl),
        columns=['ImageId','Ttogap','Uuogap','TmaskId','Tpred','Tpredori',
                  'Ttrans','TargetClass','TrueLabel','TmaskClass','UmaskId','UmaskClass','Upred','Upredori','Utrans','UpredClass'])
    test_datasets = DEVSet(df=df,transforms=transform_test,data_dir=data_dir)
    test_data_loaders = DataLoader(test_datasets,batch_size=batch_size,num_workers=1,shuffle=False,pin_memory=True)
    return test_data_loaders

#用于内循环的数据导入
class MASKSet(Dataset):
    def __init__(self,df,transforms,data_dir):
        self.df = df
        self.transform = transforms
        self.data_dir = data_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        #对imagenet_的图片做对比度增强,也就是直方图均衡化
        image_name = self.df.iloc[item]['TmaskId'] #图片的名字
        # img_path = os.path.join(self.data_dir,image_name)
        if args.update_mask_csv and args.transzl:
            img_class = self.df.iloc[item]['TmaskClass']
            img_path = os.path.join(self.data_dir,img_class,image_name)
        else:
            img_path = os.path.join(self.data_dir,image_name)
        img_o = cv2.imread(img_path)
        img_YCB = cv2.cvtColor(img_o,cv2.COLOR_BGR2YCrCb)
        Y,C,B = cv2.split(img_YCB)
        #自适应直方图均衡化
        equal_clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(10,10))
        Y = equal_clahe.apply(Y)
        #普通直方图均衡化
        # Y = cv2.equalizeHist(Y)
        channels = cv2.merge((Y,C,B))
        img = Image.fromarray(cv2.cvtColor(channels,cv2.COLOR_YCrCb2RGB)).convert('RGB')
        img = self.transform(img)

        sample = {
            'dataset_idx':item,
            'img_name':image_name,
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

def mask_dataloader_01(data_dir,df_dir,batch_size):
    transform_test = transforms.Compose([transforms.ToTensor()])

    #做增量只是为了减少计算量
    if args.transzl:
        ori_df = pd.read_csv(df_dir)
        #如果是transzl的话就相当于说在原有的基础上做
        zl_75_num = ori_df['Ttogap'].describe()['75%'] #获取前75%的最小值
        #可以做增量迭代了
        df_zl = ori_df[(ori_df['Ttogap']> zl_75_num)] #选取前0.75的图片
        df = pd.DataFrame(np.asarray(df_zl),
        columns=['TmaskId','TmaskClass','Ttrans','Tpred','TargetClass','TrueLabel','Tpredori','Ttogap','Utrans','Uuogap','UpredClass'])
    else:
        #如果是数据增量的的话就是探索新的东西
        all_imgs = sorted(glob.glob(os.path.join(data_dir, '*.JPEG')))
        all_names = [os.path.basename(all_img) for all_img in all_imgs]
        len_all_labels = len(all_imgs)
        index = []
        for i in range(len_all_labels):
            index.append(i)
        df = pd.DataFrame({'ImagePath': all_imgs, 'TmaskId': all_names, 'index': index})

    test_datasets = MASKSet(df=df, transforms=transform_test, data_dir=data_dir)
    test_data_loaders = DataLoader(test_datasets, batch_size=batch_size, num_workers=8,
                                   shuffle=False, pin_memory=True,drop_last=False)
    return test_data_loaders

#PGD白盒攻击 虽然叫PGD白盒攻击，其实已经不是了
def PGD_white(input,models,tar_lab,true_label,mean,std,img_name=None,yes=0,img_blur=None,T_update=0,U_update=0):

    # TODO 我该怎么解决速度比较慢的问题呢？ 重要的与不重要的？ 增量迭代
    # 设置一个imagenet的csv文件，每个原始图片的，或者是1216的正常的以后的
    # 文件结构是怎么样的呢？传入一个图片名，然后呢得到该名字的csv文件，该文件中有每个变化时所对应的的prediction
    # imgname.csv -MaskId,trans,pred,TargetClass,TrueLabel
    target_list_dir = os.listdir('...') #得到所有的tar的类 需要自己修改路径
    bs =args.in_bs #mask的batch_size是多少
    mn = 5 # 使用的模型的个数
    Tmax_gap = -10
    Umax_gap = -10
    int_tar = int(tar_lab.detach().cpu().numpy()+1) #用于建立目录 需要加1
    int_tru = int(true_label.detach().cpu().numpy()+1) #用于建立csv文件的,需要加1
    mask_csv_path =args.mask_tar_csv #攻击中的masktar的信息地址
    if not os.path.exists(mask_csv_path):os.makedirs(mask_csv_path)
    csv_path_i = os.path.join(mask_csv_path,str(img_name[0][:-4])+'.csv')

    if not args.update_mask_csv:
        #如果不存在这个文件，则是新的需要创建一个
        with open(csv_path_i,'w') as f:
            print('TmaskId,TmaskClass,Ttrans,Tpred,TargetClass,TrueLabel,Tpredori,Ttogap,Utrans,Uuogap,UpredClass',file=f)

    #用于迭代更新的最终的,所以需要先读进去
    dev_df = pd.read_csv(args.dev_dir)

    # 循环迭代多少次呢?
    for scope_min in args.scope_min:
        for scope in args.scope:
            for size in args.size:
                #随机采样3个,个数越多则速度越慢
                # random_class = random.sample(target_list_dir,3)
                if args.transzl:
                    #transzl是自己的地址,不用传给他,但是这里值相当于一个循环
                    random_class=random.sample(target_list_dir,0)
                    # random_class.insert(0,str(int_tar))
                elif args.datazl:
                    #datazl是从有的列表中随机采样一些出来
                    # random_class = list()
                    random_class = random.sample(target_list_dir,args.sample_num)
                    random_class.insert(0,str(int_tar))
                    # random_class.extend(['996','998','999','988','945'])
                else:
                    #其他的就是普通的,就是开始的那中
                    random_class = random.sample(target_list_dir,2)
                    random_class.insert(0,str(int_tar))
                for r_c in random_class:
                    #为了修改mask_csv_df，需要先读取
                    # mask_csv_df = pd.read_csv(csv_path_i)
                    if args.transzl:
                        data_dir_i = args.mask_img_dir
                    else:
                        data_dir_i = os.path.join(args.mask_img_dir, r_c)
                    mask_dataloader = mask_dataloader_01(data_dir=data_dir_i,df_dir=csv_path_i, batch_size=bs)
                    for masks in mask_dataloader:
                        img = masks['image'].to('cuda') #得到mask对应的原图
                        im_size = img.size()[0] #为什么不直接用bs呢?因为最后一个周期会少,那直接drop_last多可惜啊
                        mask_name = masks['img_name']
                        input_mask = torch.zeros((im_size, 3, 299, 299), device='cuda')  # 初始化input_mask的
                        output = torch.zeros((mn, im_size, 1000))  # 初始化output
                        pred_t = torch.zeros((im_size))  # 初始化预测的概率
                        pred_o = torch.zeros((im_size))  # 初始化原始预测的概率
                        pred_u = torch.zeros((im_size))  # 无目标的预测的概率
                        trans = f'{int(scope_min*255):2}_{int(scope*255):3}_{int(size):2}'
                        img = torch.clamp(img,scope_min,scope+scope_min) -scope_min
                        for i in range(im_size):
                            #这个参数无目标是越大越好，有目标是相对较小也比较好
                            input_mask[i] = img_blur[0]+((img[i]*(size/(scope*255))-(0.5*size/255)))

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

                        for i in range(im_size):
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

                        #如果不更新mask_csv文件
                        if  not args.update_mask_csv:
                            with open(csv_path_i,'a') as f:
                                for i in range(im_size):
                                    print(f'{mask_name[i]},{r_c},{trans},{pred_t[i]:.5f},'
                                          f'{int_tar},{int_tru},{pred_o[i]:.5f},{pred_tgap[i]:.5f},{trans},{pred_ugap[i]:.5f},{int(pred_class[i].item())+1}',file=f)

                        #查看最好的提升是那个,提什么了多少,是什么样子
                        for i in range(im_size):
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

                    # if args.update_mask_csv:
                        #如果要更新则将跟新以后的东西存回去
                        # mask_csv_df.to_csv(csv_path_i,index=False)


                    if Umax_gap >4.7:
                        #如果这次循环得到了大于0.8的则停止搜索.
                        # TODO 注意这个需要修改,因为在增量迭代中,保存图片应该是保存最好的那个,但是如果,增量搜索的没有那么好的话,不保存
                        # TODO 但是相对于保存图片来说,更西药关注的是csv文件的更新问题
                        yes+=1
                        npimg = (Tmax_input_mask.detach().cpu().numpy() * 255).astype(np.uint8)
                        npimg = np.transpose(npimg, (1, 2, 0))
                        save_img_mask = Image.fromarray(npimg)
                        save_yes_img_path = args.yes_save_img_path
                        if not os.path.exists(save_yes_img_path): os.makedirs(save_yes_img_path)
                        save_img_mask.save(save_yes_img_path+str(img_name[0]),quality=100)

                        #原始的开始
                        if not args.update_mask_csv:
                            with open(file=args.sdcp, mode='a') as f:
                                print(f'{img_name[0]},{Tmax_gap:.5f},{Umax_gap:.5f},{TmaskId},{Tpred:.5f},{Tpredori:.5f},{Ttrans},{int_tar},{int_tru},'
                                      f'{TmaskClass},{UmaskId},{UmaskClass},{Upred:.5f},{Upredori:.5f},{Utrans},{Upredclass + 1}',
                                      file=f)

                        #增量更新
                        else:
                            ori_Tmax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Ttogap'].item()
                            ori_Umax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Uuogap'].item()
                            if ori_Tmax_gap < Tmax_gap.detach().cpu().numpy().item():
                                m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
                                dev_df.loc[m_index, 'Ttogap'] = round(Tmax_gap.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Tpred'] = round(Tpred.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Tpredori'] = round(Tpredori.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Ttrans'] = Ttrans
                                dev_df.loc[m_index, 'TmaskClass'] = TmaskClass
                                dev_df.loc[m_index, 'TmaskId'] = TmaskId
                                T_update += 1
                                print(f'T增加{T_update} T:{ori_Tmax_gap:.5f}--{Tmax_gap:.5f} id:{TmaskId} TC:{TmaskClass}')
                            if ori_Umax_gap < Umax_gap.detach().cpu().numpy().item():
                                m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
                                dev_df.loc[m_index, 'Uuogap'] = round(Umax_gap.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Upred'] = round(Upred.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Upredori'] = round(Upredori.detach().cpu().numpy().item(), 5)
                                dev_df.loc[m_index, 'Utrans'] = Utrans
                                dev_df.loc[m_index, 'UmaskClass'] = UmaskClass
                                dev_df.loc[m_index, 'UpredClass'] = Upredclass+1
                                dev_df.loc[m_index, 'UmaskId'] = UmaskId
                                U_update += 1
                                print(f'U增加{U_update} U:{ori_Umax_gap:.5f}--{Umax_gap:.5f} id:{UmaskId} Uc:{UmaskClass}')
                            dev_df.to_csv(args.dev_dir,index=False)


                        # print(f'Tgap:{Tmax_gap:.4f} Tp:{Tpred:.4f} To{Tpredori:.4f} TId:{TmaskId} Tc:{int_tar}')
                        # print(f'Ugap:{Umax_gap:.4f} Up:{Upred:.4f} o:{Upredori:.4f} UId:{UmaskId} Uc:{Upredclass+1}')
                        # print(f'当前的yes为:{yes}')
                        return yes,T_update,U_update

    if Umax_gap - Tmax_gap >2.5:
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
    if not args.update_mask_csv:
        with open(file=args.sdcp, mode='a') as f:
            print(f'{img_name[0]},{Tmax_gap:.5f},{Umax_gap:.5f},{TmaskId},{Tpred:.5f},{Tpredori:.5f},{Ttrans},{int_tar},{int_tru},'
                  f'{TmaskClass},{UmaskId},{UmaskClass},{Upred:.5f},{Upredori:.5f},{Utrans},{Upredclass+1}', file=f)

    # 增量更新
    else:
        ori_Tmax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Ttogap'].item()
        ori_Umax_gap = dev_df.loc[dev_df['ImageId'] == img_name[0], 'Uuogap'].item()
        if ori_Tmax_gap < Tmax_gap.detach().cpu().numpy().item():
            m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
            dev_df.loc[m_index, 'Ttogap'] = round(Tmax_gap.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Tpred'] = round(Tpred.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'Tpredori'] = round(Tpredori.detach().cpu().numpy().item(), 5)
            dev_df.loc[m_index, 'TmaskClass'] = TmaskClass
            dev_df.loc[m_index, 'Ttrans'] = Ttrans
            dev_df.loc[m_index, 'TmaskId'] = TmaskId
            T_update+=1
            print(f'T增加{T_update} T:{ori_Tmax_gap:.5f}--{Tmax_gap:.5f} id:{TmaskId} TC:{TmaskClass}')
        if ori_Umax_gap < Umax_gap.detach().cpu().numpy().item():
            m_index = dev_df.loc[dev_df['ImageId'] == img_name[0]].index.item()
            dev_df.loc[m_index, 'Uuogap'] = round(Umax_gap.detach().cpu().numpy().item(),5)
            dev_df.loc[m_index, 'Upred'] = round(Upred.detach().cpu().numpy().item(),5)
            dev_df.loc[m_index, 'Upredori'] = round(Upredori.detach().cpu().numpy().item(),5)
            dev_df.loc[m_index, 'Utrans'] = Utrans
            dev_df.loc[m_index, 'UmaskClass'] = UmaskClass
            dev_df.loc[m_index, 'UpredClass'] = Upredclass+1
            dev_df.loc[m_index, 'UmaskId'] = UmaskId
            U_update+=1
            print(f'U增加{U_update} U:{ori_Umax_gap:.5f}--{Umax_gap:.5f} id:{UmaskId} UC:{UmaskClass}')
        dev_df.to_csv(args.dev_dir,index=False)

    # print(f'Tgap:{Tmax_gap:.4f} Tp:{Tpred:.4f} To{Tpredori:.4f} TId:{TmaskId} Tc:{int_tar}')
    # print(f'Ugap:{Umax_gap:.4f} Up:{Upred:.4f} o:{Upredori:.4f} UId:{UmaskId} Uc:{Upredclass+1}')
    # print(f'当前的yes为:{yes}')
    return yes,T_update,U_update

def main(args):
    yes = 0 #记录有多少个达到了设定的阈值以上的
    T_update =0 #记录哟多少个是更新了有目标的
    U_update =0 #记录有多少个是更新了无目标的
    # tool_1.seed_torch() # 随机种子
    device = torch.device('cuda')
    #dataloader
    test_dataloader = test_dataloader_01(data_dir=args.input_dir,
                            pred_min=args.pred_min,pred_max=args.pred_max,df_dir=args.dev_dir,batch_size=1)
    #model的列表
    model1 = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b1').eval().to(device)
    model2 = models.resnet50(pretrained=True).eval().to(device)
    model3 = models.densenet121(pretrained=True).eval().to(device)
    model4 = models.inception_v3(pretrained=True).eval().to(device)
    model5 = models.vgg16(pretrained=True).eval().to(device)
    model_dict={'b1':model1,'res50':model2,'des121':model3,'inc3':model4,'vgg':model5}
    mean = torch.tensor(args.mean, device=device)
    std = torch.tensor(args.std, device=device)

    if not args.update_mask_csv:
        #如果是第一次全新的开始
        with open(file=args.sdcp,mode='w') as f:
            print('ImageId,Ttogap,Uuogap,TmaskId,Tpred,Tpredori,'
                  'Ttrans,TargetClass,TrueLabel,TmaskClass,UmaskId,UmaskClass,Upred,Upredori,Utrans,UpredClass',file=f)

    for sample in tqdm(test_dataloader):
        img_name = sample['img_name'] #1216张图的名字
        data = sample['image'].to(device) #input的数据为[0-1]
        img_blur = sample['img_blur'].to(device) #经过搞死铝箔得到的图片,但是实际上没有更好,所以kenerl设为0
        tar_label = sample['tar_label'].to(device)-1 #由于dev_csv中label是多了1的所以这里减1
        true_label = sample['label'].to(device)-1
        with torch.no_grad():
            yes,T_update,U_update = PGD_white(input=data,models=model_dict,true_label=true_label,
            tar_lab=tar_label,mean=mean,std=std,img_name=img_name,yes=yes,img_blur=img_blur,T_update=T_update,U_update=U_update)

if __name__=='__main__':
    main(args)
