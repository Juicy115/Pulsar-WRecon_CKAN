import random

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, ConcatDataset
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms


class myDataset(Dataset):

    def __init__(self,data_csv_path):
        # 读取 csv 文件
        #利用pandas读取csv文件
        self.data_info = pd.read_csv(data_csv_path)
        self.data_arr = np.asarray(self.data_info.iloc[0:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[0:, 1])
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        single_data_path = self.data_arr[index]
        # print(single_data_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image = Image.open(single_data_path)
        tensor_image = transform(image)
        # print(tensor_image.shape)

        # train_date_time = np.stack((tensor_image,), axis=0)

        #读data_dm_time数据,并转为tensor
        # train_dm_time = torch.Tensor(train_date_time)

        train_data = tensor_image
        # print(train_data.shape)
        # plt.imshow(train_date_time[0].T)
        # plt.imshow(train_date_time[0].T, cmap='gray')
        # plt.show()
        label = self.label_arr[index]
        return (train_data, label)

    def __len__(self):
        return self.data_len

#sample为待筛选数据集，true_num为筛选正样本个数
def select_sample(sample , true_num , false_num , true_class = 1 , false_class = 0): #n为负样本数

    true_indicate = [i for i, (image, label) in enumerate(sample) if label == true_class]
    false_indicate = [i for i, (image, label) in enumerate(sample) if label == false_class]
    # 随机选取样本
    if(false_num >= len(false_indicate)):
        false_num = len(false_indicate)
    selected_train_indices = random.sample(false_indicate, false_num)
    if (true_num >= len(true_indicate)):
        true_num = len(true_indicate)
    selected_false_indices = random.sample(false_indicate, false_num)
    selected_true_indices = random.sample(true_indicate, true_num)
    indicate_true = Subset(sample, selected_true_indices)
    indicate_false = Subset(sample, selected_false_indices)
    combined_train_dataset = ConcatDataset([indicate_true, indicate_false])
    return combined_train_dataset

if __name__ == '__main__':
    MyTrainDataset = myDataset("./htru_dataset/dataset_1600.csv")
    print(MyTrainDataset.__getitem__(0))
    train_data_size = len(MyTrainDataset)
    print("训练数据集长度为：{}".format(train_data_size))
    # f = h5py.File("/home/hanli/nfs/wcq/crafts/FP20180213_0-1GHz_Dec+41.1_drifting1_990/./cand_tstart_58162.645185185182_tcand_312.8760000_dm_26.07850_snr_6.09356.h5", 'r')
    # train_date_time = np.array(f['data_dm_time'])
    # # train_dm_time = torch.Tensor(train_date_time)
    # print(train_date_time)
