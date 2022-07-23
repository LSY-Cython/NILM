import numpy as np
import binascii
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal,interpolate
import cv2
import random
import json
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch
import math

def read_data(file):
    raw_data = pd.read_excel(file,sheet_name="data")
    appliance = raw_data["appliance"].tolist()
    feature = raw_data["feature"].tolist()
    # 十六进制字符串转换为2字节有符号整数(int16),字节序为big-endian(后到的在前)
    feature = [np.frombuffer(binascii.unhexlify(s),np.dtype(np.int16).newbyteorder(">"))[-720:] for s in feature]
    app_data = dict()
    for i in range(len(raw_data)):
        app = appliance[i]
        if app not in app_data.keys():
            app_data[app] = list()
        app_data[app].append(feature[i])
    return app_data

def produce_VI(app_data,size):
    app_type = list(app_data.keys())
    print("appliance category：", app_type)
    for app in app_type:
        file_path = f"../dataset/VI/{app}"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        features = app_data[app]
        for i in range(len(features)):
            data = features[i]
            # 一个稳态周期采样36个点
            current = data[0:36*10]
            voltage = data[36*10:]
            # plot_wave(voltage,current,file_path,f"wave_{i}")
            app_v,app_i = app_wave(voltage,current)
            # plot_wave(app_v,app_i,file_path,f"cycle_{i}")
            norm_v = (size-1)*(app_v-np.min(app_v))/(np.max(app_v)-np.min(app_v))
            norm_i = (size-1)*(app_i-np.min(app_i))/(np.max(app_i)-np.min(app_i))
            norm_v = np.asarray(np.round(norm_v,0),dtype=np.int32)
            norm_i = np.asarray(np.round(norm_i,0),dtype=np.int32)
            img_vi = np.zeros((size,size),dtype=np.uint8)
            for j in range(len(norm_v)):
                img_vi[norm_v[j],norm_i[j]] = 255
            cv2.imwrite(os.path.join(file_path,f"VI_{i}.png"),img_vi)

# 以恒压曲线为基准,进行相位对齐后识别电器的电压电流波形
def app_wave(v,i):
    va,vb = v[0:36*5],v[36*5:]
    ia,ib = i[0:36*5],i[36*5:]
    # 求极小值索引
    va_min = signal.argrelextrema(va,np.less,order=36//4)[0][0]
    vb_min = signal.argrelextrema(vb,np.less,order=36//4)[0][0]
    va_align = np.hstack((va[va_min:],va[0:va_min]))
    vb_align = np.hstack((vb[vb_min:],vb[0:vb_min]))
    ia_align = np.hstack((ia[va_min:],ia[0:va_min]))
    ib_align = np.hstack((ib[vb_min:],ib[0:vb_min]))
    app_v = ((va_align+vb_align)/2)[0:36*4]
    # 中值滤波,去除尖峰点
    app_i = signal.medfilt(ib_align-ia_align,kernel_size=3)[0:36*4]
    # 一次线性插值,确保VI轨迹连续
    raw_x = np.linspace(1,36*4,36*4)
    inter_v = interpolate.interp1d(raw_x,app_v,kind='linear')
    inter_i = interpolate.interp1d(raw_x,app_i,kind='linear')
    new_x = np.linspace(1,36*4,36*40)
    app_v,app_i = inter_v(new_x),inter_i(new_x)
    return app_v,app_i

def plot_wave(v,i,file_path,id):
    plt.subplot(2,1,1)
    plt.plot(i,color="blue",label="I")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(v,color="red",label="V")
    plt.legend()
    plt.savefig(os.path.join(file_path,f"{id}.png"))
    plt.clf()

def split_dataset(ratio=0.8):
    app_label = {"cleaneropen": 0,"cleanerclosed": 1,"cookopen": 2,
                 "cookclosed": 3,"dryeropen":4,"dryerclosed":5}
    dataset = {"train":[],"test":[]}
    for app,label in app_label.items():
        folder = f"../dataset/VI/{app}"
        images = [os.path.join(folder,img) for img in os.listdir(folder) if "VI" in img]
        train_imgs = images[0:int(len(images)*ratio)]
        test_imgs = images[int(len(images)*ratio):]
        train_data = [[img,label] for img in train_imgs]
        test_data = [[img,label] for img in test_imgs]
        dataset["train"].extend(train_data)
        dataset["test"].extend(test_data)
        random.shuffle(dataset["train"])
    with open("../dataset/data.json","w") as f:
        json.dump(dataset,f,indent=4)

def add_dataset():
    with open("../dataset/data.json","r") as f:
        dataset = json.load(f)
    app_label = {"batterycharged":6,"batteryclosed":7}
    for app,label in app_label.items():
        folder = f"../dataset/VI/{app}"
        images = [os.path.join(folder,img) for img in os.listdir(folder) if "VI" in img]
        data = [[img,label] for img in images]
        dataset["test"].extend(data)
    with open("../dataset/data.json","w") as f:
        json.dump(dataset,f,indent=4)

class ClassDataset(Dataset):
    def __init__(self, imgData):
        self.imgData = imgData
    def __getitem__(self, index):
        img_path =self.imgData[index][0]
        label = self.imgData[index][1]
        raw_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        norm_Img = transforms.ToTensor()(raw_img)
        return norm_Img, label
    def __len__(self):
        return len(self.imgData)

# if __name__ == "__main__":
#     app_data = read_data("../dataset/db.xls")
#     produce_VI(app_data,size=32)
    # split_dataset()
    # add_dataset()
    # with open("../dataset/data.json", "r") as f:
    #     data = json.load(f)
    # train_set = ClassDataset(imgData=data["train"])
    # train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

import numpy as np

a=np.random.normal()
print(a)
