from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
# 字符串类型的日期转换为时间戳
def date2stamp(datetime):
    timeArray = time.strptime(datetime, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp
# 时间戳转换为指定格式的日期
def stamp2date(timestamp):
    timeArray = time.localtime(timestamp)
    timeStyle = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return timeStyle
# 中值滤波,去除瞬态尖峰
def median_filter(data,Lm):
    return signal.medfilt(data,Lm).tolist()
# 构建图信号,计算加权邻接矩阵A和对角对阵D
def gsp_compute(data,sigma):
    if type(data) is list:
        vec_x = np.array(data)
    else:
        vec_x = data
    node_num = len(data)  # 图节点数
    vec_x = vec_x.reshape((1, node_num))
    mat_i = np.repeat(vec_x.T, repeats=node_num, axis=1)  # 按列重复
    mat_j = np.repeat(vec_x, repeats=node_num, axis=0)  # 按行重复
    adjacency_mat = np.exp(-np.square(mat_i-mat_j)/sigma ** 2)  # 加权邻接矩阵A
    diag_mat = np.diag(np.sum(adjacency_mat, axis=1))  # 对角矩阵D
    return adjacency_mat,diag_mat
# 图双边滤波,平滑信号波动
def gb_filter(data,Lg,Tg,sigma,alpha):
    output = copy.deepcopy(data)
    for i in range(0,len(data)-Lg+1,1):
        x = data[i:i+Lg]
        deltaX = np.abs(np.array(x[1:]-np.array(x[0:-1])))
        if np.max(deltaX)<Tg:  # 保留事件边缘,高斯模糊
            A,D = gsp_compute(x,sigma)
            nodeNum = A.shape[0]
            I = np.identity(nodeNum)  # 构建单位矩阵
            Op = I-np.matmul(np.linalg.inv(D),A)  # BF算子
            xs = np.matmul(np.linalg.inv((I+alpha*np.matmul(Op,Op))),np.array(x)).tolist()
            output[int(i+Lg//2)] = xs[int(Lg//2)]
    return output
# 移动平均滤波
def average_filter(data,La,Ta):
    output = copy.deepcopy(data)
    for i in range(0,len(data)-La+1,1):
        x = data[i:i+La]
        deltaX = np.abs(np.array(x[1:]-np.array(x[0:-1])))
        if np.max(deltaX)<Ta:  # 保留事件边缘,均值模糊
            output[int(i+La//2)] = np.mean(x)
    return output
# 边缘锐化,合并连续的瞬时启停过程
def edge_sharpening(data,index):
    if type(data) is list:
        data = np.array(data)
    if type(index) is list:
        index = np.array(index)
    dataNew = list()
    indexNew = list()
    indexStack = list()  # 存储连续事件序列的堆栈
    mergeIndex = list()
    mergeData = list()
    for i in range(len(index)):
        if len(indexStack)==0:
            indexStack.append(i)
        else:
            if index[i]-index[indexStack[-1]]<=2:
                indexStack.append(i)
            else:
                mergeValue = np.sum(data[indexStack])
                dataNew.append(mergeValue)
                indexNew.append(index[indexStack[-1]])  # 合并取连续事件的末尾节点
                if len(indexStack)>1:
                    mergeIndex.append(index[indexStack].tolist())
                    mergeData.append(data[indexStack].tolist())
                indexStack = [i]
    if len(indexStack)!=0:
        dataNew.append(np.sum(data[indexStack]))
        indexNew.append(index[indexStack[-1]])
        if len(indexStack)>1:
            mergeIndex.append(index[indexStack].tolist())
            mergeData.append(data[indexStack].tolist())
    return np.array(dataNew),np.array(indexNew),mergeData,mergeIndex
# 平滑处理待优化的目标功率序列P
def obj_process(objPower,haveMatch,t):
    for match in haveMatch:
        startId = t.index(match[0][0])
        endId = t.index(match[-1][0])
        befPower = objPower[startId-12:startId-2]  # 事前稳态功率
        aftPower = objPower[endId+2:endId+12]  # 事后稳态功率
        smPower = np.mean(np.concatenate((befPower,aftPower),axis=0),axis=0)  # 平均稳态功率
        objPower[startId-2:endId+3] = smPower
    objPower = median_filter(objPower,5)
    objPower = np.array(average_filter(objPower,7,30))
    objPower[0:850] = objPower[900]
    return objPower
# 动态时间规整,度量两段不等长时间序列间的相似性,O(N*N)
def dtw(s1,s2):
    rows = len(s1)+1
    cols = len(s2)+1
    costMat = np.zeros((rows,cols))
    costMat[0,:] = float('inf')
    costMat[:,0] = float('inf')
    # 逐行向下遍历
    for i in range(1,rows,1):
        for j in range(1,cols,1):
            if i==j==1:  # 起点初始化
                costMat[1,1] = abs(s1[0]-s2[0])
            else:  # 时序对齐
                costMat[i,j] = abs(s1[i-1]-s2[j-1])+min(costMat[i-1,j-1],costMat[i-1,j],costMat[i,j-1])  # 曼哈顿距离
    return costMat[rows-1,cols-1]
# 基功率包络线检测与去除
def basepower_remove(data,Tb):
    if type(data) is not list:
        data = data.tolist()
    minData = min(data)
    minIndex = data.index(minData)
    p1 = list(reversed(data[0:minIndex]))
    p2 = data[minIndex+1:]
    b1,b2 = [minData],[minData]
    b1 = list(reversed(basepower_detect(p1,b1,Tb)))
    b2 = basepower_detect(p2,b2,Tb)
    b = b1+b2[1:]
    p = (np.array(data)-np.array(b)).tolist()
    return b,p
def basepower_detect(p,b,Tb):
    for i in range(len(p)):
        if abs(p[i]-b[-1])<Tb:
            b.append(p[i])
        elif b[-1]<p[i]:
            b.append(b[-1])
        else:
            b.append(p[i])
    return b

# import time
# aggrFile = "dataset/house1/house_1_aggr.csv"
# # aggrFile = "dataset/house1/house_1_channel_8.csv"
# # aggrFile = "GSP_energy_disaggregator-master/output_aggr.csv"
# aggrPower = pd.read_csv(aggrFile, header='infer')["aggregate"].values.tolist()
# ms = time.time()
# medPower = median_filter(aggrPower,5)
# me = time.time()
# print(f"中值滤波耗时：{me-ms}s")
# # gbs = time.time()
# # gbfPower = gb_filter(medPower,11,50,30,1)
# # gbe = time.time()
# # print(f"图双边滤波耗时：{gbe-gbs}s")
# avers = time.time()
# averPower = average_filter(medPower,9,20)
# avere = time.time()
# print(f"移动平均滤波耗时：{avere-avers}s")
# # basePower, aggrPower = basepower_remove(averPower,50)
# # deltaPower = [gbfPower[i+1]-gbfPower[i] for i in range(0,len(gbfPower)-1)]
# deltaPower = [averPower[i+1]-averPower[i] for i in range(0,len(averPower)-1)]
# plt.plot(aggrPower,label="Aggregation")
# # plt.plot(medPower,label="Median")
# # plt.plot(gbfPower,label="Bilateral")
# plt.plot(averPower,label="Average")
# # plt.plot(basePower,label="Base")
# # plt.plot(deltaPower,label="Delta")
# plt.legend()
# plt.savefig("sig.png")
# plt.show()