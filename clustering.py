from sklearn.cluster import AffinityPropagation,KMeans
import numpy as np
from preprocess import gsp_compute
# 近邻传播聚类
def ap_clustering(eData,eIndex,dampling):
    cluIndex,cluData = list(),list()
    if type(eData) is list:
        eData = np.array(eData)
    if type(eIndex) is list:
        eIndex = np.array(eIndex)
    data = eData.reshape(-1,1)
    cluster = AffinityPropagation(damping=dampling).fit(data)
    labelId = cluster.labels_
    labelNum = len(set(labelId))
    for i in range(labelNum):
        tempIndex = np.where(labelId==i)[0]
        cluIndex.append(eIndex[tempIndex].tolist())
        cluData.append(eData[tempIndex].tolist())
    return cluData,cluIndex
# AP重新聚类
def ap_refine(cluData,cluIndex,dampling,rsd,k):
    refData,refIndex = list(),list()
    for i in range(len(rsd)):
        if rsd[i]>k:
            tempData,tempIndex = ap_clustering(cluData[i],cluIndex[i],dampling)
            refData.extend(tempData)
            refIndex.extend(tempIndex)
        else:
            tempData,tempIndex = cluData[i],cluIndex[i]
            refData.append(tempData)
            refIndex.append(tempIndex)
    return refData,refIndex
# 聚类效果评估,计算类簇的相对标准方差
def rsd_compute(cluData):
    cluRsd = list()
    cluMean = list()
    for data in cluData:
        mean = np.mean(data)
        std = np.std(data)
        rsd = np.abs(std/mean)
        cluRsd.append(rsd)
        cluMean.append(mean)
    return cluRsd,cluMean
# 对事件功率[xxxAP各类簇均值xxx]做KMeans重新聚类,保证正负类簇数量相等
# def kmeans_clustering(cluNum,cluMean,cluData,cluIndex):
def kmeans_clustering(cluNum,eData,eIndex):
    cluDataNew,cluIndexNew = list(),list()
    # if type(cluMean) is list:
    #     cluMean = np.array(cluMean)
    # vecMean = cluMean.reshape(-1,1)
    if type(eData) is list:
        eData = np.array(eData)
    if type(eIndex) is list:
        eIndex = np.array(eIndex)
    data = eData.reshape(-1,1)
    kmeans = KMeans(n_clusters=cluNum)
    # kmeans.fit(vecMean)
    kmeans.fit(data)
    labelId = kmeans.labels_
    labelNum = len(set(labelId))
    for i in range(labelNum):
        tempIndex = np.where(labelId==i)[0].tolist()
        # tempCluData,temoCluIndex = list(),list()
        # for j in tempIndex:
        #     tempCluData.extend(cluData[j])
        #     temoCluIndex.extend(cluIndex[j])
        # cluDataNew.append(tempCluData)
        # cluIndexNew.append(temoCluIndex)
        cluIndexNew.append(eIndex[tempIndex].tolist())
        cluDataNew.append(eData[tempIndex].tolist())
    cluRsd,cluMean = rsd_compute(cluDataNew)
    return cluDataNew,cluIndexNew,cluRsd,cluMean
# 将类簇按均值从大到小排序
def cluster_sort(cluData,cluIndex,cluRsd,cluMean):
    if min(cluMean)>0:
        cluMeanSort = sorted(cluMean,reverse=True)
    elif max(cluMean)<0:
        cluMeanSort = sorted(cluMean,reverse=False)
    cluDataSort,cluIndexSort,cluRsdSort = list(),list(),list()
    for i in range(len(cluMeanSort)):
        sortId = cluMean.index(cluMeanSort[i])
        cluDataSort.append(cluData[sortId])
        cluIndexSort.append(cluIndex[sortId])
        cluRsdSort.append(cluRsd[sortId])
    return cluDataSort,cluIndexSort,cluRsdSort,cluMeanSort
# 谱图聚类: 最小化图信号全局平滑度
def gsp_clustering(eData,eIndex,sigma,q):  # eData和eIndex满足双射关系
    cluIndex = list()
    cluData = list()
    eIndex = eIndex.copy()
    while eIndex.any():  # 至少有一个元素非零
        remainIndex = np.where(eIndex!=0)[0]
        sigData = eData[remainIndex]
        A,D = gsp_compute(sigData,sigma)
        L = D-A  # 图拉普拉斯算子
        s1 = 1
        nodeNum = len(remainIndex)
        # 平滑度最优解s*
        s = np.matmul(np.linalg.pinv(L[1:nodeNum,1:nodeNum]),(-s1)*L.T[0,1:nodeNum]).reshape(1,nodeNum-1)[0]
        setIndex = [remainIndex[0]]+remainIndex[np.where(s>q)[0]+1].tolist()
        tempIndex = eIndex[setIndex].tolist()
        tempData = eData[setIndex].tolist()
        eIndex[setIndex] = 0  # 已聚类事件置零
        cluIndex.append(tempIndex)
        cluData.append(tempData)
    return cluData,cluIndex