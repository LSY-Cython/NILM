import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocess import date2stamp,stamp2date
# 读取指定日期范围内的REDD有功功率数据
def read_reddchannel(file,startDate,endDate):
    startStamp = date2stamp(startDate)
    endStamp = date2stamp(endDate)
    rawData = pd.read_table(file, header=None)[0].values
    utcStamp = [int(data.split(" ")[0]) for data in rawData]
    actPower = [round(float(data.split(" ")[1]),2) for data in rawData]
    scopeStamp = [stamp for stamp in utcStamp if startStamp<=stamp<=endStamp]
    startIndex = utcStamp.index(scopeStamp[0])
    endIndex = utcStamp.index(scopeStamp[-1])
    powerData = actPower[startIndex:endIndex+1]
    scopeTime = [stamp2date(stamp) for stamp in scopeStamp]
    return scopeTime,powerData
# 按天整理REDD标注数据
def process_reddchannel(dir,houseId,startDate,endDate):
    folder = os.path.join(dir,houseId)
    # 总聚合功率(mains1+mains2), 1s/p
    path1 = os.path.join(folder,"channel_1.dat")
    scopeTime,mainPower1 = read_reddchannel(path1,startDate,endDate)
    path2 = os.path.join(folder, "channel_2.dat")
    scopeTime,mainPower2 = read_reddchannel(path2,startDate,endDate)
    mainPower = (np.array(mainPower1)+np.array(mainPower2)).tolist()
    # 字典中的key值即为csv中的列名
    mainFrame = pd.DataFrame({"time":scopeTime, "aggregate":mainPower})
    # 将DataFrame存储为csv,index表示是否显示行名
    mainFrame.to_csv(f"dataset/{houseId}_aggr.csv", index=False, sep=',')
    # 单电器功率, 3s-4s/p
    channelFiles = [file for file in os.listdir(folder) if "channel" in str(file)]
    for file in channelFiles:
        channelId = int(str(file).rstrip(".dat").split("_")[1])
        if channelId != 1 and channelId != 2:
            path = os.path.join(folder,f"channel_{channelId}.dat")
            scopeTime,appPower = read_reddchannel(path,startDate,endDate)
            appFrame = pd.DataFrame({"time":scopeTime, "aggregate":appPower})
            appFrame.to_csv(f"dataset/{houseId}_channel_{channelId}.csv", index=False, sep=',')
# 单电器事件标签
def event_label(t,posData,posIndex,negData,negIndex,houseId,channelId):
    if type(posData) is not list:
        posData = posData.tolist()
    if type(posIndex) is not list:
        posIndex = posIndex.tolist()
    if type(negData) is not list:
        negData = negData.tolist()
    if type(negIndex) is not list:
        negIndex = negIndex.tolist()
    if type(t) is list:
        t = np.array(t)
    eventData = list(zip(posIndex,posData))+list(zip(negIndex,negData))
    if eventData!=[]:
        eventData.sort(key=lambda x:x[0])
        data = [i[1] for i in eventData]
        index = [i[0] for i in eventData]
        eventPower = data  # 事件功率
        eventTime = t[index]  # 事件时刻
    else:
        eventPower = []
        eventTime = []
    eventFrame = pd.DataFrame({"eventTime":eventTime,"eventPower":eventPower})
    eventFrame.to_csv(f"dataset/house_{houseId}_label_{channelId}.csv", index=False, sep=',')
if __name__ == "__main__":
    # dir = "D:/Dataset/NILM/REDD/low_freq"
    # houseId = "house_1"
    # startTime = "2011-4-20 00:00:00"
    # endTime = "2011-4-20 23:59:59"
    # process_reddchannel(dir,houseId,startTime,endTime)
    # aggrFile = "dataset/house_1_aggr.csv"
    aggrFile = "dataset/house_1_channel_5.csv"
    aggrPower = pd.read_csv(aggrFile, header='infer')["aggregate"].values.tolist()
    delta_p = [aggrPower[i+1]-aggrPower[i] for i in range(0,len(aggrPower)-1)]
    plt.plot(aggrPower)
    plt.show()