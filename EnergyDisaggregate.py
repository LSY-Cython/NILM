import os
import re
from preprocess import *
from match import *
from reddprocess import event_label
from plot import *
from configparser import ConfigParser
# 总表信号预处理
def signal_preprocess(aggrFile,Lm,La,Ta):
    aggrPower = pd.read_csv(aggrFile,header='infer')["aggregate"].values.tolist()
    medPower = median_filter(aggrPower,Lm)
    averPower = average_filter(medPower,La,Ta)
    return averPower
# def signal_preprocess(aggrFile,Lm,Lg,Tg,sigma,alpha):
#     aggrPower = pd.read_csv(aggrFile, header='infer')["aggregate"].values.tolist()
#     medPower = median_filter(aggrPower,Lm)
#     gbfPower = gb_filter(medPower,Lg,Tg,sigma,alpha)
#     return gbfPower
# 事件初始检测,Te较小,避免遗漏非瞬时启停电器中的小边沿
def edge_detection(data,Tei):
    deltaP = np.array(data[1:])-np.array(data[0:-1])  # 聚合功率一阶差分
    # eIndex = np.where(np.abs(deltaP)>Te)[0]  # 边沿事件时间索引
    # return deltaP,eIndex
    posIndex = np.where(deltaP>Tei)[0]  # 上升沿(正事件)时间索引
    negIndex = np.where(deltaP<-Tei)[0]  # 下降沿(负事件)时间索引
    posData = deltaP[posIndex]
    negData = deltaP[negIndex]
    return deltaP,posData,negData,posIndex,negIndex
# 事件重新筛查,Te较大,去除背景负荷功率波动
def edge_refine(eData,eIndex,Ter):
    if type(eData) is list:
        eData = np.array(eData)
    if type(eIndex) is list:
        eIndex = np.array(eIndex)
    delIndex = np.where(np.abs(eData)<Ter)[0]
    # 删除数组中指定索引元素
    outData = np.delete(eData,delIndex)
    outIndex = np.delete(eIndex,delIndex)
    return outData,outIndex
# 生成电器事件功率时间序列
def generate_power_series(eMatch,aggrTime):
    if type(aggrTime) is not list:
        aggrTime = aggrTime.tolist()
    powerSeries = list()
    for pair in eMatch:
        # 一维线性插值
        startIndex = aggrTime.index(pair[0][0])
        endIndex = aggrTime.index(pair[1][0])
        x = list(range(startIndex+1,endIndex,1))  # 待插值的横坐标
        xp = [startIndex,endIndex]  # 原始数据点横坐标
        fp = [pair[0][1],abs(pair[1][1])]  # 原始数据点纵坐标
        y = np.interp(x,xp,fp).tolist()
        powerSeries.append([fp[0]]+y+[fp[1]])
    return powerSeries
# 读取电器标签数据库
def read_database(dir,aggrFile):
    channelId = [5,8,9,11,12,15,17,18]
    appName = ["refrigerator","kitchen_outlets1","lighting1","microwave","bathroom_gfi","kitchen_outlets2","lighting2","lighting3"]
    dbFiles = [i for i in os.listdir(dir) if "signature_database" in i]
    appFiles = [i for i in os.listdir(dir) if "channel" in i]
    dbSeries = dict()
    dbMean = dict()
    dbPower = dict()
    for file in dbFiles:
        id = int(file.rstrip(".csv").split("_")[-1])
        path = os.path.join(dir,file)
        name = appName[channelId.index(id)]
        appData = pd.read_csv(path,header='infer')[name].values.tolist()
        dbSeries[name] = appData
        dbMean[name] = np.mean(appData)
    for file in appFiles:
        id = int(file.rstrip(".csv").split("_")[-1])
        if id in channelId:
            path = os.path.join(dir,file)
            appAggr = pd.read_csv(path,header='infer')["aggregate"].values
            appTime = pd.read_csv(path,header='infer')["time"].values.tolist()
            appTime = [date2stamp(t) for t in appTime]
            # 电器功率采样率为3-4s/p
            appInterval = (np.array(appTime[1:])-np.array(appTime[0:-1])).tolist()
            appPower = sum([appAggr[i]*appInterval[i] for i in range(len(appInterval))])+appAggr[-1]
            name = appName[channelId.index(id)]
            dbPower[name] = appPower
    aggrPower = pd.read_csv(aggrFile,header='infer')["aggregate"].values
    aggrTime = pd.read_csv(aggrFile,header='infer')["time"].values.tolist()
    aggrTime = [date2stamp(t) for t in aggrTime]
    aggrInterval = (np.array(aggrTime[1:])-np.array(aggrTime[0:-1])).tolist()
    # 总耗电量
    tolPower = sum([aggrPower[i]*aggrInterval[i] for i in range(len(aggrInterval))])+aggrPower[-1]
    dbPower["unknown"] = tolPower-sum(list(dbPower.values()))
    # refrigerator和lighting2起始负事件无法检测
    dbPower["refrigerator"] -= 190*820
    dbPower["lighting2"] -= 64*27306
    dbPower["unknown"] += 190*820+64*27306
    return dbSeries,dbMean,dbPower,tolPower
# 事件打标,分项计量电器耗电量
def event_applicance_label(eMatch,ePowerSeries,dbMean,tolPower):
    """
    1)事件功率均值相近
    2)功率序列DTW距离相近
    """
    appName = list(dbMean.keys())
    appMean = np.array(list(dbMean.values()))
    appPower = dict()
    for name in appName:
        appPower[name]=0
    print("------事件匹配与标注结果------: ")
    for i in range(len(eMatch)):
        # ePower = np.mean([abs(i[1]) for i in eMatch[i]])
        ePower = np.max([abs(i[1]) for i in eMatch[i]])
        appId = np.abs(appMean-ePower).argmin()
        appPower[appName[appId]] += np.sum(ePowerSeries[i])
        print(eMatch[i],ePower,appName[appId],appMean[appId])
    appPower["unknown"] = tolPower-sum(list(appPower.values()))
    return appPower
# 生成单电器事件标签
def app_label():
    appFiles = os.listdir("dataset")
    for file in appFiles:
        pattern = re.match(r"house_(.*?)_channel_(.*?).csv", file)
        if pattern:
            houseId = int(pattern.group(1))
            channelId = int(pattern.group(2))
            aggrFile = os.path.join("dataset", file)
            aggrTime = pd.read_csv(aggrFile,header='infer')["time"].values.tolist()
            aggrPower = pd.read_csv(aggrFile,header='infer')["aggregate"].values.tolist()
            deltaP,posData,negData,posIndex,negIndex = edge_detection(aggrPower,30)
            event_label(aggrTime,posData,posIndex,negData,negIndex,houseId,channelId)
# 总耗电量分解
def energy_disaggregate(configPath):
    # 0)读取配置参数,字符串格式
    parse = ConfigParser()
    parse.read(configPath,encoding="utf-8")
    Lm = int(parse.get("default","Lm"))
    La = int(parse.get("default","La"))
    Ta = float(parse.get("default","Ta"))
    Tei = float(parse.get("default","Tei"))
    Ter = float(parse.get("default","Ter"))
    dampling = float(parse.get("default","dampling"))
    k = float(parse.get("default","k"))
    aggrFile = parse.get("default","aggrFile")
    # 2)总功率信号预处理
    aggrPower = signal_preprocess(aggrFile,Lm,La,Ta)
    aggrTime = pd.read_csv(aggrFile, header='infer')["time"].values.tolist()
    # 3)初始事件检测
    deltaP,posData,negData,posIndex,negIndex = edge_detection(aggrPower,Tei)
    print("------初始正事件检测结果------：", "\n",posData,"\n",posIndex)
    print("------初始负事件检测结果------：", "\n",negData,"\n",negIndex)
    # 4)连续事件边缘合并
    # posData,posIndex,posMergeData,posMergeIndex = edge_sharpening(posData,posIndex)
    # negData,negIndex,negMergeData,negMergeIndex = edge_sharpening(negData,negIndex)
    events = list(zip(posIndex,posData))+list(zip(negIndex,negData))
    events.sort(key=lambda x: x[0],reverse=False)
    eData = [e[1] for e in events]
    eIndex = [e[0] for e in events]
    eData,eIndex,eMergeData,eMergeIndex = edge_sharpening(eData,eIndex)
    posData = eData[np.where(eData>0)[0]]
    posIndex = eIndex[np.where(eData>0)[0]]
    negData = eData[np.where(eData<0)[0]]
    negIndex = eIndex[np.where(eData<0)[0]]
    posMergeIndex = [i for i in eMergeIndex if sum(eMergeData[eMergeIndex.index(i)])>0]
    negMergeIndex = [i for i in eMergeIndex if sum(eMergeData[eMergeIndex.index(i)])<0]
    print("------事件边缘锐化结果------:","\n",eMergeData,"\n",eMergeIndex)
    print("------边缘锐化后正事件合并结果------：","\n",posData,"\n",posIndex,"\n",posMergeIndex)
    print("------边缘锐化后负事件合并结果------：","\n",negData,"\n",negIndex,"\n",negMergeIndex)
    # 5)事件重新检测
    posData,posIndex = edge_refine(posData,posIndex,Ter)
    negData,negIndex = edge_refine(negData,negIndex,Ter)
    print("------正事件重新检测结果------：", "\n",posData,"\n",posIndex)
    print("------负事件重新检测结果------：", "\n",negData,"\n",negIndex)
    # 6)开关二状态电器事件就近匹配
    closeMatch,posData,posIndex,negData,negIndex = close_match(aggrPower,aggrTime,posData,posIndex,negData,negIndex)
    print("------开关二状态电器事件就近匹配结果------: ")
    [print(c) for c in closeMatch]
    print("------开关二状态电器就近匹配后正事件检测结果------：","\n",list(zip(np.array(aggrTime)[posIndex+1].tolist(),posData)))
    print("------开关二状态电器就近匹配后负事件检测结果------：","\n",list(zip(np.array(aggrTime)[negIndex+1].tolist(),negData)))
    # 7)多档位电器事件就近匹配
    mulMatch,posData,posIndex,negData,negIndex = multiple_match(aggrPower,aggrTime,posData,posIndex,negData,negIndex)
    print("------多档位电器事件就近匹配结果------: ")
    [print(m) for m in mulMatch]
    print("------多档位电器就近匹配后正事件检测结果------：","\n",list(zip(np.array(aggrTime)[posIndex+1].tolist(),posData)))
    print("------多档位电器就近匹配后负事件检测结果------：","\n",list(zip(np.array(aggrTime)[negIndex+1].tolist(),negData)))
    # 8)事件聚类匹配(电器功率大于100w)
    cluMatch,posData,posIndex,negData,negIndex = cluster_match(posData,posIndex,negData,negIndex,dampling,k,aggrTime)
    # print("------事件聚类匹配结果------：", )
    # [print(c) for c in cluMatch]
    print("------聚类匹配后正事件检测结果------：","\n",list(zip(np.array(aggrTime)[np.array(posIndex)+1].tolist(),posData)))
    print("------聚类匹配后负事件检测结果------：","\n",list(zip(np.array(aggrTime)[np.array(negIndex)+1].tolist(),negData)))
    # 8)生成已匹配事件功率估值序列
    haveMatch = closeMatch+mulMatch+cluMatch
    powerSeries,havePowerSeries,posMergeIndex,negMergeIndex = estimate_powerseries(len(aggrPower),haveMatch,deltaP,posMergeIndex,negMergeIndex,aggrTime)
    if type(aggrPower) is list:
        aggrPower = np.array(aggrPower)
    objPower = aggrPower-np.min(aggrPower)-powerSeries
    # 9)优化目标功率序列平滑预处理
    objPower = obj_process(objPower,haveMatch,aggrTime)
    print("------待优化目标功率序列------:")
    plt.plot(objPower,color="blue",label="objPower")
    # plt.plot(powerSeries,color="red",label="sumPower")
    plt.legend()
    plt.savefig("obj.png")
    plt.show()
    # 9)事件组合优化匹配
    optMatch,optPowerSeries = optimize_match(posData,posIndex,negData,negIndex,posMergeIndex,negMergeIndex,len(aggrPower),objPower,deltaP,aggrTime)
    # 10)分项计量结果统计
    eMatch = haveMatch+optMatch
    print("------事件匹配结果------: ")
    [print(e) for e in eMatch]
    ePowerSeries = havePowerSeries+optPowerSeries
    dbSeries,dbMean,dbPower,tolPower = read_database("dataset/house1",aggrFile)
    appPower = event_applicance_label(eMatch,ePowerSeries,dbMean,tolPower)
    print("------电器分项计量耗电量------:","\n",appPower)
    print("------电器实际测量耗电量------:","\n",dbPower)
    print("------总表实测耗电量------:", "\n",tolPower)
    plot_power(appPower,dbPower)
if __name__ == "__main__":
    energy_disaggregate("config.ini")