# 负荷事件匹配规则
import numpy as np
from itertools import combinations
import cvxpy as cp
from clustering import *
# 开关二状态电器,正负短时间负荷事件就近匹配
def close_match(p,t,posData,posIndex,negData,negIndex):
    """
    1)正负事件相邻
    2)满足零环和约束
    3)正事件的事前功率和负事件的事后功率相等
    4)阈值比例系数考虑启停事件功率差异
    """
    closeMatch = list()
    posMatch = list()
    negMatch = list()
    for i in range(len(posIndex)):
        posId = posIndex[i]
        posPower = posData[i]  # 正事件功率
        if i!=len(posIndex)-1:
            negLoc = np.where(((negIndex>posId)&(negIndex<posIndex[i+1])))[0]
        else:
            negLoc = np.where(negIndex>posId)[0]
        # if len(negLoc)==0:
        if len(negLoc)!=1:
            continue
        else:
            negLoc = int(negLoc[0])
        negId = negIndex[negLoc]
        negPower = negData[negLoc]  # 负事件功率
        zsumCons = abs(posPower+negPower)<0.35*posPower
        epowerCons = False
        for j in range(1,6,1):
            befPower = p[posId-j]  # 正事件的事前功率
            aftPower = p[negId+j]  # 负事件的事后功率
            if abs(befPower-aftPower)<50:
                epowerCons = True
                break
        if zsumCons and epowerCons:
            posTime = t[posId+1]
            negTime = t[negId+1]
            closeMatch.append([(posTime,posPower),(negTime,negPower)])
            posMatch.append(i)
            negMatch.append(negLoc)
    posData = np.delete(posData,posMatch)
    posIndex = np.delete(posIndex,posMatch)
    negData = np.delete(negData,negMatch)
    negIndex = np.delete(negIndex,negMatch)
    return closeMatch,posData,posIndex,negData,negIndex
# 多档位电器就近匹配
def multiple_match(p,t,posData,posIndex,negData,negIndex):
    """
    1)事件相邻: 停机-一档-二档-停机,停机-二档-一档-停机
    2)满足零环和约束
    3)事前稳态功率和事后稳态功率相等
    """
    mulMatch = list()
    posMatch = list()
    negMatch = list()
    for i in range(len(posIndex)-1):
        posId0 = posIndex[i]
        posVal0 = posData[i]
        posId1 = posIndex[i+1]
        posVal1 = posData[i+1]
        # 高档位与低档位间的功率差异不应过大
        if max(posVal0,posVal1)-min(posVal0,posVal1)>100:
            continue
        negLoc = np.where(negIndex>posId1)[0][0]
        negId = negIndex[negLoc]
        negVal = negData[negLoc]
        zsumCons = abs(posVal0+posVal1+negVal)<0.2*abs(negVal)
        if not zsumCons:
            continue
        epowerCons = False
        for j in range(1,6,1):
            befPower = p[posId0-j]  # 正事件的事前功率
            aftPower = p[negId+j]  # 负事件的事后功率
            if abs(befPower-aftPower)<50:
                epowerCons = True
                break
        if epowerCons:
            posTime0 = t[posId0+1]
            posTime1 = t[posId1+1]
            negTime = t[negId+1]
            mulMatch.append([(posTime0,posVal0),(posTime1,posVal1),(negTime,negVal)])
            posMatch.append(i)
            posMatch.append(i+1)
            negMatch.append(negLoc)
    posData = np.delete(posData,posMatch)
    posIndex = np.delete(posIndex,posMatch)
    negData = np.delete(negData,negMatch)
    negIndex = np.delete(negIndex,negMatch)
    return mulMatch,posData,posIndex,negData,negIndex
# 无监督聚类事件匹配
def cluster_match(posData,posIndex,negData,negIndex,dampling,k,aggrTime):
    if type(posIndex) is list:
        posIndex = np.array(posIndex)
    if type(negIndex) is list:
        negIndex = np.array(negIndex)
    # 待匹配事件序列从正事件开始,到负事件结束
    negStart = np.where(negIndex<posIndex[0])
    posEnd = np.where(posIndex>negIndex[-1])
    posData = np.delete(posData,posEnd)
    posIndex = np.delete(posIndex,posEnd)
    negData = np.delete(negData,negStart)
    negIndex = np.delete(negIndex,negStart)
    # 挑选事件功率大于100w的电器进行聚类匹配
    posLoc = np.where(posData>100)[0]
    negLoc = np.where(np.abs(negData)>100)[0]
    posLocEx = np.where(posData<=100)[0]
    negLocEx = np.where(np.abs(negData)<=100)[0]
    posDataEx = np.array(posData)[posLocEx].tolist()
    posIndexEx = np.array(posIndex)[posLocEx].tolist()
    negDataEx = np.array(negData)[negLocEx].tolist()
    negIndexEx = np.array(negIndex)[negLocEx].tolist()
    posData = posData[posLoc].tolist()
    posIndex = posIndex[posLoc].tolist()
    negData = negData[negLoc].tolist()
    negIndex = negIndex[negLoc].tolist()
    # AP初始聚类
    posCluData,posCluIndex = ap_clustering(posData,posIndex,dampling)
    negCluData,negCluIndex = ap_clustering(negData,negIndex,dampling)
    posCluRsd,posCluMean = rsd_compute(posCluData)
    negCluRsd,negCluMean = rsd_compute(negCluData)
    # 对不合格的类簇进行AP重新聚类
    while (np.array(posCluRsd)>k).any():
        posCluData,posCluIndex = ap_refine(posCluData,posCluIndex,dampling,posCluRsd,k)
        posCluRsd,posCluMean = rsd_compute(posCluData)
    while (np.array(negCluRsd)>k).any():
        negCluData,negCluIndex = ap_refine(negCluData,negCluIndex,dampling,negCluRsd,k)
        negCluRsd,negCluMean = rsd_compute(negCluData)
    print("------正事件聚类结果------: ", "\n",posCluData,"\n",posCluIndex,"\n",posCluRsd,"\n",posCluMean)
    print("------负事件聚类结果------: ", "\n",negCluData,"\n",negCluIndex,"\n",negCluRsd,"\n",negCluMean)
    # 若类簇数目较多,作KMeans重新聚类后合并均值相近的类簇,确保正负类簇数量相等
    posCluNum,negCluNum = len(posCluMean),len(negCluMean)
    if posCluNum!=negCluNum:
        if posCluNum>negCluNum:
            cluNum = negCluNum
            posCluData,posCluIndex,posCluRsd,posCluMean = kmeans_clustering(cluNum,posData,posIndex)
            # posCluData,posCluIndex,posCluRsd,posCluMean = kmeans_clustering(cluNum,posCluMean,posCluData,posCluIndex)
        else:
            cluNum = posCluNum
            negCluData,negCluIndex,negCluRsd,negCluMean = kmeans_clustering(cluNum,negData,negIndex)
            # negCluData,negCluIndex,negCluRsd,negCluMean = kmeans_clustering(cluNum,negCluMean,negCluData,negCluIndex)
    posCluData,posCluIndex,posCluRsd,posCluMean = cluster_sort(posCluData,posCluIndex,posCluRsd,posCluMean)
    negCluData,negCluIndex,negCluRsd,negCluMean = cluster_sort(negCluData,negCluIndex,negCluRsd,negCluMean)
    print("------正类簇合并结果------: ", "\n",posCluData,"\n",posCluIndex,"\n",posCluRsd,"\n",posCluMean)
    print("------负类簇合并结果------: ", "\n",negCluData,"\n",negCluIndex,"\n",negCluRsd,"\n",negCluMean)
    # 逐个正类簇分层匹配,候选负事件满足时序夹在两相邻正事件间且事件功率相近,未匹配的事件按序归并入下一个类簇
    # 当一个正事件有多个候选负事件时,应继续事件优化匹配[xxx选择事件功率最接近的xxx]
    posCluData.append([])
    posCluIndex.append([])
    negCluData.append([])
    negCluIndex.append([])
    cluPair = list()
    for i in range(len(posCluIndex)-1):
        tempSet = list(zip(posCluIndex[i]+negCluIndex[i],posCluData[i]+negCluData[i]))
        tempSet.sort(key=lambda x:x[0])
        pairStack = list()
        for e in tempSet:
            if len(pairStack)==0 and e[1]>0:
                pairStack.append(e)
            elif len(pairStack)==0 and e[1]<0:
                continue
            elif pairStack[-1][1]>0 and e[1]>0:
                pairStack = [e]
            elif pairStack[-1][1]<0 and e[1]>0:
                if len(pairStack)>2:
                    # tempNegPower = [c[1] for c in pairStack[1:]]
                    # tempNegId = np.abs(np.array(tempNegPower)+pairStack[0][1]).argmin()+1
                    # pairStack = [pairStack[0],pairStack[tempNegId]]
                    pairStack = [e]
                    continue
                if len(pairStack)==2 and abs(pairStack[0][1]+pairStack[1][1])<0.35*pairStack[0][1]:
                    posTime = aggrTime[pairStack[0][0]+1]
                    negTime = aggrTime[pairStack[1][0]+1]
                    posPower = pairStack[0][1]
                    negPower = pairStack[1][1]
                    cluPair.append([(posTime,posPower),(negTime,negPower)])
                    posCluIndex[i].remove(pairStack[0][0])
                    posCluData[i].remove(pairStack[0][1])
                    negCluIndex[i].remove(pairStack[1][0])
                    negCluData[i].remove(pairStack[1][1])
                    pairStack = [e]
            elif pairStack[-1][1]>0 and e[1]<0:
                pairStack.append(e)
            elif pairStack[-1][1]<0 and e[1]<0:
                pairStack.append(e)
        if len(pairStack)==2 and abs(pairStack[0][1]+pairStack[1][1])<0.35*pairStack[0][1]:
            posTime = aggrTime[pairStack[0][0]+1]
            negTime = aggrTime[pairStack[1][0]+1]
            posPower = pairStack[0][1]
            negPower = pairStack[1][1]
            cluPair.append([(posTime,posPower),(negTime,negPower)])
            posCluIndex[i].remove(pairStack[0][0])
            posCluData[i].remove(pairStack[0][1])
            negCluIndex[i].remove(pairStack[1][0])
            negCluData[i].remove(pairStack[1][1])
        posCluIndex[i+1].extend(posCluIndex[i])
        posCluData[i+1].extend(posCluData[i])
        negCluIndex[i+1].extend(negCluIndex[i])
        negCluData[i+1].extend(negCluData[i])
    posData = posCluData[-1]+posDataEx
    posIndex = posCluIndex[-1]+posIndexEx
    negData = negCluData[-1]+negDataEx
    negIndex = negCluIndex[-1]+negIndexEx
    return cluPair,posData,posIndex,negData,negIndex
# 负荷事件序列集最优组合匹配
def optimize_match(posData,posIndex,negData,negIndex,posMergeIndex,negMergeIndex,tolNum,objPower,deltaP,aggrTime):
    if type(posIndex) is list:
        posIndex = np.array(posIndex)
    if type(negIndex) is list:
        negIndex = np.array(negIndex)
    # 待匹配事件序列从正事件开始,到负事件结束
    negStart = np.where(negIndex<min(posIndex))
    posEnd = np.where(posIndex>max(negIndex))
    posData = np.delete(posData,posEnd)
    posIndex = np.delete(posIndex,posEnd)
    negData = np.delete(negData,negStart)
    negIndex = np.delete(negIndex,negStart)
    eData = list(zip(posIndex,posData))+list(zip(negIndex,negData))
    eData.sort(key=lambda x:x[0])
    # print("------候选负荷事件集合------: ","\n",eData)
    eSet = list()
    eSetNew = list()
    ePower = list()  # 有功功率估值序列
    # 按时间顺序,生成候选负荷事件序列集合,假设现实生活中家用电器一个工作循环内负荷事件数目一般不超过3
    # 1)一正一负
    for i in range(len(posIndex)):
        negLoc = np.where(negIndex>posIndex[i])[0]
        candIndex = negIndex[negLoc]
        candData = negData[negLoc]
        for j in range(len(candIndex)):
            if abs(posData[i]+candData[j])<0.35*posData[i]:
                e = [(posIndex[i],posData[i]),(candIndex[j],candData[j])]
                eSet.append(e)
                eNew = [[(aggrTime[posIndex[i]+1],posData[i]),(aggrTime[candIndex[j]+1],candData[j])]]
                eSetNew.append(eNew[0])
                ep,_,_,_ = estimate_powerseries(tolNum,eNew,deltaP,posMergeIndex,negMergeIndex,aggrTime)
                ePower.append(ep)
    # # 2)两正一负
    # for comb in combinations(posIndex,2):
    #     negLoc = np.where(negIndex>max(comb[0],comb[1]))[0]
    #     candIndex = negIndex[negLoc]
    #     candData = negData[negLoc]
    #     posId0 = posIndex.tolist().index(comb[0])
    #     posId1 = posIndex.tolist().index(comb[1])
    #     # 高档位与低档位间的功率差异不应过大
    #     if max(posData[posId0],posData[posId1])-min(posData[posId0],posData[posId1])>150:
    #         continue
    #     posSum = posData[posId0]+posData[posId1]
    #     for j in range(len(candIndex)):
    #         if abs(posSum+candData[j])<0.35*posSum and posData[posId0]>50 and posData[posId1]>50:
    #             e = [(posIndex[posId0],posData[posId0]),(posIndex[posId1],posData[posId1]),(candIndex[j],candData[j])]
    #             eSet.append(e)
    #             eNew = [[(aggrTime[posIndex[posId0]+1],posData[posId0]),(aggrTime[posIndex[posId1]+1],posData[posId1]),(aggrTime[candIndex[j]+1],candData[j])]]
    #             eSetNew.append(eNew[0])
    #             ep,_,_,_ = estimate_powerseries(tolNum,eNew,deltaP,posMergeIndex,negMergeIndex,aggrTime)
    #             ePower.append(ep)
    print("------候选负荷事件序列集------: ","\n",eSetNew)
    # 组合优化
    x = optcomb_eset(eSet,eData,ePower,objPower)
    optMatch = list()
    print("------事件组合优化匹配结果------：")
    optPowerSeries = list()
    for i in range(len(x)):
        if int(x[i])==1:
            print(eSetNew[i])
            optMatch.append(eSetNew[i])
            optPowerSeries.append(ePower[i].tolist())
    return optMatch,optPowerSeries
# 负荷事件序列最优组合求解
def optcomb_eset(eSet,eData,ePower,P):
    sNum = len(eSet)  # 候选序列总数
    # Construct a CVXPY problem
    x = cp.Variable(sNum,boolean=True)  # 待求变量β类型：0-1整型变量
    eIndex = [i[0] for i in eData]
    eNum = len(eIndex)  # 事件数目
    S = list()  # 候选0/1矩阵S
    for es in eSet:
        ei = [eIndex.index(i[0]) for i in es]
        rowS = [0]*eNum
        for i in ei:
            rowS[i] = 1
        S.append(rowS)
    S = np.array(S)
    # 约束条件：确保任意负荷事件只能由一个电器(事件序列)产生
    constraints = list()
    for i in range(eNum):
        s = S[:,i].reshape(1,-1)
        # constraints.append((s@x)>=0)
        constraints.append((s@x)<=1)
    if type(ePower) is list:
        ePower = np.array(ePower)
        # ePower = np.array(ePower).T
    if type(P) is list:
        P = np.array(P).reshape(1,-1)
        # P = np.array(P)
    # 目标函数多元泰勒展开,一阶项近似
    w = list()
    for i in range(len(ePower)):
        wi = np.matmul(P,ePower[i].reshape(-1,1))
        w.append(wi)
    w = np.array(w).reshape(1,-1)
    objective = cp.Maximize(w@x)
    # objective = cp.Minimize(cp.pnorm(P-ePower@x,1))
    prob = cp.Problem(objective,constraints)
    prob.solve(solver=cp.MOSEK,verbose=True)
    print("------事件序列组合优化结果------：","\n",x.value,"\n",prob.value)
    return x.value
# 估计已匹配事件集的有功功率时间序列
def estimate_powerseries(tolNum,haveMatch,deltaP,posMergeIndex,negMergeIndex,t):
    powerSeries = np.zeros(tolNum)
    havePowerSeries = list()
    if type(deltaP) is list:
        deltaP = np.array(deltaP)
    for match in haveMatch:
        eventSeries = np.zeros(tolNum)
        if len(match)==3:  # 两正一负
            eventSeries[t.index(match[0][0]):t.index(match[1][0])] = match[0][1]
            match = [(match[1][0],match[0][1]+match[1][1]),match[-1]]
        # 将字符串时间转换为时间索引,tuple只读不写
        if type(match[0][0]) is str:
            posTi = int(t.index(match[0][0]))-1
        else:
            posTi = match[0][0]
        if type(match[1][0]) is str:
            negTi = int(t.index(match[1][0]))-1
        else:
            negTi = match[1][0]
        posEp = match[0][1]
        negEp = abs(match[1][1])
        posMergeFlag = False
        negMergeFlag = False
        for pmi in posMergeIndex:
            if posTi in pmi:
                pmiNew = np.arange(min(pmi),max(pmi)+1,1)  # 连续事件间隔为1或2
                pmp = np.cumsum(deltaP[pmiNew])
                eventSeries[pmiNew+1] = pmp
                posTi = pmiNew[-1]
                posEp = eventSeries[posTi+1]
                posMergeIndex.remove(pmi)
                posMergeFlag = True
                break
        for nmi in negMergeIndex:
            if negTi in nmi:
                nmiNew = np.arange(min(nmi),max(nmi)+1,1)
                nmp = np.flipud(np.cumsum(deltaP[nmiNew][::-1]))  # 数组反转
                eventSeries[nmiNew] = np.abs(nmp)
                negTi = nmiNew[0]
                negEp = eventSeries[negTi]
                negMergeIndex.remove(nmi)
                negMergeFlag = True
                break
        if not posMergeFlag:
            eventSeries[posTi+1] = posEp  # 正事件索引返回原始事件索引时需要加1
        if not negMergeFlag:
            eventSeries[negTi] = negEp  # 负事件索引返回原始事件索引时不需要加1
        x = list(range(posTi+2,negTi,1))  # 待插入数据的横坐标
        xp = [posTi+1,negTi]  # 原始数据点的横坐标
        fp = [posEp,negEp]   # 原始数据点的纵坐标
        y = np.interp(x,xp,fp)  # 插入数据点的纵坐标
        if len(x)!=0:
            eventSeries[np.array(x)] = y
        havePowerSeries.append(eventSeries.tolist())
        powerSeries += eventSeries
    return powerSeries,havePowerSeries,posMergeIndex,negMergeIndex

