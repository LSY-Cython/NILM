import matplotlib.pyplot as plt

# 绘制正负事件边缘
def plot_edge(num,posData,posIndex,negData,negIndex):
    plt.xlim(0,num)
    plt.plot(range(num+1),[0]*(num+1),color="black",linewidth=1)  # 显示X轴
    # 通过颜色控制绘制空心散点图
    plt.scatter(posIndex,posData,color="white",marker='o',edgecolors="red",s=40)
    plt.scatter(negIndex,negData,color="white",marker='o',edgecolors="blue",s=40)
    for i in range(len(posIndex)):
        posx = [posIndex[i],posIndex[i]]
        posy = [0,posData[i]]
        plt.plot(posx,posy,color="red",linewidth=1)
    for i in range(len(negIndex)):
        negx = [negIndex[i],negIndex[i]]
        negy = [0,negData[i]]
        plt.plot(negx,negy,color="blue",linewidth=1)
    plt.show()
# 绘制电器分项耗电量饼图
def plot_power(appPower,dbPower):
    appName = list(appPower.keys())
    appCount = list(appPower.values())
    dbName = list(appPower.keys())
    dbCount = list(dbPower.values())
    fig = plt.figure()
    ax1 = fig.add_axes([0,.3,.5,.5],aspect=1)
    ax2 = fig.add_axes([.5,.3,.5,.5],aspect=1)
    fig.suptitle('Total energy consumption', fontsize=14)
    ax1.pie(appCount,autopct='%1.1f%%',startangle=90)
    ax2.pie(dbCount,autopct='%1.1f%%',startangle=90)
    ax1.legend(appName,loc='lower center',bbox_to_anchor=(.5, -.4),fontsize=8)
    ax2.legend(dbName,loc='lower center',bbox_to_anchor=(.5, -.4),fontsize=8)
    ax1.set_title('Disaggregated')
    ax2.set_title('Ground Truth')
    ax1.axis('equal')
    ax2.axis('equal')
    plt.tight_layout()
    plt.savefig("powerpie.png")
    plt.show()