#无监督学习  K-means 聚类

import numpy as np

#加载数据集
def loadDataSet(fileName):
    dataMat=[]
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        fltLine = list(map(float, curLine)) #逐个调用float()函数，将字符串转为浮点数
        dataMat.append(fltLine)
    return dataMat

#计算两点 欧式距离
def distEclud(vecA,vecB):
    diffAB=vecA-vecB
    return np.sqrt(sum(np.power(diffAB,2)))

#如果是球面上的两点 计算球面距离



#构建簇 质心-随机初始化
def randCent(dataSet,k):
    n=dataSet.shape[1] #取列
    centroids=np.zeros((k,n)) #生成k个点
    for j in range(n):
        minJ=dataSet[:,j].min()
        rangeJ=float(dataSet[:,j].max()-minJ)
        centroids[:,j]=minJ+rangeJ*np.random.rand(k)
    return centroids


# KMeans++ 初始化 基本思想：随机选择一个点，后续的点 离第一个越远越好
def kmeansPlusPlus(dataSet, k):
    m, n = dataSet.shape

    # 保存k个质心
    centroids = np.zeros((k, n))

    # 随机选择第一个质心
    firstIndex = np.random.randint(m)
    centroids[0] = dataSet[firstIndex]

    # 依次选择后续质心
    for cent in range(1, k):

        # 每个样本到“最近已有质心”的距离平方
        distSq = np.zeros(m)

        for i in range(m):
            minDist = float('inf')

            # 与已有质心比较
            for j in range(cent):
                diff = dataSet[i] - centroids[j]
                dist = np.sum(diff ** 2)

                if dist < minDist:
                    minDist = dist
            distSq[i] = minDist

        #  按概率选择
        # 概率 ∝ 距离平方

        totalDistSq = np.sum(distSq)

        # 特殊情况：所有点重合
        if totalDistSq == 0:
            centroids[cent:] = dataSet[np.random.randint(m)]
            break

        probs = distSq / totalDistSq

        # 累积概率
        cumulativeProbs = np.cumsum(probs)

        # 随机数
        r = np.random.rand()

        # 找到对应区间
        for i in range(m):
            if r < cumulativeProbs[i]:
                centroids[cent] = dataSet[i]
                break

    return centroids

def Kmeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    m=dataSet.shape[0]
    clusterAssment=np.zeros((m,2)) #记录每个样本点 被分给哪个质心, 对每个样点 (簇编号，以及距离该簇的距离）
    centroids=createCent(dataSet,k)
    clusterChanged=True
    while clusterChanged:
        clusterChanged=False
        # 遍历每个样本点 距离哪个质心 最近，就分给哪个
        for i in range(m):
            minDist=float('inf')
            minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            if clusterAssment[i,0]!=minIndex:
                clusterChanged=True
            clusterAssment[i,:]=minIndex,minDist**2
        #print(centroids)
        # 遍历每个簇
        for cent in range(k):
            # 取出分到 该簇的 样本点 clusterAssment[:,0]取0列 等于cent的样本点 是true的返回所有样本的索引
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0]==cent)[0]]

            #在求平均时,应该判断该簇是否为空
            if len(ptsInClust) > 0:
                centroids[cent,:]=np.mean(ptsInClust,axis=0) #按列求平均
    return centroids,clusterAssment

fileName=r"E:\机器学习笔记\KMeans\testSet.txt"
dataMat=loadDataSet(fileName)
dataArr=np.array(dataMat)



# for k in range(1,30,1):
#     centroids,clusterAssment=Kmeans(dataArr,k)
#     totalDist=np.sum(clusterAssment[:,1])
#     print(f'{k}个簇的总距离：{totalDist}')





