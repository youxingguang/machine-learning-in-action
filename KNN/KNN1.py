# k-近邻算法

import numpy as np
import operator
import random

# 1.加载数据-datingTestSet
"""
 数据说明: 数据一共 四列，前三列 分别是 每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每周消费冰淇淋公升数
                      最后一列 标签
"""
def loadDataSet(fileName):
    dataMat=[];labelMat=[] #新建两个空列表
    fr=open(fileName)
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1]),float(lineArr[2])])
        labelMat.append(lineArr[3])
    return dataMat,labelMat

# 2.归一化处理  x_i=(x_i1,x_i2,...)
def normSet(dataSet):
    #按属性列 取最大 最小值
    dataSet=np.array(dataSet)
    minVals=dataSet.min(axis=0)
    maxVals=dataSet.max(axis=0)

    m,n=dataSet.shape
    newData=np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            newData[i][j]=(dataSet[i][j]-minVals[j])/(maxVals[j]-minVals[j])
    return newData,minVals,maxVals

# 3.计算欧式距离
def computeEuDistance(x1,x2):
    diffX=x1-x2 #逐点差分
    squareDiffX=diffX.dot(diffX) #差分向量内积
    return np.sqrt(squareDiffX) #开方

# 4.排序与筛选 选出k个相邻的样本
def selectNearK(inX,newData,minVals,maxVals,k,labels):
    inX_norm=(inX-minVals)/(maxVals-minVals)
    n=len(newData)
    distances=np.zeros(n)
    for i in range(n):
        distances[i]=computeEuDistance(inX_norm,newData[i])
    #返回排序后的索引
    sortedIndex=distances.argsort()
    classCount={} #字典  统计前k个类别
    for i in range(k):
        label=labels[sortedIndex[i]]
        classCount[label]=classCount.get(label,0)+1
    # 类别 按票数 排序
    sortedClass=sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回第一个类别
    return sortedClass[0][0]


# 5.数据测试


# 6.K折交叉验证
def crossValidation(dataSet, labelMat, foldNum, kList):

    dataSet = np.array(dataSet)
    labelMat = np.array(labelMat)
    m = len(dataSet)

    # 打乱索引
    indexList = list(range(m))
    random.shuffle(indexList)

    # 每折大小
    foldSize = m // foldNum

    #记录 最优 k
    bestK=-1
    bestErrorRate=float("inf")
    # 测试不同k
    for k in kList:
        errorCount = 0
        totalTest = 0
        # 开始K折
        for fold in range(foldNum):
            start = fold * foldSize

            # 最后一折补齐
            if fold == foldNum - 1:
                end = m
            else:
                end = start + foldSize

            testIndex = indexList[start:end]

            trainIndex = (
                indexList[:start] +
                indexList[end:]
            )

            # 区分训练集和测试集  后面从测试集取点找k个  比较 训练集的标签 与 测试点本身的标签
            trainData = dataSet[trainIndex]
            trainLabel = labelMat[trainIndex]
            testData = dataSet[testIndex]
            testLabel = labelMat[testIndex]

            # 训练集归一化
            normTrain, minVals, maxVals = normSet(trainData)

            # 开始测试
            for i in range(len(testData)):
                result = selectNearK(testData[i],normTrain,minVals,
                    maxVals,k,trainLabel)
                if result != testLabel[i]:
                    errorCount += 1
                totalTest += 1
        errorRate = errorCount / totalTest
        print("k=%d 错误率=%.4f" %(k, errorRate))

        #更新k
        if errorRate<bestErrorRate:
            bestErrorRate=errorRate
            bestK=k
    return bestK

fileName=r"E:\机器学习笔记\KNN\datingTestSet.txt"
dataSet,labelMat=loadDataSet(fileName)

# 将样本拆分 5份 做交叉测试 输出不同k的错误率
k=crossValidation(dataSet,labelMat,foldNum=5,kList=[1,3,5,7,9])

inX=[7467,14.445740,1]
newData,minVals,maxVals=normSet(dataSet)
result=selectNearK(inX,newData,minVals,maxVals,k,labelMat)
print(f'新加入的测试点：{result}')
