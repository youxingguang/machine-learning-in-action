#使用 ID3 算法 生成 决策树

"""
   ID3 通过信息增益，来选择关键特征

   计算熵
   为每个属性,计算条件熵
   计算每个属性 带来的信息增益
   选择信息增益大的作为关键特征，划分子集
   判断子集是否属于同一类，是不再继续划分；否继续划分
"""

from math import log
import pandas as pd
import operator
import tree_plot
import pprint

#以西瓜书的例子创建数据集
def loadDataSet(file_path):
    df = pd.read_csv(file_path)

    #  清洗列名（去空格）
    df.columns = df.columns.str.strip()

    #  清洗每个单元格（去空格）
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    #  前面去空格，方便删除编号列
    if '编号' in df.columns:
        df = df.drop(columns=['编号'])

    feature_names = df.columns[:-1].tolist()
    dataSet = df.values.tolist()

    return dataSet, feature_names



# 计算给定样本的熵
def calcShannonEnt(dataSet):
    numEntries=len(dataSet) #样本数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] #featVect 一行特征向量，取最后一位（即分类标签）
        if currentLabel not in labelCounts.keys():
          labelCounts[currentLabel]=0 #类似哈希，不在默认0，在 加1
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#划分数据集
# axis固定列 一个特征  找出该特征列 是value的部分
def splitDataSet(dataSet,axis,value):
    #创建一个新 list
    retDataSet=[]

    #将 属性是value 提取
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的特征划分
def chooseBestFeatureToSplit(dataSet):
    # 属性个数=列数-1
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0 #信息增益
    bestFeature=-1  #记录 信息增益最大的特征 索引=列号
    row=len(dataSet)
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(row)
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#假定数据集 处理了所有的属性，类标签仍然不唯一，采用多数表决
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

# 创建树
def createTree(dataSet,labels):
    labels=labels[:] #复制 避免del破坏
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


#读取数据
file_path = r'D:\watermelon.csv'
dataSet, labels = loadDataSet(file_path)
tree=createTree(dataSet,labels)
pprint.pprint(tree) #简单输出结果
tree_plot.createPlot(tree)
