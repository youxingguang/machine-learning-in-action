# 朴素贝叶斯分类: 文本分类 —— 判定邮件是否为垃圾邮件


"""
 postingList[][]  每行视作一条留言或一封邮件  classVec为每行 标注分类
 createVocabList 将所有留言，处理为不重复的词库
 setOfWords2Vec 如果 某个input 的word 是否在词库里面，计1  string 转 数值类型

 trainNB0
 计算类别的先验概率 直接类标签变量统计， 二分类 垃圾邮件（包含侮辱、暴力等） 和 正常邮件
 计算 不同词，每个词都可作为一个属性（因为处理了无重复） 在不同类别下，
      p0Vect=p0Num/p0Denom    条件概率

测试样本  同样转化为无重复的一个向量，判断词是否出现， 根据不同类别出现概率，求和 ——>得分
        这样重判定 该测试留言是属于哪个分类

"""

from numpy import*

# 词表到向量的转换
def loadDataSet():
    #词表
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #手动 标注标签 1-侮辱性文字  0-正常性言论
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

# 创建词库 将所有文档不重复词塞进一个列表
def createVocabList(dataSet):
    vocabSet=set([]) #避免重复 用set
    for document in dataSet:
        vocabSet=vocabSet|set(document) #取并集
    return list(vocabSet)


#与词库等长的向量，初始为0，后面遍历输入的文档 在词库，写1
# set-of-words model 词集模型，只关注词是否出现，如果一个词出现多次应该用 词袋模型
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else: print (f"the word: {word} is not in my Vocabulary!")
    return returnVec

# bag-of-words model 词袋模型
# 当一个词不止出现一次，就统计次数
def bagofWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec



# 计算样本 w的后验概率 p(c_i|w)=p(w|c_i)p(c_i)/p(w)

"""
w 是向量 可包含多个属性(特征)  
若计算 后验,  需要p(w|c_i) 对每个类别的词统计
p(w|c_i)=p(w_0,w_1,...,|c_i) 假设各个词独立   条件独立性假设是朴素贝叶斯的朴素

"""
# 对训练集 数据处理
# 求类的先验概率  文本在已知类的条件概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix) #将postingList每行视作一篇文档
    numWords=len(trainMatrix[0]) #总词数

    #计算类别的先验概率 二分类 计算侮辱言论 剩下的1-p
    pAbusive=sum(trainCategory)/float(numTrainDocs)

    p0Num=ones(numWords) #初始 每个词都是1  拉普拉斯修正
    p1Num=ones(numWords)
    p0Denom=2.0 #0 分类的分母
    p1Denom=2.0
    for i in range(numTrainDocs):
        #遍历每篇文档
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
        #print(f"文档{i} ：{trainMatrix[i]}")
        #print(f"p0：{p0Num}")
        #print(f"p1：{p1Num}")
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #计算 新样本在 每个类别的得分
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

#测试 入口  使用词集模型
def testingNB(testEntry):
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(f"{testEntry} classified as: {classifyNB(thisDoc,p0V,p1V,pAb)}")

#测试 入口  使用词袋模型
def testingNB1(testEntry):
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postDoc in listOPosts:
        trainMat.append(bagofWords2VecMN(myVocabList,postDoc))
    p0V,p1V,pAb=trainNB0(array(trainMat),array(listClasses))
    thisDoc=array(bagofWords2VecMN(myVocabList,testEntry))
    print(f"{testEntry} classified as: {classifyNB(thisDoc,p0V,p1V,pAb)}")

testEntry1=['love','my','dalmation']
testEntry2=['stupid','garbage']
testingNB1(testEntry1)
testingNB1(testEntry2)
