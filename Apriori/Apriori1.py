#关联学习

"""
发现毒蘑菇的相似特征：
数据集， 第一列 表示可食用/有毒 1-可使用 2-有毒
       第二列 蘑菇伞的形状 有六种可能的值  用3-8来表示

"""
#1.加载数据集  mushroom.dat
import pandas as pd
def loadDataSet(filePath):
    """
    读取蘑菇数据集并直接转换为集合列表
    :param filePath: 文件路径
    :return:
    """
    df=pd.read_csv(filePath,sep=r'\s+')
    dataSet=[set(row) for row in df.values]
    return dataSet


# 返回C1是记录 候选项集(size)为1 集合
def createC1(dataSet):
    C1=[]
    #数据集的每条记录tran
    for tran in dataSet:
        for item in tran:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #使用map 返回迭代器 并转成list
    return list(map(frozenset,C1))



def scanD(D,Ck,minSupport):
    """
    :param D:  数据集
    :param Ck:  候选项集的集合 { {},{},...}
    :param minSupport:
    :return: retList 记录>=minSupport的项集  supportData 字典{项集-支持度, }
    """
    ssCnt={}
    #扫描数据集D 取出一条记录 tid
    for tid in D:
        #从Ck取出一个项集 (C_1时 每个元素是大小为1的项集)
        for can in Ck:
            # 避免组合出不存在的 先判断是否在tid
            if can.issubset(tid):
                # 哈希计数
                if can not in ssCnt:
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    #计算每个项集的支持度
    numItems=float(len(D))
    retList=[]
    supportData={}
    for key in ssCnt:
        support=ssCnt[key]/numItems
        if support>=minSupport:
            retList.insert(0,key)#不一定插到开头,只是为了
        supportData[key]=support
    return retList,supportData


#项集合并 创建候选项集 Ck
def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)

    """
    取出Lk两个项集 (i,j)-- (L1,L2)
    Lk[i]的大小=k-1 取出k-2,留最后一项 
    因为项集是不重复的， 示例{0,1} {0,2} {1,2}
    如果前k-2项是重复，且这两个项集不能完全重复，最后一项必不同，用于合并
    反之 如果前k-2项有重复  没必要合并(只会是重复结果)
    
    """
    # 从取出两个项集i,j
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2]
            L2=list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

#创建C1后 不断由Ck到Lk 直到
def apriori(dataSet,minSupport=0.5):
    """

    :param dataSet:
    :param minSupport:
    :return: 所有项集的列表L 以及项集的支持度
    """
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1] #列表
    k=2
    #列表元素 项集个数=0 退出
    while(len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData   #L={ {{项集1},{项集2}},{{项集1},{项集2}},...}

 #从频繁项集挖掘关联规则
def generateRules(L,supportData,minConf=0.7):

     bigRuleList=[]
     for i in range(1,len(L)): #从1开始 是从包含两个物品/特征的频繁项集的列表 开始, 以便组成前件-后件
         #L[i] 是长度相同的 频繁项集 集合
         for freqSet in L[i]:
             H1=[frozenset([item]) for item in freqSet]
             if i>1:

                 rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)

             else:
                 # i=1   比如freqSet={A,B}  后件无法组合 否则前件为空  即当后件无法组合,直接calc
                 calcConf(freqSet,H1,supportData,bigRuleList,minConf)
     return bigRuleList

#扫描H, 取出候选 后件,计算置信度, 当达到最小置信度,视为一条规则
def calcConf(freqSet,H,supportData,brl,minConf=0.7):
     prunedH=[]
     #关联规则 前件-->后件 P'-->H'
     # P': freqSet-conseq   H': conseq

     for conseq in H: # H={特征1,特征2,...}
         conf=supportData[freqSet]/supportData[freqSet-conseq]
         if conf>=minConf:
             #print(f'前件:{freqSet-conseq}-->后件:{conseq},可信度:{conf}')
             brl.append((freqSet-conseq,conseq,conf))
             prunedH.append(conseq)
     return prunedH

# 判断 H里面的候选项集(用于后件)的个数 >1 rules计算
#                                  否则 直接计算 cal
def rulesFromConseq(freqSet,H,supportData,brl,minConf=0.7):

    m=len(H[0]) # 测H的每个元素大小 初始m=1
    if(len(freqSet)>(m+1)):
        #生成 m+1 项集  的集合
        Hmp1=aprioriGen(H,m+1)
        Hmp1=calcConf(freqSet,Hmp1,supportData,brl,minConf)

         #当后件的个数大于1 可以继续组合
         #示例 {23}-->{01},{13}-->{0,2}  进一步 后件组合{0,1,2}  分析{0,1,2,3}?{0,1,2}
        # 比如得到 {3}-->{0,1,2}
        if(len(Hmp1)>1):
            rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)


filePath="E:\机器学习笔记\Apriori\mushroom.dat"
dataSet=loadDataSet(filePath)

L,supportData=apriori(dataSet,minSupport=0.4)
all_rules=generateRules(L,supportData,minConf=0.7)

print("\n" + "="*20 + " 筛选结果 " + "="*20)
poisonous_rules=[]#有毒规则
edible_rules=[] #可食用规则

for item in all_rules:
    f_set,b_set,conf=item  #f_set 前件  b_set 后件
    # 2 in f_set(在前件发现有毒, 已知有毒-->伴随哪些特征)
    # 2 in b_set(在后件发现有毒,出现哪些特征-->有毒)

    if 2 in f_set or 2 in b_set:
        poisonous_rules.append(item)
    if 1 in f_set or 1 in b_set:
        edible_rules.append(item)

print(f"\n--- 共找到与[有毒(2)] 相关的强关联规则 {len(poisonous_rules)} 条：---")
for f_set,b_set,conf in poisonous_rules[:20]:
    print(f"规则:{set(f_set)}-->{set(b_set)}|置信度:{conf:.2f}")

print(f"\n--- 共找到与[无毒(1)]相关的强关联规则 {len(edible_rules)} 条：---")
for f_set,b_set,conf in edible_rules[:20]:
    print(f"规则:{set(f_set)}-->{set(b_set)}|置信度:{conf:.2f}")








