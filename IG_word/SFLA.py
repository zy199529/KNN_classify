# -*- coding:utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from IG_word.tf_idf_sfla import *
from IG_word.KNN_classify import *


def words():  # 读取降维后的词语
    words = []
    with open('reduction_words.txt', encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            words.append(line[:-1])
            line = f.readline()
    return words


# 使用蛙跳算法智能优化特征二次选择
def SFLA_init():
    vocabList = words()  # 使用信息熵降维后的词典
    frogNum = 20  # 蛙群规模N=20一共20个青蛙
    L = 10  # 族群内进化次数L
    T = 10  # 总迭代次数T
    Dmax = 45  # 最大移动步长Dmax
    memeplexNum = 5  # 族群数量5
    myarray = np.random.randint(0, 2, (20, len(vocabList)))  # myarray初始化青蛙的特征项0或1
    Xg_value_first, Xb_value_first = max_min_fitness(frogNum, myarray)  # 找到分类最好的那只青蛙
    return myarray, myarray[Xg_value_first], frogNum, L, T, Dmax, memeplexNum  # 返回随机生成的青蛙序列，最大的那只青蛙，词典


# 青蛙个数，族群内进化次数L，总迭代次数T，最大移动步长Dmax


def SFLA_memeplex():  # 族群内进化
    myarray, Xg, frogNum, L, T, Dmax, memeplexNum = SFLA_init()
    # 将20只青蛙排序按适应度从大到小降序
    M = sort_frog(frogNum, memeplexNum, myarray)  # 第一次降序后放入族群中
    for s in range(T):
        M = Iteration(memeplexNum, L, M, Xg)  # 一次迭代，一次迭代后将二维数组放入一维数组中
        array = []
        for y in range(memeplexNum):  # 放入以为数组后进行第二次迭代
            for x in range(4):
                array.append(M[y][x])
        M = sort_frog(frogNum, memeplexNum, array)  # 降序后放入族群中
        print("t=", s, second_reduction(Xg))
        Xg_end, Xb_end = max_min_fitness(frogNum, array)
        Xg = array[Xg_end]

    return array[Xg_end]


def Iteration(memeplexNum, L, M, Xg):  # 一次迭代
    for i in range(memeplexNum):  # 每个族群内部进化
        for t in range(L):  # 族群内进化次数L，每个族群进化10次
            r1 = np.random.randint(0, 100)  # 生成随机数r1
            r2 = np.random.randint(0, 100)  # 生成随机数r2
            Xb_value, Xw_value = max_min_fitness(4, M[i])  # 每个族群中最好和最差的青蛙的位置
            Xb = M[i][Xb_value]  # 记录下族群内最好的青蛙的特征
            Xw = M[i][Xw_value]  # 记录下族群内最坏的青蛙的特征

            Xw_new = frog_evolution(Xb, Xw, r1, r2)  # 青蛙进化，Xb是最好的青蛙，Xw是最坏的青蛙，r1和r2是随机数
            if second_reduction(Xw_new) > second_reduction(Xw):  # 将产生的新的青蛙和旧的青蛙比较适应度函数
                M[i][Xw_value] = Xw_new  # 若新青蛙好则用新青蛙代替
            else:
                # 否则用全局最优代替局部最优再进行优化
                Xw_new = frog_evolution(Xg, Xw, r1, r2)  # 全局最优进行优化
                if second_reduction(Xw_new) > second_reduction(Xw):  # 假如效果好则使用全局最优后选出的青蛙
                    M[i][Xw_value] = Xw_new  # 新青蛙代替
                else:
                    M[i][Xw_value] = np.random.randint(0, 2, len(Xb))  # 若效果还是不好则随机产生
    return M


def frog_evolution(Xb, Xw, r1, r2):
    G = np.zeros(len(Xb))  # G产生一个零向量
    xb = Xb
    xw = Xw
    Xb_Xw = np.zeros(len(xb))  # Xb_Xw产生一个零向量
    Xw_Xb = np.zeros(len(xb))  # Xw_Xb产生一个零向量
    for j in range(len(xb)):  # 将最好和最差的青蛙共同为1的位置保存下来，也将G的相同位置赋值为1，将Xb和Xw
        if xb[j] == xw[j]:
            G[j] = 1
    a = 0
    b = 0
    for k in range(len(xb)):
        if xb[k] == 1 and xw[k] == 0:
            Xb_Xw[k] = 1
            a = a + 1
    a = int(a * r1)  # Xb_Xw需要保留几个1,其余清为0
    for q in range(len(xb)):
        if xw[q] == 1 and xb[q] == 0:
            Xw_Xb[q] = 1
            b = b + 1
    b = int(b * r2)  # Xw_Xb需要保留几个1,其余清为0
    o = 0
    for w in range(len(xb)):  # Xb_Xw保留1,其余清为0
        if Xb_Xw[w] == 1 and o <= a:
            o = o + 1
        else:
            Xb_Xw[w] = 0
    z = 0
    for w in range(len(xb)):  # Xw_Xb保留1,其余清为0
        if Xw_Xb[w] == 1 and z <= b:
            z = z + 1
        else:
            Xw_Xb[w] = 0
    Xw_new = Xw_Xb + Xb_Xw + G  # 新的青蛙产生
    return Xw_new


def max_min_fitness(frogNum, myarray):
    accuracy_all = []
    for i in range(frogNum):
        accuracy_all.append(second_reduction(myarray[i]))
    accuracy_all = np.array(accuracy_all)
    index = np.array(np.argsort(accuracy_all)[:: -1])
    return index[0], index[frogNum - 1]  # 找到适应度最好和最差的那只青蛙的位置


def sort_frog(frogNum, memeplexNum, myarray):  # 将青蛙从大到小排序后放入族群中
    M = []
    accuracy_all = []
    for i in range(frogNum):
        accuracy_all.append(second_reduction(myarray[i]))
    accuracy_all = np.array(accuracy_all)
    print(accuracy_all)
    index = np.argsort(accuracy_all)[:: -1]
    # accuracy_all = accuracy_all[np.lexsort(-accuracy_all.T)]
    print(index)
    for p in range(memeplexNum):  # 族群进行初始化为空
        M.append([])
    g = 0
    for p in index:  # 将青蛙加入到族群中
        if g < 20:
            M[g % memeplexNum].append(myarray[p])
            g = g + 1
    return M


def KNN_classify(second_redution):  # 对每一只青蛙（词库）KNN分类
    train_vec_List_sfla, idf_array_sfla, label_sfla = tf_idf_sfla(second_redution)  # 计算训练集的tf_idf
    test_vec_List_sfla, test_label_sfla = test_tf_idf_sfla(second_redution)  # 计算测试集的tf_idf
    k = 25
    prediction_sfla = []
    for x in range(len(test_vec_List_sfla)):
        neighbors_sfla = getNeighbors(train_vec_List_sfla, test_vec_List_sfla[x], k, label_sfla)
        result_sfla = getResponse(neighbors_sfla)
        prediction_sfla.append(result_sfla)
        # print("prediction=" + repr(result_sfla) + "  actual=" + repr(test_label_sfla[x]))#预测值和真实值
    accuracy_sfla = getAccurcy(test_label_sfla, prediction_sfla)  # 正确率%
    return accuracy_sfla


def second_reduction(frog_word):
    frog_vocablist = []
    vocabList = np.array(words())
    for i in range(len(frog_word)):
        if frog_word[i] == 1:
            frog_vocablist.append(vocabList[i])
    accuracy = KNN_classify(frog_vocablist)
    return accuracy


if __name__ == '__main__':
    print(SFLA_memeplex())
