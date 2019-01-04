# -*- coding:utf-8 -*-
import os
import shutil
import jieba
import numpy as np

np.set_printoptions(threshold=np.nan)


def create_fenci(filename):
    # step2 读取文本，预处理，分词，以及每个词的单词个数
    raw_word_list = []
    sentence_list = []
    stop_word = stop_words()
    with open(filename, encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n', '')
            while ' ' in line:
                line = line.replace(' ', '')
            if len(line) > 0:  # 如果句子非空
                raw_words = list(jieba.cut(line, cut_all=False))
                dealed_words = []
                for word in raw_words:
                    if word not in stop_word and word not in ['日期', '版号', '标题', '作者', '正文']:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)
            line = f.readline()
    return raw_word_list


def fenci(content):
    raw_word_list = []
    sentence_list = []
    stop_word = stop_words()
    while '\n' in content:
        content = content.replace('\n', '')
    while ' ' in content:
        content = content.replace(' ', '')
    if len(content) > 0:  # 如果句子非空
        raw_words = list(jieba.cut(content, cut_all=False))
        dealed_words = []
        for word in raw_words:
            if word not in stop_word and word not in ['日期', '版号', '标题', '作者', '正文']:
                raw_word_list.append(word)
                dealed_words.append(word)
        sentence_list.append(dealed_words)
    return raw_word_list


def eachFile(filepath):  # 读取每个文件
    file = []
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        file.append(child)
    return file


def stop_words():  # 读取停用词
    stop_words = []
    with open('stop_words.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words


def fenci_all(file_list):  # 对每个文件分词
    word_list = []
    # 存放每个文本的标签即所属的类
    label = []
    # class_df_list 每个单词在每个类中出现的文本数
    class_df_list = np.zeros(10)
    for article in file_list:
        tmp = []
        tmp = create_fenci(article)
        word_list.append(tmp)
        if '电脑' in article:
            label.append(1)
            class_df_list[0] += 1
        elif '环境' in article:
            label.append(2)
            class_df_list[1] += 1
        elif '交通' in article:
            label.append(3)
            class_df_list[2] += 1
        elif '教育' in article:
            label.append(4)
            class_df_list[3] += 1
        elif '经济' in article:
            label.append(5)
            class_df_list[4] += 1
        elif '军事' in article:
            label.append(6)
            class_df_list[5] += 1
        elif '体育' in article:
            label.append(7)
            class_df_list[6] += 1
        elif '医药' in article:
            label.append(8)
            class_df_list[7] += 1
        elif '艺术' in article:
            label.append(9)
            class_df_list[8] += 1
        elif '政治' in article:
            label.append(10)
            class_df_list[9] += 1
    return label, class_df_list, word_list


def read_file(filename):
    """读取文件数据"""

    PATH_NAME = "new_train"
    shutil.rmtree(PATH_NAME)  # 将整个文件夹删除
    os.makedirs(PATH_NAME)  # 创建一个文件夹
    word_list = []
    class_df_list = np.zeros(10)
    label = []
    with open(filename, encoding='utf-8', errors='ignore') as f:
        for line in f:
            word=[]
            labels, content = line.strip().split('\t')
            word_list.append(fenci(content))
            if '体育' in labels:
                label.append(1)
                class_df_list[0] += 1
                word.append(fenci(content))
                with open('new_train/体育.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '财经' in labels:
                label.append(2)
                class_df_list[1] += 1
                word.append(fenci(content))
                with open('new_train/财经.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '房产' in labels:
                label.append(3)
                class_df_list[2] += 1
                word.append(fenci(content))
                with open('new_train/房产.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '家居' in labels:
                label.append(4)
                class_df_list[3] += 1
                word.append(fenci(content))
                with open('new_train/家居.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '教育' in labels:
                label.append(5)
                class_df_list[4] += 1
                word.append(fenci(content))
                with open('new_train/教育.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '科技' in labels:
                label.append(6)
                class_df_list[5] += 1
                word.append(fenci(content))
                with open('new_train/科技.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '时尚' in labels:
                label.append(7)
                class_df_list[6] += 1
                word.append(fenci(content))
                with open('new_train/时尚.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '时政' in labels:
                label.append(8)
                class_df_list[7] += 1
                word.append(fenci(content))
                with open('new_train/时政.txt', 'a+',encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '游戏' in labels:
                label.append(9)
                class_df_list[8] += 1
                word.append(fenci(content))
                with open('new_train/游戏.txt', 'a+' , encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')
            elif '娱乐' in labels:
                label.append(10)
                class_df_list[9] += 1
                word.append(fenci(content))
                with open('new_train/娱乐.txt', 'a+' , encoding='utf-8', errors='ignore') as f:
                        f.write(str(word))
                        f.write('\n')

    return label, class_df_list, word_list
if __name__ == '__main__':
    # filePathC = "text1000"  # 从text文件中读取文件
    # file_list = "text1000"  # 每个文件名数组
    # file_path = ""
    # print("文件名", file_list)
    # # word_list存放词典
    # label, class_df_list, word_list = fenci_all(file_list)  # 分词后1）获得标签2）每个词对应的类别的文本数3）每个文本词典
    #
    # print(class_df_list)
    file_path = "cnew.txt"
    label, class_df_list, word_list=read_file(file_path)
    print(word_list)
