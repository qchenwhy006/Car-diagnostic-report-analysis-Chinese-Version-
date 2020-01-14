#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
import string
import re
from collections import Counter
import jieba
import tensorflow.python.framework.d

# ## 1. 加载数据

# In[4]:


train_data = pd.read_csv("AutoMaster_Trainset.csv")
test_data = pd.read_csv("AutoMaster_TestSet.csv")

# ## 2. 查看数据集结构

# In[5]:


display(train_data.head())
display(test_data.head())
print(train_data.shape)
print(test_data.shape)
print(train_data.info())
print(test_data.info())

# ## 3. 去掉缺失值

# In[39]:


train_data = train_data.dropna(axis=0)
test_data = test_data.dropna(axis=0)

# ## 4. 合并dataframe、预处理标点符号、切分句子

# In[183]:


train_data_report = train_data[['Report']]
train_data_main = train_data.loc[:, 'Brand':'Dialogue']
test_data_main = test_data.loc[:, 'Brand':]
frame = pd.concat([train_data_main, test_data_main])

# 加上中文符号
punc_str = '[’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】：。，、《》（）～？“”‘’！％——[\\]^_`{|}~]+...'


def cut(f, freq=50):
    f = re.sub('\[\S*?\]', ' ', f)
    cut_result = jieba.cut(f, cut_all=False)
    sr = ' '.join(cut_result)
    sr_after = re.sub(punc_str, ' ', sr)

    word_counts = Counter(sr_after.split(' '))

    # 考虑词频
    trimmed_words = [word for word in word_counts if word_counts[word] > freq]
    return trimmed_words


words_dialogue = cut(' '.join(frame['Dialogue']))
words_question = cut(' '.join(frame['Question']))

# In[184]:


# 考虑report列
words_report = cut(' '.join(train_data['Report']))

# ## 5. 去掉数字、字母、空字符

# In[185]:


ls = '[a-zA-Z0-9’"#$%&\'()*+,-./:;<=>…“”‘’！[\\]^_`{|}~]+'

# 去掉字母数字
words_dialogue1 = [re.sub(ls, '', i) for i in words_dialogue]
words_question1 = [re.sub(ls, '', i) for i in words_question]
words_report1 = [re.sub(ls, '', i) for i in words_report]


# 去掉空字符
def null_remove(wordslist):
    while '' in wordslist:
        wordslist.remove('')
    return wordslist


words_dialogue2 = null_remove(words_dialogue1)
words_question2 = null_remove(words_question1)
words_report2 = null_remove(words_report1)

# ## 6. 停用词采样

# In[186]:


## 前两列选出unique的词语加入字典
s1 = set(frame['Brand'])
s2 = set(frame['Model'])
car_set = s1.union(s2)
total_list = list(car_set) + words_dialogue2 + words_question2 + words_report2
total_count = len(total_list)
print(total_count)
# ((set(words_dialogue2).union(set(words_question2))). union(set(words_report2))).union(car_set)


# In[187]:


vocab_initial = list(set(total_list))
vocab_count = len(vocab_initial)
print(vocab_count)

# In[193]:


vocab_to_int = {word: index for index, word in enumerate(vocab_initial)}  # 7797
int_to_vocab = {index: word for index, word in enumerate(vocab_initial)}
int_words = [vocab_to_int[w] for w in total_list]  # index 12714
print(len(int_words))
print(len(list(set(int_words))))

# In[189]:


##所以vocab_initial中无停用词的删除
vocab_final = vocab_initial

# ## 7. 输出文件

# In[191]:


# 先创建并打开一个文本文件
file = open('user_vocab.txt', 'w')

# 遍历字典的元素，将每项元素的key和value分拆组成字符串，注意添加分隔符和换行符
for k, v in vocab_to_int.items():
    file.write(str(k) + ' ' + str(v) + '\n')

# 注意关闭文件
file.close()

# for k,v in sorted(dict_temp.items()):
# 	file.write(str(k)+' '+str(v)+'\n')
# file.close()