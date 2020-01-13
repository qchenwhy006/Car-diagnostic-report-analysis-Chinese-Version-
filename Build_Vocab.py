import jieba
import pandas as pd
import re

path1='C:\\Users\\qianqian\\Desktop\\AutoMaster_TrainSet1.txt'
path2='C:\\Users\\qianqian\\Desktop\\AutoMaster_TestSet.txt'

def load_dataset(train_data_path,test_data_path):
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    return train_data,test_data

train_df,test_df=load_dataset(path1,path2)
print('train data size: {}, test data size: {}'.format(len(train_df),len(test_df)))

train_df=train_df.fillna(' ')
test_df=test_df.fillna(' ')

def clean__sentence(sentence):
    if isinstance(sentence,str):
        return re.sub(r'[\s+\-\|\!\/\[\]\{\}_,，,\【\】！？]',' ',sentence)
    else:
        return ' '

def load_stop_words(stop_word_path):
    file=open(stop_word_path,'r',encoding='utf-8')
    stop_words=file.readlines()
    stop_words=[stop_word.strip() for stop_word in stop_words]
    return stop_words

user_dict_words=pd.concat([rawdata['Model'],rawdata['Brand']],axis=0,join='outer')
new_words=add_words.unique()
with open("C:\\Users\\qianqian\\Desktop\\new_words_dict.txt", "w", encoding='utf-8') as f:
    for ele in new_words:
        f.write(str(ele))
        f.write('\n')
jieba.load_userdict("C:\\Users\\qianqian\\Desktop\\new_words_dict.txt")

111
'''



#添加自定义词汇，词汇来自于Brand和Model
add_words=pd.concat([rawdata['Model'],rawdata['Brand']],axis=0,join='outer')
new_words=add_words.unique()
with open("C:\\Users\\qianqian\\Desktop\\new_words_dict.txt", "w", encoding='utf-8') as f:
    for ele in new_words:
        f.write(str(ele))
        f.write('\n')
jieba.load_userdict("C:\\Users\\qianqian\\Desktop\\new_words_dict.txt")


#jieba分词
columns=['Question','Dialogue','Report']
seg=[]
dic={}
for i in range(len(rawdata)):
    for one in columns:
        for word in jieba.cut(str(rawdata[one]),cut_all=False):
            seg.append(word)


for element in seg:
    if element not in dic:
        dic[element]=1
    else:
        dic[element]=dic[element]+1


#对词进行筛选
import re
pattern=r"[\u4e00-\u9fa5]+"
prevocab=[]
import operator
sorted_word=sorted(dic.items(),key=operator.itemgetter(1),reverse=True)
for result in sorted_word:
    if len(result[0])>=2: #去除词的长度为1的词
        Chinese_word=re.findall(pattern,result[0]) #选取中文词
        prevocab+=Chinese_word
for ele in new_words:
    prevocab.append(ele) #加入汽车牌子和款型的专有名词


#创建vocab词表
vocab={}
for index,value in enumerate(prevocab):
    vocab[value]=index

file = open('vocab_CQQ.txt', 'w')
for k, v in vocab.items():
    file.write(str(k) + ':' + str(v) + '\n')
file.close()
'''