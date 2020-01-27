# encoding=utf-8

import jieba
import pandas as pd
import re
from collections import Counter
import gensim
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
import logging


#输入输出文件信息
path1='C:\\Users\\qianqian\\Desktop\\AutoMaster_TrainSet.txt'
path2='C:\\Users\\qianqian\\Desktop\\AutoMaster_TestSet.txt'
path_user_dict='C:\\Users\\qianqian\\Desktop\\user_dict.txt'
path_stop_words='C:\\Users\\qianqian\\Desktop\\chineseStopword.txt'



#读取文件，返回文件的dataframe格式
def load_dataset(train_data_path,test_data_path):
    train_data=pd.read_csv(train_data_path)
    test_data=pd.read_csv(test_data_path)
    return train_data,test_data


#对文件中的标点进行过滤删除
def clean__sentence(sentence):
    if isinstance(sentence,str):
        return re.sub(r'[\s+\-\|\!\/\[\]\{\}_,，,\【\】！？：]',' ',sentence)
    else:
        return ' '


#对文件中的停用词进行过滤删除
def filter_stopwords(stop_word_path,words):
    file=open(stop_word_path,'r',encoding='utf-8')
    stop_words=file.readlines()
    stop_words=[stop_word.strip() for stop_word in stop_words]
    return [word for word in words if word not in stop_words]


#分词
def cut(f, freq=5):
    cut_result = jieba.cut(f, cut_all=False)
    sr = ' '.join(cut_result)
    word_counts = Counter(sr.split(' '))

    # 考虑词频
    trimmed_words = [str(word) for word in word_counts if word_counts[word] > freq]
    return trimmed_words









# 主函数
if __name__ == "__main__":
    # 读取文件
    train_df, test_df = load_dataset(path1, path2)
    train_df = train_df.fillna(' ')
    test_df = test_df.fillna(' ')
    print('train data size: {}, test data size: {}'.format(len(train_df), len(test_df)))

    #合并文件,形成句子
    train_df=pd.concat([train_df['Question'],train_df['Dialogue'],train_df['Report']],axis=0,join='outer', ignore_index=True)
    test_df=pd.concat([test_df['Question'],test_df['Dialogue']],axis=0,join='outer', ignore_index=True)
    df=pd.concat([train_df,test_df],join='outer',axis=0, ignore_index=True)



    # 使用正则过滤
    for i in range(len(df)):
        df[i] = clean__sentence(df[i])


    #切词
    jieba.load_userdict("C:\\Users\\qianqian\\Desktop\\user_dict.txt")
    words=cut(' '.join(df))




    #形成vocab
    vocab_initial = list(set(words))
    print('filter words: {}'.format(len(vocab_initial)))
    vocab_to_int = {word: index for index, word in enumerate(vocab_initial)}
    int_to_vocab = {index: word for index, word in enumerate(vocab_initial)}



    # 将结果保存到文件中
    file = open('vocab_CQQ.txt', 'w',encoding='utf-8')
    for k, v in vocab_to_int.items():
        file.write(str(k) + ':' + str(v) + '\n')
    file.close()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    data_path = 'C:\\Users\\hawko.DESKTOP-LD1FGF7\\Desktop\\data_merged_train_test_seg_data.txt'
    model = word2vec.Word2Vec(LineSentence(data_path), workers=4, sg=1, hs=1, min_count=5, size=300)
    embedding_matrix = {}
    reverse_vocab = {k: v for k, v in enumerate(model.wv.index2word)}
    for i in range(len(reverse_vocab)):
        key = reverse_vocab[i]
        matrix = model[key]
        embedding_matrix[key] = matrix

    print(embedding_matrix)

    file = open('C:\\Users\\hawko.DESKTOP-LD1FGF7\\Desktop\\word2vec.txt', 'w')
    save_model_path = 'C:\\Users\\hawko.DESKTOP-LD1FGF7\\Desktop\\word2vec.txt'
    model.wv.save_word2vec_format(save_model_path, binary=False)
