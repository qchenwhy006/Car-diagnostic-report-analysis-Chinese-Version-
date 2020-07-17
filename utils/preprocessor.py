import pandas as pd
import os, re, jieba
from utils.config import train_data_file,test_data_file,root


def data2file_with_seg(file_dir):
    df=pd.read_csv(file_dir,encoding='utf-8')
    print('Datasize:', len(df))

    df_special1 = df.copy()
    df_special2 = df.copy()
    df_special3 = df.copy()
    df_special4 = df.copy()


    if 'Report' in df.columns:
        df['Dialogue'] = df['Dialogue'].apply(lambda x: str(x))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: x.replace('技师说：[语音]|', ''))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: x.replace('技师说：[图片]|', ''))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: x.replace('技师说：', ''))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: clean_sentence(x))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: seperate_sentence_into_words(x))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: None if len(x) < 2 else x)

        df_special2 = df.copy()
        df_special3 = df.copy()

        special_sample1=[] # len(special_sample1) 4474  help='Strange sample'
        special_sample2=[] # len(special_sample2) 16654 help='copy with the exactly same technician answer'
        for index,row in enumerate(df['Dialogue']):
            if row==None:
                special_sample1.append('Q'+str(index+1))
        for index, row in enumerate(df['Dialogue']):
            if row!=None:
                if row.find('|') == -1:
                    special_sample2.append('Q'+str(index+1))

        df_special1 = df_special1[df_special1['QID'].isin(special_sample1)]
        df_special1= df_special1[['QID','Question','Dialogue','Report']]
        df_special11 = df_special1[['QID', 'Dialogue', 'Report']]
        file = os.path.join(root,'data','special_samples','train_special1.csv')
        file1 = os.path.join(root, 'data', 'special_samples', 'train_special11.csv')
        df_special1.to_csv(file, index=None, header=False, encoding='utf-8')
        df_special11.to_csv(file1, index=None, header=False, encoding='utf-8')


        df_special2 = df_special2[df_special2['QID'].isin(special_sample2)]
        df_special2 = df_special2[['QID','Dialogue', 'Report']]
        file = os.path.join(root, 'data', 'special_samples', 'train_special22.csv')
        df_special2.to_csv(file, index=None, header=False, encoding='utf-8')


        df['Report'] = df['Report'].apply(lambda x: str(x))

        special_sample3 = []
        for index, row in enumerate(df['Report']):
            if row.find('您当前的问题已经解决') == 0:
                special_sample3.append('Q'+str(index+1))

        df['Report'] = df['Report'].apply(lambda x: re.sub(r'随时联系|详见图片|见图|看图|如图购买|直接建议', '', x))
        df['Report'] = df['Report'].apply(lambda x: clean_sentence(x))
        df['Report'] = df['Report'].apply(lambda x: ' '.join(jieba.cut(x, cut_all=False)))


        df['Report'] = df['Report'].apply(lambda x: None if len(x) < 3 else x)


        special_sample4=[] # len(special_sample3) 795
        for index,row in enumerate(df['Report']):
            if row==None:
                special_sample4.append('Q'+str(index+1))

        df_special3 = df_special3[df_special3['QID'].isin(special_sample3)]
        df_special3 = df_special3[['QID','Question','Dialogue', 'Report']]
        file = os.path.join(root, 'data', 'special_samples', 'train_special3.csv')
        df_special3.to_csv(file, index=None, header=False, encoding='utf-8')

        df_special4 = df_special4[df_special4['QID'].isin(special_sample4)]
        df_special4 = df_special4[['QID','Question', 'Dialogue', 'Report']]
        file = os.path.join(root, 'data', 'special_samples', 'train_special4.csv')
        df_special4.to_csv(file, index=None, header=False, encoding='utf-8')


        print('len(special_sample1)', len(special_sample1))
        print('len(special_sample2)', len(special_sample2))
        print('len(special_sample3)', len(special_sample3))
        print('len(special_sample4)', len(special_sample4))
        special_sample=special_sample1+special_sample2+special_sample3+special_sample4
        special_sample=list(set(special_sample)) # len(special_sample) 21601
        print(special_sample)
        print('len(special_sample)', len(special_sample))

        df=df[~df['QID'].isin(special_sample)]
        print('Datasize after cleaning:', len(df)) # Datasize after cleaning: 61342
        df=df[['QID','Dialogue','Report']]


        file = os.path.join(root, 'data','clean_train.csv')
        df.to_csv(file, index=None, header=False, encoding='utf-8')

    else:
        df['Dialogue'] = df['Dialogue'].apply(lambda x: str(x))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: x.replace('技师说：', ''))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: clean_sentence(x))
        df['Dialogue'] = df['Dialogue'].apply(lambda x: seperate_sentence_into_words(x))

        df['Dialogue'] = df['Dialogue'].apply(lambda x: None if len(x) < 2 else x)

        special_sample1 = []  # len(special_sample1) 4474  help='Strange sample'
        special_sample2 = []  # len(special_sample2) 16654 help='copy with the exactly same technician answer'
        for index, row in enumerate(df['Dialogue']):
            if row == None:
                special_sample1.append('Q' + str(index + 1))
        for index, row in enumerate(df['Dialogue']):
            if row != None:
                if row.find('|') == -1:
                    special_sample2.append('Q' + str(index + 1))

        df_special1 = df_special1[df_special1['QID'].isin(special_sample1)]
        df_special1 = df_special1[['QID','Dialogue']]
        file = os.path.join(root, 'data', 'special_samples', 'test_special1.csv')
        df_special1.to_csv(file, index=None, header=False, encoding='utf-8')

        df_special2 = df_special2[df_special2['QID'].isin(special_sample2)]
        df_special2 = df_special2[['QID','Dialogue']]
        file = os.path.join(root, 'data', 'special_samples', 'test_special2.csv')
        df_special2.to_csv(file, index=None, header=False, encoding='utf-8')



        special_sample = special_sample1 + special_sample2
        special_sample = list(set(special_sample))  # len(special_sample) 21601
        print(special_sample)
        print('len(special_sample)', len(special_sample))

        df = df[~df['QID'].isin(special_sample)]
        print('Datasize after cleaning:', len(df))  # Datasize after cleaning: 61342
        df = df[['QID','Dialogue']]


        file = os.path.join(root,'data','clean_test.csv')
        df.to_csv(file, index=None, header=False, encoding='utf-8')


def clean_sentence(sentence):
    sentence = re.sub(r'[a-zA-Z0-9]|[\s+\-\/\[\]\{\}_$%^*(+\"\')]+|[+——()【】“”~@#￥%……&*（）"～:..×±→Ω⊙※°]|不客气|嗯嗯|是的|①|②|③|④', '', sentence)
    return sentence



def seperate_sentence_into_words(sentence):
    result = []
    tokens = sentence.split('|')
    for token in tokens:
        if token.find('车主说')!=0:
            temp=list(jieba.cut(token,cut_all=False))
            if len(temp) > 2:
                result.append(' '.join(temp))
                # result.append(' '.join(list(jieba.cut(token,cut_all=False))))
    return '|'.join(result)
    #return ' '.join(result)


if __name__ == '__main__':

    data2file_with_seg(train_data_file) # Datasize: 82943 and Datasize after cleaning: 61292

    data2file_with_seg(test_data_file) # Datasize: 20000 and Datasize after cleaning: 14552
