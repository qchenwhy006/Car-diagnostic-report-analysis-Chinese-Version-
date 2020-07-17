import pandas as pd
import numpy as np
import collections
from utils.config import Train_X,Test_X,embedding_matrix_path,stopwords_path,\
    sentence_embedding_matrix_path,sentence_train_x,sentence_train_y,extractive_train_file,Train_y,sentence_test_x
from sklearn.decomposition import PCA
from utils.data_loader import load_embedding_matrix,load_vocab,load_stopwords




class SentenceEmbedding():
    def __init__(self,a=0.001):
        self.embedding_matrix=load_embedding_matrix(embedding_matrix_path)
        self.vocab,self.reversed_vocab=load_vocab()
        self.embedding_size=len(self.embedding_matrix[0])
        self.a=a

    def calculate_sentence_vector(self,sentence):
        words=sentence.strip().split()
        word_freq_dict=collections.Counter(words)
        vector=np.zeros(self.embedding_size)
        for word in words:
            word_index=self.vocab[word]
            word_embedding=self.embedding_matrix[word_index]
            word_freq=word_freq_dict[word]/len(words)
            word_weight=self.a/(self.a+word_freq)
            vector=np.add(vector,np.multiply(word_weight,word_embedding))
        sentence_vector=np.divide(vector,len(words))
        return sentence_vector

    def sentences_to_vector(self,sentence_list):
        vec_list=[]
        for sentence in sentence_list:
            vec_list.append(self.calculate_sentence_vector(sentence))

        pca=PCA()
        pca.fit(np.array(vec_list))
        u=pca.components_[0]
        u=np.multiply(u,np.transpose(u))

        if len(u)<self.embedding_size:
            for i in range(self.embedding_size-len(u)):
                u=np.append(u,0)

        sentence_vector=[]
        for vec in vec_list:
            sub=np.multiply(u,vec)
            sentence_vector.append(np.subtract(vec,sub))
        return sentence_vector

    def create_dataset_embeddingmatrix(self):
        df_train = pd.read_csv(Train_X, encoding='utf-8', names=['texts'])
        df_test =pd.read_csv(Test_X,encoding='utf-8',names=['texts'])
        print('There are total train {} samples'.format(len(df_train)))
        print('There are total test {} samples'.format(len(df_test)))

        df_train['texts_vector'] = df_train['texts'].apply(lambda x:self.create_sample_vector(x))
        df_test['texts_vector'] = df_test['texts'].apply(lambda x: self.create_sample_vector(x))

        df_train['sent_num_per_sample'] = df_train['texts'].apply(lambda x:self.count_sentence_num(x))
        df_test['sent_num_per_sample'] = df_test['texts'].apply(lambda x: self.count_sentence_num(x))
        max_enc_len=max(np.max(df_train['sent_num_per_sample']),np.max(df_test['sent_num_per_sample']))
        print('max_enc_len',max_enc_len)

        embedding_len=0
        for num in df_train['sent_num_per_sample']:
            embedding_len=embedding_len+num
        print('embedding_len',embedding_len)
        embedding_matrix =list(np.zeros((1,300)))
        for text_vector in df_train['texts_vector']:
            text_vector=text_vector[0]
            for matrix in text_vector:
                embedding_matrix.append(matrix)
        print('len(embedding_matrix)',len(embedding_matrix))
        for text_vector in df_test['texts_vector']:
            text_vector = text_vector[0]
            for matrix in text_vector:
                embedding_matrix.append(matrix)
        print('len(embedding_matrix)', len(embedding_matrix))
        np.save(sentence_embedding_matrix_path, embedding_matrix)

        sentence_x=np.zeros((len(df_train['texts']),max_enc_len))
        sum_train = 1
        for i,num in enumerate(list(df_train['sent_num_per_sample'])):
            for j in range(num):
                sentence_x[i,j]=sum_train+j
            sum_train=sum_train+num
        np.save(sentence_train_x,sentence_x)

        embedding_len = 0
        for num in df_test['sent_num_per_sample']:
            embedding_len = embedding_len + num
        print('embedding_len', embedding_len)


        sentence_x = np.zeros((len(df_test['texts']), max_enc_len))
        sum = sum_train
        for i, num in enumerate(list(df_test['sent_num_per_sample'])):
            for j in range(num):
                sentence_x[i, j] = sum + j
            sum = sum + num
        np.save(sentence_test_x, sentence_x)


    def create_sample_vector(self,sample):
        sentence_list=[]
        sample_vector=[]
        sentences = str(sample).split('|')
        for sentence in sentences:
            sentence_list.append(sentence)
        sentence_vector=self.sentences_to_vector(sentence_list)
        sample_vector.append(sentence_vector)
        return sample_vector

    def count_sentence_num(self,sample):
        sentences = str(sample).split('|')
        return len(sentences)



def create_sentences_binarylabel(max_sent_per_sample):
    stopwords = load_stopwords(stopwords_path)
    df = pd.read_csv(Train_X, encoding='utf-8', names=['texts'])
    df_new=df.copy()
    df_test=pd.read_csv(Train_y, encoding='utf-8', names=['Reports'])
    df_test_new=df_test.copy()

    df_test['Reports']=df_test['Reports'].apply(lambda x:x.strip().split())
    df_test['Reports'] = df_test['Reports'].apply(lambda x:[word for word in x if word not in stopwords])
    print(df_test['Reports'])

    df['texts'] = df['texts'].apply(lambda x: seperate_sample(x,remove_stopwords=True,only_sentence=True))
    print(df['texts'])

    label=np.zeros((len(df['texts']),max_sent_per_sample))

    for i,target_tokens in enumerate(df_test['Reports']):
        texts=list(df['texts'][i:i+1])[0]
        for j,text in enumerate(texts):
            print('text',text)
            coincide=set(text)& set(target_tokens)
            if len(coincide)<1:
                label[i,j]=float(0)
            else:
                label[i,j]=float(1)

    sum_label=[]
    for i in label:
        sum(i)
        sum_label.append(sum(i))
    sum_label=np.array(sum_label)

    nan_index=np.argwhere(sum_label==0.0)
    label=np.delete(label,nan_index,axis=0)
    print(len(label))
    np.save(sentence_train_y, label)

    nan=[]
    for i in nan_index:
        nan.append(int(i))
    df_new=df_new.drop(index=nan)
    df_new.reset_index()
    df_test_new=df_test_new.drop(index=nan)
    df_test_new.reset_index()
    df_new['texts'].to_csv(Train_X, index=None, header=False)
    df_test_new['Reports'].to_csv(Train_y,index=None,header=False)
    return label



def extract_sentence_based_on_label(train=False):
    if train:
        label = np.load(sentence_train_y + '.npy')
        df_train = pd.read_csv(Train_X, encoding='utf-8', names=['texts'])
        df_train['texts'] = df_train['texts'].apply(lambda x: seperate_sample(x,remove_stopwords=False,only_sentence=False))

        whole_train=[]
        train=[]
        for i in range(label.shape[0]):
            num=len(list(df_train['texts'][i:i+1])[0])
            index=np.argwhere(label[i][:num]==0)
            index=sorted(index,reverse=True)
            for j in index:
                df_train['texts'][i].pop(int(j))
            whole_train.append(df_train['texts'][i])
        for sample in whole_train:
            sample=' '.join(sample)
            train.append(sample)

        df = pd.DataFrame(train,columns=['texts'])
        df = df[df['texts'].notna()]
        df.to_csv(extractive_train_file,index=None,header=False,encoding='utf-8')


def seperate_sample(sample,remove_stopwords,only_sentence=True):
    if remove_stopwords:
        stopwords = load_stopwords(stopwords_path)
    sentence_list=[]
    sentences = str(sample).split('|')
    if only_sentence:
        sentence_list=sentences
    else:
        for sentence in sentences:
            words=sentence.strip().split()
            if remove_stopwords:
                words = [word for word in words if word not in stopwords]
            sentence_list.append(words)

    return sentence_list

