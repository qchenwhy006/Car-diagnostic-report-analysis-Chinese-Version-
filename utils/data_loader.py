import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec,LineSentence
from utils.config import train_seg_file,test_seg_file,corpus_data_path,Train_X,Test_X,reversed_vocab_file,vocab_file,\
    embedding_matrix_path,test_X_path,train_X_path,train_Y_path,sentence_train_x,sentence_train_y,Train_y,sentence_test_x
import math
import tensorflow as tf



def build_word_embedding_matrix(train_data_path,test_data_path,pad_sentence=True):

    df_train = pd.read_csv(train_data_path, encoding='utf-8',names=['texts', 'reports'])
    df_test = pd.read_csv(test_data_path, encoding='utf-8',names=['QID','texts'])

    max_enc_len = max(get_max_len(df_train['texts']), get_max_len(df_test['texts']))
    max_dec_len = get_max_len(df_train['reports'])

    print('max_enc_len',max_enc_len)
    print('max_dec_len',max_dec_len)

    df = pd.DataFrame(columns=['Corpus'])
    df['Corpus'] = pd.concat([df_train['texts'], df_test['texts'], df_train['reports']], axis=0, ignore_index=True)
    df['Corpus'].to_csv(corpus_data_path, index=None, header=False, encoding='utf-8')


    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(corpus_data_path), size=300, workers=8, min_count=5, window=3, sg=1, iter=10)
    vocab = wv_model.wv.vocab

    if pad_sentence: # Fill the corpus with <START>,<STOP>,<UNK> and <PAD>
        df_train['reports'] = df_train['reports'].apply(lambda x: pad_process(x, max_dec_len))
        print(df_train['reports'])

        df_train['texts'] = df_train['texts'].apply(lambda x: UNK_process(x, vocab))
        df_test['texts'] = df_test['texts'].apply(lambda x: UNK_process(x, vocab))
        df_train['reports'] = df_train['reports'].apply(lambda x: UNK_process(x, vocab))
    else:
        df_train['texts'] = df_train['texts'].apply(lambda x:UNK_process(x,vocab))
        df_test['texts'] = df_test['texts'].apply(lambda x:UNK_process(x,vocab))
        df_train['reports'] = df_train['reports'].apply(lambda x:UNK_process(x,vocab))

    df_train['texts'].to_csv(Train_X, index=None, header=False)
    df_train['reports'].to_csv(Train_y, index=None, header=False)
    df_test=df_test[['QID','texts']]
    df_test.to_csv(Test_X, index=None, header=False)
    print(df_test)


    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(Train_X), update=True)
    wv_model.train(LineSentence(Train_X), epochs=1, total_examples=wv_model.corpus_count)
    print('1/3')
    wv_model.build_vocab(LineSentence(Train_y), update=True)
    wv_model.train(LineSentence(Train_y), epochs=1, total_examples=wv_model.corpus_count)
    print('2/3')
    wv_model.build_vocab(LineSentence(Test_X), update=True)
    wv_model.train(LineSentence(Test_X), epochs=1, total_examples=wv_model.corpus_count)

    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}

    save_dict(vocab_file, vocab)
    save_dict(reversed_vocab_file, reverse_vocab)

    embedding_matrix = wv_model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)



def get_max_len(dataframe):
    dataframe=dataframe.apply(lambda x: [str(word) for word in str(x).split()])
    num=dataframe.apply(lambda x: len(x))
    return int(np.max(num))
    # return int(np.mean(num)+2*np.std(num))

def pad_process(sentence,max_len):
    words = sentence.strip().split(' ')
    words = words[:min(max_len, len(words))]
    sentence = ['<START>']+words +['<END>']+['<PAD>'] * (max_len - len(words))
    return ' '.join(sentence)


def UNK_process(sentence,vocab):
    words=sentence.strip().split(' ')
    sentence=[word if word in vocab else '<UNK>' for word in words]
    return ' '.join(sentence)

def transform_into_index(sentence,vocab):
    words=sentence.strip().split(' ')
    idx=[vocab[word] if word in vocab else vocab['<UNK>'] for word in words]
    return idx

def save_dict(path,vocab):
    with open(path,'w+',encoding='utf-8') as f:
        for k,v in vocab.items():
            f.write('{}\t{}\n'.format(k,v))


def load_embedding_matrix(embedding_matrix_path):
    embedding_matrix=np.load(embedding_matrix_path+'.npy')
    print(embedding_matrix.shape)
    return embedding_matrix



def load_vocab(vocab_max_size=None):
    vocab = {}
    reverse_vocab = {}
    for line in open(vocab_file,'r',encoding='utf-8').readlines():
        word,index=line.strip().split("\t")
        index=int(index)
        # If the size of vocab is beyond the specific size, break the circulation.
        if vocab_max_size and index> vocab_max_size:
            print('max_size of vocab was specified as %i; we now have %i words. Stopping reading.'%(vocab_max_size,index))
            break
        vocab[word] = index
        reverse_vocab[index]=word
    return vocab,reverse_vocab


def load_dataset(max_enc_len=3125,max_dec_len=487):
    train_X=np.load(train_X_path+ '.npy')
    train_y=np.load(train_Y_path+ '.npy')
    test_X=np.load(test_X_path+ '.npy')

    train_X = train_X[:, :max_enc_len]
    train_y = train_y[:, :max_dec_len]
    test_X = test_X[:, :max_enc_len]

    return train_X,train_y,test_X

def load_train_dataset():
    train_X = np.load(sentence_train_x + '.npy')
    train_y = np.load(sentence_train_y+ '.npy')
    return train_X, train_y

def load_test_dataset():
    test_X = np.load(sentence_test_x + '.npy')
    return test_X



def train_batch_generator(params):
    X,y =load_train_dataset()

    indices=np.arange(len(X))
    np.random.shuffle(indices)
    X,y=X[indices],y[indices]

    train_X, train_y = X[:math.ceil(X.shape[0] * 0.999)], y[:math.ceil(X.shape[0] * 0.999)]
    dev_x, dev_y = X[math.ceil(X.shape[0] * 0.999):], y[math.ceil(X.shape[0] * 0.999):]


    dataset=tf.data.Dataset.from_tensor_slices((train_X,train_y))
    dataset=dataset.batch(params['batch_size'],drop_remainder=True)
    steps_per_epoch=len(train_X)//params['batch_size']
    if len(train_X) % params['batch_size']!=0:
        steps_per_epoch+=1
    return dataset,steps_per_epoch,dev_x, dev_y



if __name__ == '__main__':
    build_word_embedding_matrix(train_seg_file, test_seg_file,pad_sentence=False)