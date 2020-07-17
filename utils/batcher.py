import tensorflow as tf
from utils.config import train_x_path, train_y_inp_path, train_y_real_path, test_x_path
from utils.params import get_params
import numpy as np
import math
from utils.data_loader import CarQuestionAnswer

def Batcher(vocab,params):
    if params['mode']=='train':
        tokenized_result = tokenize_data(params, vocab)

        tokenized_result['enc_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['enc_input'],
                                                                                      maxlen=params['max_enc_len'],
                                                                                      padding='post',
                                                                                      value=vocab.word2id['<PAD>'])
        tokenized_result['extended_enc_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['extended_enc_input'],
                                                                                               maxlen=params['max_enc_len'],
                                                                                               padding='post',
                                                                                               value=vocab.word2id['<PAD>'])

        tokenized_result['dec_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['dec_input'],
                                                                                      maxlen=params['max_dec_len'],
                                                                                      padding='post',
                                                                                      value=vocab.word2id['<PAD>'])
        tokenized_result['extended_dec_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['extended_dec_input'],
                                                                                      maxlen=params['max_dec_len'],
                                                                                      padding='post',
                                                                                      value=vocab.word2id['<PAD>'])

        dataset=tf.data.Dataset.from_tensor_slices((tokenized_result["enc_input"],
                                    tokenized_result["extended_enc_input"],
                                    tokenized_result["max_oov_len"],
                                    tokenized_result["dec_input"],
                                    tokenized_result['extended_dec_input'])).shuffle(params['batch_size'])
        dataset=dataset.batch(params['batch_size'],drop_remainder=True)

        steps_per_epoch = len(tokenized_result['enc_input']) // params['batch_size']
    else:
        tokenized_result = tokenize_data(params, vocab)

        tokenized_result['enc_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['enc_input'],
                                                                                      maxlen=params['max_enc_len'],
                                                                                      padding='post',
                                                                                      value=vocab.word2id['<PAD>'])
        tokenized_result['extended_enc_input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['extended_enc_input'],
                                                                                               maxlen=params['max_enc_len'],
                                                                                               padding='post',
                                                                                               value=vocab.word2id['<PAD>'])

        dataset = tf.data.Dataset.from_tensor_slices((tokenized_result["enc_input"],
                                                     tokenized_result["extended_enc_input"],
                                                     tokenized_result["max_oov_len"]))
        dataset = dataset.batch(params['batch_size'], drop_remainder=True)
        steps_per_epoch = len(tokenized_result['enc_input']) // params['batch_size']


    return dataset,steps_per_epoch



def tokenize_data(params,vocab):
    result={
        'enc_input':[],
        'extended_enc_input':[],
        'max_oov_len':[],
        'dec_input':[],
        'extended_dec_input':[]}

    if params['mode']=='train':
        train_x=pd.read_csv(params['train_seg_x_dir'],encoding='utf-8',squeeze=True,header=None)
        train_y=pd.read_csv(params['train_seg_y_dir'],encoding='utf-8',squeeze=True,header=None)

        for input,output in zip(train_x,train_y):
            tokenized_x,extended_x,oov_list=tokenize_one_sentence(input,vocab,None)
            tokenized_y,extended_y,_=tokenize_one_sentence(output,vocab,oov_list)
            tokenized_y[-1]=vocab.word2id['<STOP>']
            extended_y[-1]=vocab.word2id['<STOP>']


            result['enc_input'].append(tokenized_x)
            result['extended_enc_input'].append(extended_x)
            result['max_oov_len'].append(len(oov_list))
            result['dec_input'].append(tokenized_y)
            result['extended_dec_input'].append(extended_y)

    else:
        test_x = pd.read_csv(params['test_seg_x_dir'], encoding='utf-8', squeeze=True, header=None)
        for input in test_x:
            tokenized_x,extended_x,oov_list=tokenize_one_sentence(input,vocab,None)
            result['enc_input'].append(tokenized_x)
            result['extended_enc_input'].append(extended_x)
            result['max_oov_len'].append(len(oov_list))

    return result



def tokenize_one_sentence(sentence,vocab,extended_vocab):
    idx=[]
    extended_idx=[]
    oovs=[]   # only used when oovs is None
    sentence=sentence.split()


    for word in sentence:
        ids = vocab.word_to_id(word)
        idx.append(ids)
        if ids==vocab.word2id['<UNK>']:
            if extended_vocab is None: # for article
                if word not in oovs:
                    oovs.append(word)
                    oov_num = oovs.index(word)
                    extended_idx.append(vocab.size() + oov_num)
                else:
                    oov_num = oovs.index(word)
                    extended_idx.append(vocab.size() + oov_num)

            else: # for abstract
                if word in extended_vocab:
                    oov_num=extended_vocab.index(word)
                    extended_idx.append(vocab.size() + oov_num)
                else:
                    extended_idx.append(vocab.word2id['<UNK>'])
        else:
            extended_idx.append(ids)
    return idx,extended_idx, oovs


def train_batch_generator(params):
    data = CarQuestionAnswer(max_enc_lens=512, max_dec_lens=512)
    X, train_y_inp, train_y_real = data.train_x, data.train_y_inp, data.train_y_real

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    train_Y_inp, train_Y_real = train_y_inp[indices], train_y_real[indices]

    train_X, train_y_inp, train_y_real = X[:math.ceil(X.shape[0] * 0.999)], train_Y_inp[:math.ceil(
        X.shape[0] * 0.999)], train_Y_real[:math.ceil(X.shape[0] * 0.999)]
    dev_x, dev_y_inp, dev_y_real = X[math.ceil(X.shape[0] * 0.999):], train_Y_inp[math.ceil(X.shape[0] * 0.999):], train_Y_real[math.ceil(X.shape[0] * 0.999):]

    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y_inp, train_y_real))
    dataset = dataset.batch(params['batch_size'], drop_remainder=True)
    steps_per_epoch = len(train_X) // params['batch_size']
    if len(train_X) % params['batch_size'] != 0:
        steps_per_epoch += 1

    return dataset, steps_per_epoch, dev_x, dev_y_inp, dev_y_real


def load_train_dataset(params):
    train_X = np.load(train_x_path + '.npy')
    train_y_inp = np.load(train_y_inp_path + '.npy')
    train_y_real = np.load(train_y_real_path + '.npy')

    train_X = train_X[:, :params['max_enc_len']]
    train_y_inp = train_y_inp[:, :params['max_dec_len']]
    train_y_real = train_y_real[:, :params['max_dec_len']]

    return train_X, train_y_inp, train_y_real


def load_test_dataset(params):
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :params['max_enc_len']]

    return test_X


if __name__ == '__main__':
    params = get_params()
    dataset, steps_per_epoch, dev_x, dev_y_inp, dev_y_real = train_batch_generator(params)