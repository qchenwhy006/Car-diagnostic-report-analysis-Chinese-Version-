# -*- coding:utf-8 -*-
# Created by Chen Qianqian
import tensorflow as tf
from utils.data_loader import load_embedding_matrix,Vocab
from model_layers import PGN_Coverage,beam_search
import pandas as pd
from utils.config import root,test_data_file
import os
from utils.params import get_params
from utils.config import checkpoint_dir
from utils.batcher import tokenize_data



def test(params,embedding_matrix,vocab):
    assert params['mode']=='test','change training mode to test mode'

    print('Building the model ...')
    model=PGN_Coverage.PGN(params,embedding_matrix,vocab)

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager =tf.train.CheckpointManager(checkpoint,checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")


    print("Creating the batcher ...")

    tokenized_test=tokenize_data(params,vocab)


    print('start prediction !')
    beam_decode = beam_search.Beam_Search(params,model,vocab)
    predictions=[]
    for i in range(len(tokenized_test['enc_input'])):
        predictions.append(beam_decode(tokenized_test[i:i+1]))


    for idx, result in enumerate(predictions):
        if result == '':
            print(idx)

    return predictions



def submit_proc(sentence):
    sentence=sentence.lstrip(' ，！。')
    sentence = sentence.replace(' ', '')
    if sentence=='':
        sentence='随时联系'
    return sentence





if __name__ == '__main__':
    params=get_params()
    vocab,reverse_vocab=Vocab()
    embedding_matrix=load_embedding_matrix()
    params['mode'] = 'test'
    predictions=test(params,embedding_matrix,vocab)

    test_df = pd.read_csv(test_data_file)
    test_df['Prediction'] = predictions
    test_df = test_df[['QID', 'Prediction']]
    test_df['Prediction'] = test_df['Prediction'].apply(submit_proc)
    test_df.to_csv(os.path.join(root, 'data', 'result_pgn_beam_search.csv'), index=None, sep=',')