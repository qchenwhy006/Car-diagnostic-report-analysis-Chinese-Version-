# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import pathlib
import os

root=pathlib.Path(os.path.abspath(__file__)).parent.parent

train_data_file=os.path.join(root,'data','AutoMaster_TrainSet.txt')

test_data_file=os.path.join(root,'data','AutoMaster_TestSet.txt')

train_seg_file=os.path.join(root,'data','clean_train.csv')

test_seg_file=os.path.join(root,'data','clean_test.csv')

stopwords_path=os.path.join(root,'data','stopwords.txt')

stopwords=os.path.join(root,'data','stopwords1.txt')

# user_dict path

user_dict=os.path.join(root,'data','user_dict.txt')

# save model
checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_transformer_checkpoints_after_removing_stopwords')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

save_result_dir = os.path.join(root, 'result')

# Extractive method

corpus_data_path=os.path.join(root,'data','Extractive','corpus_data.txt')

Train_X=os.path.join(root,'data','Extractive','train_X.csv')

Test_X=os.path.join(root,'data','Extractive','test_X.csv')

Train_y=os.path.join(root,'data','Extractive','Train_y.csv')

vocab_file=os.path.join(root,'data','Extractive','vocab.txt')

reversed_vocab_file=os.path.join(root,'data','Extractive','reversed_vocab.txt')

embedding_matrix_path=os.path.join(root,'data','Extractive','embedding_matrix')

sentence_embedding_matrix_path=os.path.join(root,'data','Extractive','sentence_embedding_matrix')

test_sentence_embedding_matrix_path=os.path.join(root,'data','Extractive','test_sentence_embedding_matrix')

sentence_train_x=os.path.join(root,'data','Extractive','sentence_train_x')

sentence_train_y=os.path.join(root,'data','Extractive','sentence_train_y')

sentence_test_x=os.path.join(root,'data','Extractive','sentence_test_x')

train_X_path=os.path.join(root,'data','Extractive','train_X')

train_Y_path=os.path.join(root,'data','Extractive','train_Y')

test_X_path=os.path.join(root,'data','Extractive','test_X')

extractive_train_file=os.path.join(root,'data','Extractive','extractive_clean_train.csv')



# .npy form
train_x_path=os.path.join(root,'data','train_x')

train_y_real_path=os.path.join(root,'data','train_y_real')

train_y_inp_path=os.path.join(root,'data','train_y_inp')

dev_x_path=os.path.join(root,'data','dev_x')

dev_y_path=os.path.join(root,'data','dev_y')

test_x_path=os.path.join(root,'data','test_x')



# Bert document
bert_dir=os.path.join(root,'data','bert','chinese_L-12_H-768_A-12')

bert_vocab_file=os.path.join(bert_dir,'vocab.txt')

bert_config_file=os.path.join(bert_dir,'bert_config.json')

bert_ckpt_file=os.path.join(bert_dir,'bert_model.ckpt')




