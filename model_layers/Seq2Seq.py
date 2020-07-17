# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import tensorflow as tf
from model_layers.layers import Encoder,BahdanauAttention,Decoder
import random

class Seq2seq(tf.keras.Model):
    def __init__(self,params,embedding_matrix):
        super(Seq2seq, self).__init__()
        self.encoder=Encoder(params,embedding_matrix)
        self.attention=BahdanauAttention(params)
        self.decoder=Decoder(params,embedding_matrix)
        self.use_scheduled_sampling=params['use_scheduled_sampling']

    def call_encoder(self,inp):
        init_state=self.encoder.initial_state()
        output,state=self.encoder(inp,init_state)
        return output,state



    def call(self,dec_input,dec_hidden,enc_output,target): # target with <START>, <STOP> and <PAD> tokens
        predictions=[]

        context_vector,_,_=self.attention(enc_output,dec_hidden,enc_pad_mask=None,prev_coverage=None)

        for t in range(1,target.shape[1]):
            prediction,dec_hidden=self.decoder(dec_input,dec_hidden,context_vector)
            predictions.append(prediction)

            context_vector,_,_=self.attention(enc_output,dec_hidden,enc_pad_mask=None,prev_coverage=None)

            if self.use_scheduled_sampling:
                cur_samp = random.uniform(0,1)
                threshhold = tf.pow(0.95,t) # schedule_type: "exponential"
                if cur_samp < threshhold:
                    dec_input = tf.expand_dims(target[:, t], axis=1)
                else:
                    dec_input = tf.expand_dims(prediction,axis=1)
            else:
                dec_input = tf.expand_dims(target[:,t],axis=1)  # Teacher forcing - feeding the target as the next input


        return tf.stack(predictions,axis=1) # (batch_size,seq_len-1,vocab_size)