# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import tensorflow as tf
from utils.data_loader import Vocab

def loss_function(real,pred,vocab):
    '''
    loss function for seq2seq model
    :param real: shape (batch_size,seq_len-1)
    :param pred: shape (batch_size,seq_len-1,vocab_size)
    :return:
    '''
    mask = tf.math.logical_not(tf.math.equal(real,vocab.word2id['<PAD>'])) # mask.shape (batch_size, seq_len-1)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    loss_ = loss_object(real,pred) # loss_.shape (batch_size, seq_len-1)
    mask = tf.cast(mask,dtype=loss_.dtype)
    dec_len = tf.reduce_sum(mask, axis=1)
    loss_*= mask
    loss_ = tf.reduce_sum(loss_,axis=1) / dec_len # loss_.shape (batch_size)
    return  tf.reduce_mean(loss_,axis=0) # batch-wise


def pgn_loss_function(dec_inp_ext,predictions,padding_mask):
    '''
    :param dec_inp_ext.shape (batch_size,max_dec_len)
    :param predictions.shape (batch_size,max_dec_len,vocab_size + max(batch_oov_len))
    :param padding_mask.shape (batch_size,max_dec_len)
    '''
    loss = 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    dec_len = tf.reduce_sum(tf.cast(padding_mask, dtype=tf.float32),axis=1)
    for t in range(dec_inp_ext.shape[1]):
        loss_ = loss_object(dec_inp_ext[:,t],predictions[:,t,:]) # loss_.shape (batch_size,)
        mask = tf.cast(padding_mask[:,t],dtype=loss_.dtype)
        loss_ *= mask
        loss += loss_

    dec_len = tf.cast(dec_len,dtype=loss.dtype)
    loss = tf.reduce_mean(loss / dec_len, axis=0)  # batch-wise

    return loss

def mask_coverage_loss(attentions, coverages, padding_mask):
    '''
    Calculates the coverage loss from the attention distributions.
    :param attentions.shape (max_dec_len+1,batch_size,max_enc_len)
    :param coverages.shape (max_dec_len+2,batch_size,max_enc_len,1)
    :param padding_mask.shape (batch_size,max_dec_len)
    '''
    cover_losses = []
    coverages = tf.squeeze(coverages, axis=3) # transfer coverages to [max_dec_len+2,batch_size,max_enc_len]
    attentions = tf.convert_to_tensor(attentions)

    for t in range(len(attentions[:-1])):
        cover_loss_ = tf.reduce_sum(tf.minimum(attentions[t,:,:],coverages[t,:,:]),axis=-1) # max_enc_len wise
        cover_losses.append(cover_loss_)

    cover_losses = tf.stack(cover_losses, 1) # change from[max_dec_len, batch_sz] to [batch_sz, max_dec_len]
    mask = tf.cast(padding_mask, dtype=cover_loss_.dtype)
    cover_losses *= mask
    loss = tf.reduce_sum(tf.reduce_mean(cover_losses, axis=0))  # mean loss of each time step and then sum up

    return loss











if __name__ == '__main__':
    vocab=Vocab()
