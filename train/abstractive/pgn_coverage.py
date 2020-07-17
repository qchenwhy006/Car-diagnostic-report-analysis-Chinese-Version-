# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import tensorflow as tf
from model_layers.PGN_Coverage import PGN
from utils.params import get_params
from utils.data_loader import load_embedding_matrix,Vocab
from utils.batcher import Batcher,train_batch_generator
from model_layers.loss import pgn_loss_function,mask_coverage_loss
import numpy as np
import time


def train_model(model,params,vocab,checkpoint_manager):

    def train_step(inp, inp_ext, oov_len, dec_inp, dec_inp_ext, training):
        enc_pad_mask = tf.math.logical_not(tf.math.equal(inp, vocab.word2id['<PAD>'])) # enc_pad_mask.shape (batch_size,max_enc_len)
        padding_mask = tf.math.logical_not(tf.math.equal(dec_inp,vocab.word2id['<PAD>'])) # padding_mask.shape (batch_size,max_dec_len)

        if training:
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = model.call_encoder(inp)
                dec_hidden = enc_hidden
                predictions,_,attentions,coverages=model(enc_output,dec_hidden,enc_pad_mask,dec_inp,oov_len,inp_ext)
                # predictions.shape (batch_size,max_dec_len,vocab_size + max(batch_oov_len))
                # attentions.shape (max_dec_len+1,batch_size,max_enc_len)
                # coverages.shape (max_dec_len+2,batch_size,max_enc_len,1)

                loss = pgn_loss_function(dec_inp_ext,predictions,padding_mask)
                coverage_loss = mask_coverage_loss(attentions, coverages, padding_mask)
                batch_loss = loss + coverage_loss
                variables = model.encoder.trainable_variables + model.attention.trainable_variables + model.decoder.trainable_variables + model.pointer.trainable_variables
                gradients = tape.gradient(batch_loss,variables)
                optimizer = tf.keras.optimizers.Adam(params['learning_rate'])
                optimizer.apply_gradients(zip(gradients,variables))
        else:
            enc_output, enc_hidden = model.call_encoder(inp)
            dec_hidden = enc_hidden
            predictions, _, attentions, coverages = model(enc_output, dec_hidden, enc_pad_mask, dec_inp, oov_len,inp_ext)
            loss = pgn_loss_function(dec_inp_ext, predictions, padding_mask)
            coverage_loss = mask_coverage_loss(attentions, coverages, padding_mask)
            batch_loss = loss + coverage_loss

        return batch_loss,coverage_loss



    for epoch in range(params['epochs']):
        start = time.time()

        train_dataset, steps_per_epoch, dev_x, dev_y_inp, dev_y_real = train_batch_generator(params)

        for (batch, (inp, targ_inp, targ_real)) in enumerate(train_dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ_inp, targ_real, training=True)

            if batch % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch + 1,batch_loss.numpy()))

        batch_loss = train_step(dev_x, dev_y_inp, dev_y_real, training=False)


        #  saving (checkpoint) the model every 2 epochs
        if (epoch+1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,ckpt_save_path))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))






def train(params,embedding_matrix,vocab):
    print('Building the model ...')
    model=PGN(params,embedding_matrix,vocab)

    print('Creating the checkpoint manager ...')
    checkpoint=tf.train.Checkpoint(model=model)
    checkpoint_manager=tf.train.CheckpointManager(checkpoint,'drive/NLP1/data/checkpoints/training_pgn_checkpoints',max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print('Resotred from {}'.format(checkpoint_manager.latest_checkpoint))
    else:
        print('Initializing from scratch ...')

    print('Start the training process ...')
    train_model(model,params,vocab,checkpoint_manager)




if __name__ == '__main__':
    params=get_params()
    vocab = Vocab()
    embedding_matrix=load_embedding_matrix()
    train(params,embedding_matrix,vocab)