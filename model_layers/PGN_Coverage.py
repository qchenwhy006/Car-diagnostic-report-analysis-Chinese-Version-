# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import tensorflow as tf
from model_layers.layers import Encoder,BahdanauAttention,Decoder,Pointer
from utils.params import get_params
from utils.data_loader import load_embedding_matrix,Vocab
import random



class PGN(tf.keras.Model):
    def __init__(self,params,embedding_matrix,vocab):
        super(PGN, self).__init__()
        self.encoder = Encoder(params, embedding_matrix)
        self.attention = BahdanauAttention(params)
        self.decoder = Decoder(params, embedding_matrix)
        self.pointer = Pointer()
        self.batch_size = params['batch_size']
        self.max_enc_len = params['max_enc_len']
        self.use_scheduled_sampling = params['use_scheduled_sampling']
        self.vocab_size = params['vocab_size']
        self.mode = params['mode']
        self.vocab=vocab


    def call_encoder(self,inp):
        init_state = self.encoder.initial_state()
        output,state = self.encoder(inp,init_state)
        return output, state

    def call_decoder(self,dec_input,enc_output, dec_hidden, enc_pad_mask,prev_coverage,max_oov_len,extended_enc_inp):
        context_vector, attention_weight, prev_coverage = self.attention(enc_output, dec_hidden, enc_pad_mask,prev_coverage)
        prediction, dec_hidden = self.decoder(dec_input, dec_hidden, context_vector)
        print(prediction)
        p_gen = self.pointer(context_vector, dec_hidden, dec_input)
        final_dists = self.calc_final_dist(prediction, attention_weight, p_gen, max_oov_len, extended_enc_inp)
        final_dists = tf.stack(final_dists, 1)
        return final_dists


    def call(self,enc_output,dec_hidden,enc_pad_mask,dec_target,batch_oov_len,extended_enc_inp):
        predictions = []
        attentions = []
        p_gens = []
        coverages = [tf.zeros((self.batch_size, self.max_enc_len, 1))] # C0 = 0

        context_vector,attention_weights,coverage=self.attention(enc_output,dec_hidden,enc_pad_mask,prev_coverage=None)
        coverages.append(coverage)
        attentions.append(attention_weights)

        dec_input = tf.expand_dims([self.vocab.word2id['<START>']] * self.batch_size, axis=1)

        for t in range(dec_target.shape[1]): # dec_input with <STOP> and <PAD> tokens, without <START>
            prediction, dec_hidden = self.decoder(dec_input,dec_hidden,context_vector)
            p_gen = self.pointer(context_vector, dec_hidden, dec_input)

            predictions.append(prediction)
            p_gens.append(p_gen)

            context_vector, attention_weights, coverage=self.attention(enc_output,dec_hidden,enc_pad_mask,prev_coverage=coverage)
            coverages.append(coverage)
            attentions.append(attention_weights)


            if self.use_scheduled_sampling:
                cur_samp = random.uniform(0, 1)
                threshhold = tf.pow(0.95, t)  # schedule_type: "exponential"
                if cur_samp < threshhold:
                    dec_input = tf.expand_dims(dec_target[:, t], axis=1)
                else:
                    dec_input = tf.expand_dims(prediction, axis=1)
            else:
                dec_input = tf.expand_dims(dec_target[:, t],axis=1)  # Teacher forcing - feeding the target as the next input

        final_dists = self.calc_final_dist(predictions, attentions, p_gens, batch_oov_len, extended_enc_inp)


        if self.mode == 'train':
            return tf.stack(final_dists,axis=1), dec_hidden, attentions, coverages
        else:
            return tf.stack(final_dists,axis=1), dec_hidden, context_vector,tf.stack(attentions,1), tf.stack(p_gens,1)


    def calc_final_dist(self,predictions, attentions, p_gens, batch_oov_len, extended_enc_inp):
        '''
        Get the final distribution
        :param predictions.shape (max_dec_len,batch_size,vocab_size)
        :param attentions.shape (max_dec_len+1,batch_size,max_enc_len)
        :param p_gens.shape (max_dec_len,batch_size,1)
        :param batch_oov_len.shape (batch_size,)
        :param extended_enc_inp.shape (batch_size,max_enc_len)
        '''
        vocab_dists = [p_gen * predict for (p_gen,predict) in zip(p_gens,predictions)] # vocab_dists.shape (max_dec_len-1,batch_size,vocab_size)
        attn_dists = [(1-p_gen) * attn for (p_gen,attn) in zip (p_gens,attentions[:-1])]

        # Concatenate some zeros to each vocabulary dist,to hold the probabilities for in_articel OOV words
        batch_oov_len = tf.math.reduce_max(batch_oov_len,axis=-1).numpy()
        extended_size = self.vocab_size + batch_oov_len
        extra_zeros = tf.zeros((self.batch_size,batch_oov_len))
        vocab_dists_extended = [tf.concat([dist,extra_zeros],axis=1) for dist in vocab_dists] # vocab_dists_extended.shape (max_dec_len,batch_size,extended_size)

        # Project the values in the attention distributions onto the appropriate entries in the final distributions.
        batch_nums = tf.range(0,limit=self.batch_size)
        batch_nums = tf.expand_dims(batch_nums,axis=1) # batch_nums.shape (batch_size,1)
        attn_len = tf.shape(extended_enc_inp)[1]
        batch_nums = tf.tile(batch_nums,[1,attn_len]) # batch_nums.shape (batch_size,max_enc_len)
        indices = tf.stack((batch_nums,extended_enc_inp),axis=2) # indices.shape (batch_size,max_enc_len,2)
        shape = [self.batch_size,extended_size]
        attn_dists_projected = [tf.scatter_nd(indices,copy_dist,shape) for copy_dist in attn_dists]
        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists # final_dists.shape (max_dec_len,batch_size,extended_size)












if __name__ == '__main__':
    params=get_params()
    embedding_matrix=load_embedding_matrix()
    vocab=Vocab()
    pgn=PGN(params,embedding_matrix,vocab)