# -*- coding:utf-8 -*-
# Created by Chen Qianqian

import tensorflow as tf
from utils.params import get_params
from utils.data_loader import load_embedding_matrix,load_train_dataset


class Encoder(tf.keras.Model):
    def __init__(self,params,embedding_matrix):
        super(Encoder,self).__init__()
        self.embedding=tf.keras.layers.Embedding(params['vocab_size'],params['embedding_dim'],weights=[embedding_matrix],trainable=False)
        self.GRU_unit=params['GRU_unit']
        self.LSTM_unit=params['LSTM_unit']
        self.bidirectional=params['bidirectional']
        self.batch_size=params['batch_size']
        self.enc_units=params['enc_units']

        if self.GRU_unit:
            self.rnn=tf.keras.layers.GRU(params['enc_units'],return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

        if self.LSTM_unit:
            self.rnn=tf.keras.layers.LSTM(params['enc_units'],return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')

        if self.bidirectional:
            self.rnn=tf.keras.layers.Bidirectional(self.rnn,merge_mode='concat')


    def call(self,x,init_state):
        x=self.embedding(x) # After embedding layer, x.shape (batch_size,seq_len,embedding_dim)

        if self.bidirectional:
            if self.GRU_unit:
                init_state=tf.split(init_state,num_or_size_splits=2,axis=1)
                output,state_forward,state_backward=self.rnn(x,initial_state=init_state)
                state = tf.concat([state_forward,state_backward], axis=1)

            else:
                init_state = tf.split(init_state, num_or_size_splits=4, axis=1)
                output,h_forward,c_forward,h_backward,c_backward = self.rnn(x, initial_state=init_state)
                h_state=tf.concat([h_forward,h_backward],axis=1)
                c_state=tf.concat([c_forward,c_backward],axis=1)
                state=tf.concat([h_state,c_state],axis=1)
        else:
            if self.GRU_unit:
                output,state=self.rnn(x,initial_state=init_state)
            else:
                init_state = tf.split(init_state, num_or_size_splits=2, axis=1)
                output,h_state,c_state=self.rnn(x,initial_state=init_state)
                state=tf.concat([h_state,c_state],axis=1)

        return output,state

    def initial_state(self):
        init_state=tf.zeros((self.batch_size,self.enc_units))

        if self.bidirectional:
            if self.GRU_unit:
                init_state=tf.concat([init_state,init_state],axis=1)
            else:
                init_state=tf.concat([init_state,init_state,init_state,init_state],axis=1)
        else:
            if self.LSTM_unit:
                init_state = tf.concat([init_state, init_state], axis=1)
            else:
                init_state = init_state
        return init_state


class BahdanauAttention(tf.keras.Model):
    def __init__(self,params):
        super(BahdanauAttention,self).__init__()
        self.W1=tf.keras.layers.Dense(params['attn_units'])
        self.W2=tf.keras.layers.Dense(params['attn_units'])
        self.V=tf.keras.layers.Dense(1)
        self.W3=tf.keras.layers.Dense(params['attn_units'])
        self.use_coverage=params['use_coverage']


    def call(self,enc_output,dec_hidden,enc_pad_mask,prev_coverage):
        # For Bi_LSTM, enc_output.shape (batch_size,seq_len,2*enc_units)
        if self.use_coverage and prev_coverage is not None:
            dec_hidden_with_time_axis=tf.expand_dims(dec_hidden,axis=1)
            score=self.V(tf.nn.tanh(self.W1(enc_output)+self.W2(dec_hidden_with_time_axis)+self.W3(prev_coverage))) # score.shape (batch_size,seq_len,1)

            mask=tf.cast(enc_pad_mask,dtype=score.dtype) # mask.shape (batch_size,seq_len)
            masked_score=tf.squeeze(score,axis=-1) * mask
            masked_score=tf.expand_dims(masked_score,axis=2)

            attention_weights=tf.nn.softmax(masked_score,axis=1)
            coverage = attention_weights + prev_coverage # coverage.shape (batch_size,seq_len,1)

            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)

        elif enc_pad_mask is not None:
            dec_hidden_with_time_axis=tf.expand_dims(dec_hidden,axis=1)
            score=self.V(tf.nn.tanh(self.W1(enc_output)+self.W2(dec_hidden_with_time_axis))) # score.shape (batch_size,seq_len,1)

            mask=tf.cast(enc_pad_mask,dtype=score.dtype)
            masked_score=tf.squeeze(score,axis=-1)*mask
            masked_score=tf.expand_dims(masked_score,axis=2)

            attention_weights=tf.nn.softmax(masked_score,axis=1) # attention_weights.shape (batch_size,seq_len,1)

            if self.use_coverage:
                coverage=attention_weights

            context_vector = attention_weights * enc_output
            context_vector = tf.reduce_sum(context_vector, axis=1)


        else:
            dec_hidden_with_time_axis = tf.expand_dims(dec_hidden, axis=1)
            score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden_with_time_axis)))  # score.shape (batch_size,seq_len,1)

            attention_weights=tf.nn.softmax(score,axis=1) # attention_weights.shape (batch_size,seq_len,1)
            context_vector = attention_weights * enc_output  # context_vector.shape (batch_size,seq_len,2*enc_units)
            context_vector = tf.reduce_sum(context_vector,axis=1) # context_vector.shape (batch_size,2*enc_units)
            coverage=None

        return context_vector,tf.squeeze(attention_weights,axis=-1),coverage


class Decoder(tf.keras.Model):
    def __init__(self,params,embedding_matrix):
        super(Decoder,self).__init__()
        self.embedding=tf.keras.layers.Embedding(params['vocab_size'],params['embedding_dim'],weights=[embedding_matrix],trainable=False)
        self.GRU_unit = params['GRU_unit']
        self.LSTM_unit = params['LSTM_unit']

        if self.GRU_unit:
            self.rnn=tf.keras.layers.GRU(params['dec_units'],recurrent_initializer='glorot_uniform',return_sequences=True,return_state=True)

        if self.LSTM_unit:
            self.rnn=tf.keras.layers.LSTM(params['dec_units'],recurrent_initializer='glorot_uniform',return_sequences=True,return_state=True)

        self.use_PGN=params['use_PGN']

        if self.use_PGN:
            self.dense=tf.keras.layers.Dense(params['vocab_size'],activation=tf.keras.activations.softmax)
        else:
            self.dense=tf.keras.layers.Dense(params['vocab_size'])

    def call(self,x,dec_hidden,context_vector):
        x=self.embedding(x) # x.shape (batch_size,1,embedding_dim)

        if self.use_PGN:
            if self.LSTM_unit:
                dec_hidden=tf.split(dec_hidden,num_or_size_splits=2,axis=1)
                output,h_state,c_state=self.rnn(x,initial_state=dec_hidden)
                state=tf.concat([h_state,c_state],axis=1)
            else:
                output,state=self.rnn(x,initial_state=dec_hidden)

            output=tf.concat([tf.expand_dims(context_vector,axis=1),output],axis=-1)
            # output shape after concatenation == (batch_size, 1, hidden_size + hidden_size)

            output=tf.reshape(output,shape=(-1,output.shape[2])) # output.shape (batch_size * 1,  hidden_size + hidden_size)
            prediction=self.dense(output) # prediction.shape (batch_size,vocab_size)

        else:
            x=tf.concat([tf.expand_dims(context_vector,axis=1),x],axis=-1) # x.shape (batch_size,1,embedding_dim+hidden_size)

            if self.LSTM_unit:
                dec_hidden=tf.split(dec_hidden,num_or_size_splits=2,axis=1)
                output,h_state,c_state=self.rnn(x,initial_state=dec_hidden)
                state=tf.concat([h_state,c_state],axis=1)
            else:
                output,state=self.rnn(x,initial_state=dec_hidden)

            output=tf.reshape(output,shape=(-1,output.shape[2])) # output.shape (batch_size * 1,  hidden_size )
            prediction=self.dense(output)  # prediction.shape (batch_size,vocab_size)

        return prediction,state


class Pointer(tf.keras.layers.Layer):
    def __init__(self):
        super(Pointer,self).__init__()
        self.W1=tf.keras.layers.Dense(1)
        self.W2=tf.keras.layers.Dense(1)
        self.W3=tf.keras.layers.Dense(1)

    def call(self,context_vector,dec_hidden,dec_input):
        '''
        calculate the Pgen value
        :param context_vector: shape (batch_size,2*enc_units) Because it is Bi_LSTM encoder
        :param dec_hidden: shape (batch_size,dec_units)
        :param dec_input: shape (batch_size,1)
        '''
        return tf.nn.sigmoid(self.W1(context_vector) + self.W2(dec_hidden) + self.W3(dec_input)) # shape (batch_size,1)





if __name__ == '__main__':
    params=get_params()
    embedding_matrix=load_embedding_matrix()
    encoder=Encoder(params,embedding_matrix)

    train_x,train_y=load_train_dataset()
    x=train_x[:params['batch_size']]
    init_state=encoder.initial_state()
    output,state=encoder(x,init_state)
    print('encoder output shape:{}, encoder hidden state shape:{}'.format(output.shape,state.shape))
    # For Bi_LSTM, encoder output shape:(16, 353, 256), encoder hidden state shape:(16, 512)

    attention=BahdanauAttention(params)
    dec_hidden=tf.ones((params['batch_size'],params['dec_units']))
    context_vector,attention_weight,_=attention(output,dec_hidden,prev_coverage=None,enc_pad_mask=None)
    print('context_vector shape:{}, attention_weights shape:{}'.format(context_vector.shape,attention_weight.shape))

    decoder=Decoder(params,embedding_matrix)
    y=tf.ones((params['batch_size'],1))
    prediction,dec_state=decoder(y,state,context_vector)
    print('prediction shape:{}, dec hidden state shape:{}'.format(prediction.shape,dec_state.shape))

    pointer=Pointer()
    Pgen=pointer(context_vector,dec_hidden,y)
    print('Pgen shape:',Pgen.shape)
