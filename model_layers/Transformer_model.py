import tensorflow as tf
import numpy as np


# Positional encoding

def get_angles(pos,i,d_model):
    return  pos / np.power(10000,(2*(i//2))/np.float32(d_model))


def positional_encoding(pos,d_model):
    angle_rads = get_angles(np.arange(pos)[:,np.newaxis],np.arange(d_model)[np.newaxis,:],d_model)
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2])
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2])
    pos_encoding = angle_rads[np.newaxis,...]
    return tf.cast(pos_encoding,dtype = tf.float32)


# Mask operation
def create_pad_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0),tf.float32)
    return seq[:,tf.newaxis,tf.newaxis,:] # (batch_size,1,1,seq_len)

def create_look_ahead_mask(size):
    mask = 1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
    return mask


# Multi_head attention

def scaled_dot_product_attention(q,k,v,mask):
    matmul_qk= tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        mask=tf.cast(mask,tf.float32)
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits,axis = -1)
    output = tf.matmul(attention_weights,v)
    return output,attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm =[0,2,1,3]) # (batch_size,num_heads,seq_len,depth)

    def call(self,q,k,v,mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q) # (batch_size,seq_len,d_model)
        k = self.wk(k) # (batch_size,seq_len,d_model)
        v = self.wv(v) # (batch_size,seq_len,d_model)

        q = self.split_heads(q,batch_size) # (batch_size,num_heads,seq_len_q,depth)
        k = self.split_heads(k,batch_size) # (batch_size,num_heads,seq_len_k,depth)
        v = self.split_heads(v,batch_size) # (batch_size,num_heads,seq_len_v,depth)

        scaled_attention,attention_weights = scaled_dot_product_attention(q,k,v,mask)
        # scaled_attention.shape == (batch_size,num_heads,seq_len_q,depth)
        # attention_weights.shape == (batch_size,num_heads,seq_len_q,seq_len_k)

        scaled_attention = tf.transpose(scaled_attention,perm = [0,2,1,3]) # (batch_size,seq_len_q,num_heads,depth)
        concat_attention = tf.reshape(scaled_attention,(batch_size,-1,self.d_model)) # (batch_size,seq_len_q,d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


# Point wise feed forward network

def point_wise_feed_forward_network(d_model,dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff,activation = 'relu'),tf.keras.layers.Dense(d_model)])


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self,epsilon=1e-6,scale=True,center=True):
        super(LayerNorm,self).__init__()
        self.epsilon=epsilon
        self.scale=scale
        self.center=center
        self.beta_initializer=tf.keras.initializers.get('zeros')
        self.gamma_initializer=tf.keras.initializers.get('ones')
        self.beta_regularizer=tf.keras.regularizers.get(None)
        self.gamma_regularizer=tf.keras.regularizers.get(None)
        self.beta_constraint=tf.keras.constraints.get(None)
        self.gamma_constraint=tf.keras.constraints.get(None)

    def build(self,input_shape):
        if self.scale:
            self.gamma=self.add_weight(name='gamma',shape=input_shape[-1],initializer=self.gamma_initializer,
                                       regularizer=self.gamma_initializer,constraint=self.gamma_constraint,trainable=True)
        else:
            self.gamma=None
        if self.center:
            self.beta=self.add_weight(name='beta',shape=input_shape[-1],initializer=self.beta_initializer,
                                      regularizer=self.beta_regularizer,constraint=self.beta_constraint,trainable=True)
        else:
            self.beta=None

    def call(self,inputs):
        input_shape=inputs.shape
        mean,variance=tf.nn.moments(x=inputs,axes=[-1],keepdims=True)
        return self.gamma * (inputs-mean)/(variance+self.epsilon) + self.beta



# Encoder layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate = 0.1):
        super(EncoderLayer,self).__init__()

        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.layernorm=LayerNorm(epsilon=1e-6,scale=True,center=True)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self,x,mask,training):
         attn_output,_ = self.mha(x,x,x,mask)
         attn_output = self.dropout1(attn_output,training=training)
         #out1=self.layernorm(x+attn_output)

         out1 = self.layernorm1(x+attn_output)


         ffn_output = self.ffn(out1)
         ffn_output = self.dropout2(ffn_output,training = training)
         out2 = self.layernorm2(out1 + ffn_output)

         return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,dff,rate=0.1):
        super(DecoderLayer,self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,look_ahead_mask,padding_mask,training):
        attn1,attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.dropout1(attn1,training=training)
        out1 = self.layernorm1(attn1+x)

        attn2,attn_weights_block2 = self.mha2(out1,enc_output,enc_output,padding_mask)
        attn2 = self.dropout2(attn2,training = training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output,training=training)
        out3 = self.layernorm3(ffn_output +out2)

        return out3,attn_weights_block1,attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,maximum_position_encoding,embedding_matrix,rate = 0.1,use_trained_embedding=False):
        super(Encoder,self).__init__()
        self.d_model = d_model
        self.num_alyers = num_layers
        self.use_trained_embedding=use_trained_embedding
        if use_trained_embedding:
            self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], self.d_model,
                                                       weights=[embedding_matrix], trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(input_vocab_size,d_model)

        self.pos_encoding = positional_encoding(maximum_position_encoding,self.d_model)
        self.enc_layers = [EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,x,mask,training):
        seq_len = tf.shape(x)[1]

        if self.use_trained_embedding:
            x = self.embedding(x)
        else:
            x = self.embedding(x)
            x *=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
            x+=self.pos_encoding[:,:seq_len,:]
            x=self.dropout(x,training=training)

        for i in range(self.num_alyers):
            x = self.enc_layers[i](x,mask,training)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,maximum_position_encoding,embedding_matrix,rate = 0.1,use_trained_embedding=False):
        super(Decoder,self).__init__()
        self.d_model = d_model
        self.num_alyers = num_layers
        self.use_trained_embedding = use_trained_embedding
        if use_trained_embedding:
            self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0], self.d_model,
                                                       weights=[embedding_matrix], trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)

        self.dec_layers = [DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self,x,enc_output,look_ahead_mask,padding_mask,training):
        seq_len = tf.shape(x)[1]
        attention_weights ={}
        if self.use_trained_embedding:
            x = self.embedding(x)
        else:
            x = self.embedding(x)
            x *=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
            x+=positional_encoding(seq_len,self.d_model)
            x=self.dropout(x,training=training)

        for i in range(self.num_alyers):
            x,block1,block2 = self.dec_layers[i](x,enc_output,look_ahead_mask,padding_mask,training)
            attention_weights['decoder_layer{}_block1'.format(i+1)]=block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x,attention_weights



class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,target_vocab_size, max_enc_len, max_dec_len, embedding_matrix,rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, max_enc_len, embedding_matrix,rate,use_trained_embedding=False)


        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, max_dec_len,embedding_matrix, rate,use_trained_embedding=False)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, enc_padding_mask,
             look_ahead_mask, dec_padding_mask,training):
        enc_output = self.encoder(inp, enc_padding_mask,training)  # (batch_size, inp_seq_len, d_model)


        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask,training)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


def create_masks(inp, tar):
    enc_padding_mask = create_pad_mask(inp)
    dec_padding_mask = create_pad_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1].numpy())
    dec_target_padding_mask = create_pad_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,d_model,warmup_steps):
        super(CustomSchedule,self).__init__()
        self.d_model=tf.cast(d_model,tf.float32)
        self.warmup_steps=warmup_steps

    def __call__(self,step):
        arg1=tf.math.rsqrt(step)
        arg2=step*(self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)

