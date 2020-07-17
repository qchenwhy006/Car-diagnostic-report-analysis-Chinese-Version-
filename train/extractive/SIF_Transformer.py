from model_layers.Transformer_model import positional_encoding,EncoderLayer,create_pad_mask
import tensorflow as tf
from utils.metrics import micro_f1,macro_f1
from utils.data_loader import load_embedding_matrix,load_train_dataset,load_test_dataset
from utils.config import sentence_embedding_matrix_path,root
import os
import numpy as np

def train():
    train_X, train_y = load_train_dataset()
    train_X_padding_mask = create_pad_mask(train_X)
    print('train_y.shape',train_y.shape)


    output_dim = train_y.shape[1]
    print('output_dim',output_dim)
    max_seq_len=train_X.shape[1]
    print('max_seq_len',max_seq_len)
    embedding_matrix = load_embedding_matrix(sentence_embedding_matrix_path)
    model = Transformer_for_Classification(embedding_matrix,max_enc_len=max_seq_len, output_dim=output_dim)
    model.fit(x=(train_X,train_X_padding_mask), y=train_y, batch_size=16,epochs=6,shuffle=True,
              validation_split=0.001)

    model.save_weights(os.path.join(root, 'data', 'Extractive','BaiduQuestion_SIF_transformer.h5'), overwrite=True)

def inference():
    test_X=load_test_dataset()
    test_X_padding_mask=create_pad_mask(test_X)
    print('test_X.shape',test_X.shape)

    output_dim=test_X.shape[1]
    print('output_dim', output_dim)
    max_seq_len=test_X.shape[1]
    print('max_seq_len', max_seq_len)
    embedding_matrix = load_embedding_matrix(sentence_embedding_matrix_path)
    model = Transformer_for_Classification(embedding_matrix,max_enc_len=max_seq_len, output_dim=output_dim)
    model.load_weights(os.path.join(root, 'data', 'Extractive','BaiduQuestion_SIF_transformer.h5'))

    pred_y = model.predict((test_X, test_X_padding_mask))
    pred_y = np.where(pred_y > 0.5, 1, 0)
    np.save(os.path.join(root,'data','result','SIF_test'), pred_y)



def Transformer_for_Classification(embedding_matrix,max_enc_len,output_dim):
    encoder=Encoder(num_layers=6,num_heads=6,dff=1200,embedding_matrix=embedding_matrix,max_enc_len=157,rate=0.1)

    input_ids=tf.keras.layers.Input(shape=(max_enc_len,),dtype='int32')
    enc_padding_mask=tf.keras.layers.Input(shape=(max_enc_len,),dtype='int32')
    encoder_output=encoder(input_ids,enc_padding_mask,training=True)
    flatten_output = tf.keras.layers.Flatten()(encoder_output)
    final_output=tf.keras.layers.Dense(output_dim,activation='sigmoid')(flatten_output)
    model=tf.keras.Model(inputs=[input_ids,enc_padding_mask],outputs=final_output)
    model.build(input_shape=[(None,max_enc_len),(None,max_enc_len)])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=[micro_f1, macro_f1])
    model.summary()
    return model


class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_heads,dff,embedding_matrix,max_enc_len,rate = 0.1):
        super(Encoder,self).__init__()
        self.num_alyers = num_layers
        self.d_model=embedding_matrix.shape[1]
        self.embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0],self.d_model,
                                                   weights=[embedding_matrix], trainable=False)
        self.pos_encoding = positional_encoding(max_enc_len,self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model,num_heads,dff,rate) for _ in range(num_layers)]


    def call(self,x,mask,training):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x+=self.pos_encoding[:,:seq_len,:]

        for i in range(self.num_alyers):
            x = self.enc_layers[i](x,mask,training)

        return x


if __name__ == '__main__':
    train()
    inference()