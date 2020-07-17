# -*- coding:utf-8 -*-
# Created by Chen Qianqian
import tensorflow as tf
from model_layers.Seq2Seq import Seq2seq
from model_layers.loss import loss_function
from utils.params import get_params
from utils.data_loader import load_vocab
from utils.batcher import train_batch_generator
import time




def train_model(model,params,vocab,checkpoint_manager):
    # learning_rate = CustomSchedule(params['d_model'], params['train_x_sample_num'] / params['batch_size'])
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])



    def train_step(input,target,training):
        # sentence = "SOS A lion in the jungle is sleeping EOS"
        # tar_inp = "SOS A lion in the jungle is sleeping"
        # tar_real = "A lion in the jungle is sleeping EOS"

        if training:
            with tf.GradientTape() as tape:
                enc_output,enc_hidden = model.call_encoder(input)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([vocab.word2id['<START>']] * params['batch_size'],axis=1)
                predictions = model(dec_input,dec_hidden,enc_output,target)
                batch_loss = loss_function(target[:,1:],predictions,vocab)

                variables = model.encoder.trainable_variables+model.attention.trainable_variables+model.decoder.trainable_variables
                gradients = tape.gradient(batch_loss,variables)
                optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                optimizer.apply_gradients(zip(gradients,variables))

        else:
            enc_output, enc_hidden = model.call_encoder(input)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([vocab.word2id['<START>']] * params['batch_size'], axis=1)
            predictions = model(dec_input, dec_hidden, enc_output, target)
            batch_loss = loss_function(target[:, 1:], predictions,vocab)

        return batch_loss

    loss_per_epoch = []
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
    model=Seq2seq(params,embedding_matrix)

    print('Creating the checkpoint manager ...')
    checkpoint=tf.train.Checkpoint(model=model)
    checkpoint_manager=tf.train.CheckpointManager(checkpoint,'drive/NLP1/data/checkpoints/training_seq2seq_checkpoints',max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print('Resotred from {}'.format(checkpoint_manager.latest_checkpoint))
    else:
        print('Initializing from scratch ...')

    print('Start the training process ...')
    train_model(model,params,vocab,checkpoint_manager)






if __name__ == '__main__':
    params=get_params()
    vocab,reversed_vocab=load_vocab()
    train(params,vocab)