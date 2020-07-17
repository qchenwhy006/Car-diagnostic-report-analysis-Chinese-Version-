# -*- coding:utf-8 -*-
# Created by Chen Qianqian
import tensorflow as tf
from utils.data_loader import Vocab
from utils.params import get_params



class Beam_Search(tf.keras.layers.Layer):
    def __init__(self,params,model,vocab):
        super(Beam_Search,self).__init__()
        self.params=params
        self.model=model
        self.start_index=vocab.word2id['<START>']
        self.stop_index=vocab.word2id['<STOP>']



    def get_top_k_for_one_step(self, enc_output, dec_input, dec_hidden):
        predictions,dec_hidden=self.model.call_decoder(enc_output,dec_input,dec_hidden)
        top_k_probs, top_k_ids = tf.nn.top_k(predictions, k=self.params['beam_size'], sorted=True)
        top_k_log_probs = tf.math.log(top_k_probs)

        return top_k_log_probs, top_k_ids, dec_hidden


    def beam_search_for_single_item(self,enc_output,dec_hidden):

        dec_input=tf.expand_dims(tf.convert_to_tensor([self.start_index]),axis=0)


        prediction,dec_hidden=self.model.call_decoder(enc_output,dec_input,dec_hidden) # prediction.shape (1,vocab_size),dec_hidden.shape (1,dec_units)
        top_k_probs, top_k_ids=tf.nn.top_k(prediction, k=params['beam_size'], sorted=True)
        top_k_log_probs=tf.math.log(top_k_probs)
        cur_candidates = []

        for i in range(self.params['beam_size']):
            cur_token_list=[self.start_index]+[top_k_ids[0,i].numpy()]
            cur_tot_log_prob=0+top_k_log_probs[0,i].numpy()
            cur_dec_hidden=dec_hidden
            cur_candidates.append((cur_tot_log_prob, cur_token_list,cur_dec_hidden))
        candidates=cur_candidates

        enc_output = tf.tile(enc_output, [3, 1, 1])


        for i in range(2, self.params['max_dec_step']):

            latest_token=[]
            temp_dec_hidden=[]
            for item in candidates:
                latest_token.append(item[1][-1])
                temp_dec_hidden.append(item[2])

            dec_hidden=tf.squeeze(tf.convert_to_tensor(temp_dec_hidden),[1])
            dec_input=tf.expand_dims(latest_token,axis=1)


            top_k_log_probs, top_k_ids, dec_hidden=self.get_top_k_for_one_step(enc_output,dec_input,dec_hidden)

            cur_candidates = []
            for i in range(len(candidates)):
                cur_item=candidates[i]
                cur_dec_hidden=dec_hidden[i:i+1]
                for j in range(self.params['beam_size']):
                    cur_token_list = cur_item[1] + [top_k_ids[i, j].numpy()]
                    cur_tot_log_prob = cur_item[0] + top_k_log_probs[i, j].numpy()
                    cur_candidates.append((cur_tot_log_prob, cur_token_list,cur_dec_hidden))

            candidates = []
            sorted_candidates = sorted(cur_candidates, key=lambda x: x[0], reverse=True)

            for candidate in sorted_candidates:
                candidates.append(candidate)
                if len(candidates)==self.params['beam_size']:
                    break


        result = []
        for candidate in candidates:
            if candidate[1][-1] == self.stop_index:
                if len(candidate[1]) >= self.min_steps:
                    result.append(candidate)  # 有结束符且满足最小长度要求
                    break
            else:
                result.append(candidate)
                break

        return ' '.join([self.reverse_vocab[index] for index in result[0][1]])



    def call(self,batch):


        enc_output,enc_hidden=self.model.call_encoder_for_beam_search(batch['enc_input']) #  enc_output.shape (1,seq_len,enc_units)

        dec_hidden=enc_hidden

        result=self.beam_search_for_single_item(enc_output,dec_hidden)

        print(result)
        return result



if __name__ == '__main__':
    vocab=Vocab()
    params=get_params()