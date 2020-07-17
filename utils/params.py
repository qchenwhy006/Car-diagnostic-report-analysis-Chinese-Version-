import argparse

def get_params():
    parser=argparse.ArgumentParser()

    parser.add_argument('--bidirectional', default=True, help='whether it is bidirectional training model', type=bool)
    parser.add_argument('--GRU_unit', default=False, help='Encoder and Decoder unit appied to the model', type=bool)
    parser.add_argument('--LSTM_unit', default=True, help='Encoder and Decoder unit appied to the model', type=bool)
    parser.add_argument('--use_coverage', default=True, help='Whether use coverage mechanism', type=bool)
    parser.add_argument('--use_PGN', default=True, help='Whether use PGN mechanism', type=bool)
    parser.add_argument('--use_scheduled_sampling', default=False,
                        help='Whether use scheduled sampling or teacher forcing sampling', type=bool)

    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle", type=bool)

    parser.add_argument('--mode',default='test',help='run mode',type=str)
    parser.add_argument('--max_enc_len',default=198,help='Encoder input max sequence length',type=int)
    parser.add_argument('--max_dec_len',default=52,help='Decoder input max sequence length',type=int)
    parser.add_argument('--batch_size',default=16,help='batch size',type=int)
    parser.add_argument("--epochs", default=10, help="train epochs", type=int)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--vocab_size", default=17002, help="max vocab size , None-> Max ", type=int)
    parser.add_argument("--beam_size", default=3,help="beam size for beam search decoding ",type=int)
    parser.add_argument("--embedding_dim", default=128, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=128, help="Encoder cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward \
                                result dimension - this result is used to compute the attention weights",type=int)

    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)
    parser.add_argument("--max_train_steps", default=1250, help="max_train_steps", type=int)

    parser.add_argument('--num_layers',default=4,help='the number of layers in Transformer Encoder',type=int)
    parser.add_argument('--dff',default=512,help='the unit in feed_forward neural network in Transformer',type=int)
    parser.add_argument('--num_heads',default=6,help='the number of head in Multi_head_attention layer in Transformer',type=int)
    parser.add_argument('--rate',default=0.1,help='the ratio on dropout layer in Transformer',type=float)
    parser.add_argument('--d_model',default=300,help='the unit in query,key,value transform matrix ',type=int)
    parser.add_argument('--train_x_sample_num',default=59494,help='the sample number of train_x except the dev data',type=int)

    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. "
                             "Please refer to the Adagrad optimizer API documentation "
                             "on tensorflow site for more details.",
                        type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)

    args=parser.parse_args()
    params=vars(args)

    return params




if __name__ == '__main__':
    params=get_params()
