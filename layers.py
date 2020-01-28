import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def _init_(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units // 2
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2 * self.enc_units))


class Attention(tf.keras.layers.Layer):
    def _init_(self, units):
        super(Attention, self).__init__()
        self.Wc = tf.keras.layers.Dense(units)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_padding_mask, use_coverage=False, prev_coverage=None):
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        def masked_attention(score):
            mask = tf.cast(enc_padding_mask, dtype=score.dtype)
            masked_score = tf.squeeze(score, axis=-1) * mask
            masked_score = tf.expand_dims(masked_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            return attention_weights

        if use_coverage and prev_coverage is not None:
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis) + self.Wc(prev_coverage)))
            attn_dist = masked_attention(e)
            tf.keras.layers.RNN
            coverage=attn_dist+prev_coverage
        else:
            e = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
            attn_dist = masked_attention(e)
            if use_coverage:
                coverage = attn_dist
            else:
                coverage = []

        context_vector = attn_dist * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, tf.squeeze(attn_dist, -1), coverage


class Decoder(tf.keras.layers.Layer):
    def _init_(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dropout(0.5)
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        out = self.fc(output)
        return x, out, state


