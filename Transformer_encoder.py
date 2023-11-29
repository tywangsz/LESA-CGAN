import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional
import time
import numpy as np
import matplotlib.pyplot as plt





def scaled_dot_product_attention(q, k, v, mask):

    Q = tf.nn.l2_normalize( q ,dim=1, epsilon=1e-10, name='nn_l2_norm')
    Q = tf.transpose(Q, perm=[0, 3, 2, 1])
    K = tf.nn.l2_normalize( k ,dim=1, epsilon=1e-10, name='nn_l2_norm')
    V = v
    matrixkv = tf.einsum('bnlm, bnlm->bnlm', K, V)
    attention_weights = matrixkv
    matrixqv = tf.einsum('bmln, bnlm->bnlm', Q, V)
    matrixqkv = tf.einsum('bnlm, bmln->bnlm', matrixkv, Q)
    summ_numa = V + matrixkv + matrixqv + matrixqkv
    summ_dema = 1/( Q + K + tf.einsum('bmln, bnlm->bnlm', Q, K) )
    fraction = tf.matmul(summ_numa, summ_dema, transpose_b=True)
    return fraction, attention_weights





class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)



        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)



        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='swish'),

        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding, rate = 0.1):

        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.pos_emb = tf.keras.layers.Embedding(input_dim = self.maximum_position_encoding,
                                                 output_dim = self.d_model)

        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(self.rate)

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'maximum_position_encoding': self.maximum_position_encoding,
            'dropout': self.dropout,
        })
        return config

    def call(self, x, training, mask = None):

        seq_len = tf.shape(x)[1]


        positions = tf.range(start = 0, limit = seq_len, delta = 1)
        x += self.pos_emb(positions)

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):

            x = self.enc_layers[i](x, training, mask)

        return x
