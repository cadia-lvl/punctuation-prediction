import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import codecs
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

tf.get_logger().setLevel("INFO")

END = "</S>"
UNK = "<UNK>"

SPACE = "_SPACE"

MAX_WORD_VOCABULARY_SIZE = 100000
MIN_WORD_COUNT_IN_VOCAB = 2
MAX_SEQUENCE_LEN = 200

MINIBATCH_SIZE = 16

WORD_VOCAB_FILE = "biRNN/vocabulary"
PUNCT_VOCAB_FILE = "biRNN/punctuations"

EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}


def iterable_to_dict(arr):
    return dict((x.strip(), i) for (i, x) in enumerate(arr))


def read_vocabulary(file_name):
    with codecs.open(file_name, "r", "utf-8") as f:
        vocabulary = f.readlines()
        # print('Vocabulary "%s" size: %d' % (file_name, len(vocabulary)))
        return iterable_to_dict(vocabulary)


def _get_shape(i, o, keepdims):
    if (i == 1 or o == 1) and not keepdims:
        return [
            max(i, o),
        ]
    else:
        return [i, o]


def _slice(tensor, size, i):
    """Gets slice of columns of the tensor"""
    return tensor[:, i * size : (i + 1) * size]


def weights_Glorot(i, o, name, rng, is_logistic_sigmoid=False, keepdims=False):
    # http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    d = np.sqrt(6.0 / (i + o))
    if is_logistic_sigmoid:
        d *= 4.0
    return tf.Variable(tf.random.uniform(_get_shape(i, o, keepdims), -d, d))


def load(file_path, x, p=None):
    with open(file_path, "rb") as f:
        state = pickle.load(f)

    rng = np.random
    rng.set_state(state["random_state"])

    net = GRU(rng=rng, x=x, n_hidden=state["n_hidden"])

    for net_param, state_param in zip(net.params, state["params"]):
        net_param.assign(state_param)

    return (
        net,
        (state["learning_rate"], state["validation_ppl_history"], state["epoch"], rng),
    )


class GRUCell(layers.Layer):
    def __init__(self, rng, n_in, n_out, minibatch_size):
        super(GRUCell, self).__init__()
        # Notation from: An Empirical Exploration of Recurrent Network Architectures

        self.n_in = n_in
        self.n_out = n_out

        # Initial hidden state
        self.h0 = tf.zeros([minibatch_size, n_out])

        # Gate parameters:
        self.W_x = weights_Glorot(n_in, n_out * 2, "W_x", rng)
        self.W_h = weights_Glorot(n_out, n_out * 2, "W_h", rng)
        self.b = tf.Variable(tf.zeros([1, n_out * 2]))
        # Input parameters
        self.W_x_h = weights_Glorot(n_in, n_out, "W_x_h", rng)
        self.W_h_h = weights_Glorot(n_out, n_out, "W_h_h", rng)
        self.b_h = tf.Variable(tf.zeros([1, n_out]))

        self.params = [self.W_x, self.W_h, self.b, self.W_x_h, self.W_h_h, self.b_h]

    # inputs = x_t, h_tm1
    def call(self, inputs):

        rz = tf.nn.sigmoid(
            tf.matmul(inputs[0], self.W_x) + tf.matmul(inputs[1], self.W_h) + self.b
        )
        r = _slice(rz, self.n_out, 0)
        z = _slice(rz, self.n_out, 1)

        h = tf.nn.tanh(
            tf.matmul(inputs[0], self.W_x_h)
            + tf.matmul(inputs[1] * r, self.W_h_h)
            + self.b_h
        )

        h_t = z * inputs[1] + (1.0 - z) * h

        return h_t


class GRU(tf.keras.Model):
    def __init__(self, rng, x, n_hidden):
        super(GRU, self).__init__()

        self.minibatch_size = tf.shape(x)[1]

        self.n_hidden = n_hidden
        self.x_vocabulary = read_vocabulary(WORD_VOCAB_FILE)
        self.y_vocabulary = read_vocabulary(PUNCT_VOCAB_FILE)

        self.x_vocabulary_size = len(self.x_vocabulary)
        self.y_vocabulary_size = len(self.y_vocabulary)

        # input model
        self.We = weights_Glorot(
            self.x_vocabulary_size, n_hidden, "We", rng
        )  # Share embeddings between forward and backward model
        self.GRU_f = GRUCell(
            rng=rng, n_in=n_hidden, n_out=n_hidden, minibatch_size=self.minibatch_size
        )
        self.GRU_b = GRUCell(
            rng=rng, n_in=n_hidden, n_out=n_hidden, minibatch_size=self.minibatch_size
        )

        # output model
        self.GRU = GRUCell(
            rng=rng,
            n_in=n_hidden * 2,
            n_out=n_hidden,
            minibatch_size=self.minibatch_size,
        )
        self.Wy = tf.Variable(tf.zeros([n_hidden, self.y_vocabulary_size]))
        self.by = tf.Variable(tf.zeros([1, self.y_vocabulary_size]))

        # attention model
        n_attention = (
            n_hidden * 2
        )  # to match concatenated forward and reverse model states
        self.Wa_h = weights_Glorot(
            n_hidden, n_attention, "Wa_h", rng
        )  # output model previous hidden state to attention model weights
        self.Wa_c = weights_Glorot(
            n_attention, n_attention, "Wa_c", rng
        )  # contexts to attention model weights
        self.ba = tf.Variable(tf.zeros([1, n_attention]))
        self.Wa_y = weights_Glorot(
            n_attention, 1, "Wa_y", rng
        )  # gives weights to contexts

        # Late fusion parameters
        self.Wf_h = tf.Variable(tf.zeros([n_hidden, n_hidden]))
        self.Wf_c = tf.Variable(tf.zeros([n_attention, n_hidden]))
        self.Wf_f = tf.Variable(tf.zeros([n_hidden, n_hidden]))
        self.bf = tf.Variable(tf.zeros([1, n_hidden]))

        self.params = [
            self.We,
            self.Wy,
            self.by,
            self.Wa_h,
            self.Wa_c,
            self.ba,
            self.Wa_y,
            self.Wf_h,
            self.Wf_c,
            self.Wf_f,
            self.bf,
        ]

        self.params += self.GRU.params + self.GRU_f.params + self.GRU_b.params
        # print([x.shape for x in self.params])

    def call(self, inputs, training=None):

        # bi-directional recurrence
        def input_recurrence(initializer, elems):
            x_f_t, x_b_t = elems
            h_f_tm1, h_b_tm1 = initializer

            h_f_t = self.GRU_f(inputs=(tf.nn.embedding_lookup(self.We, x_f_t), h_f_tm1))
            h_b_t = self.GRU_b(inputs=(tf.nn.embedding_lookup(self.We, x_b_t), h_b_tm1))
            return [h_f_t, h_b_t]

        [h_f_t, h_b_t] = tf.scan(
            fn=input_recurrence,
            elems=[inputs, inputs[::-1]],  # forward and backward sequences
            initializer=[self.GRU_f.h0, self.GRU_b.h0],
        )

        # 0-axis is time steps, 1-axis is batch size and 2-axis is hidden layer size
        context = tf.concat([h_f_t, h_b_t[::-1]], axis=2)
        # projected_context = tf.matmul(context, self.Wa_c) + self.ba for each tensor slice
        projected_context = (
            tf.matmul(
                context,
                tf.tile(
                    tf.expand_dims(self.Wa_c, 0), tf.stack([tf.shape(context)[0], 1, 1])
                ),
            )
            + self.ba
        )

        def output_recurrence(initializer, elems):
            x_t = elems
            h_tm1, _, _ = initializer

            # Attention model
            h_a = tf.nn.tanh(projected_context + tf.matmul(h_tm1, self.Wa_h))

            # alphas = tf.exp(tf.matmul(h_a, self.Wa_y))
            # alphas = tf.reshape(alphas, [tf.shape(alphas)[0], tf.shape(alphas)[1]]) # drop 2-axis (sized 1) is replaced by:
            # sess.run(tf.reshape(tf.matmul(tf.reshape(x, [-1, tf.shape(x)[-1]]), tf.expand_dims(z,-1)), tf.shape(x)[:2]))
            alphas = tf.exp(
                tf.reshape(
                    tf.matmul(
                        tf.reshape(h_a, [-1, tf.shape(h_a)[-1]]),
                        tf.expand_dims(self.Wa_y, -1),
                    ),
                    tf.shape(h_a)[:2],
                )
            )
            alphas = alphas / tf.reduce_sum(alphas, axis=0, keepdims=True)
            weighted_context = tf.reduce_sum(context * alphas[:, :, None], axis=0)

            h_t = self.GRU(inputs=(x_t, h_tm1))

            # Late fusion
            lfc = tf.matmul(weighted_context, self.Wf_c)  # late fused context
            fw = tf.nn.sigmoid(
                tf.matmul(lfc, self.Wf_f) + tf.matmul(h_t, self.Wf_h) + self.bf
            )  # fusion weights
            hf_t = lfc * fw + h_t  # weighted fused context + hidden state

            z = tf.matmul(hf_t, self.Wy) + self.by
            y_t = z  # tf.nn.softmax(z)

            return [h_t, hf_t, y_t]

        [_, self.last_hidden_states, self.y] = tf.scan(
            fn=output_recurrence,
            elems=context[
                1:
            ],  # ignore the 1st word in context, because there's no punctuation before that
            initializer=[
                self.GRU.h0,
                self.GRU.h0,
                tf.zeros([self.minibatch_size, self.y_vocabulary_size]),
            ],
        )

        return self.y

