from __future__ import division

from time import time

import models, data
import pickle
import sys
import os.path

import tensorflow as tf
import numpy as np

MAX_EPOCHS = 50
MINIBATCH_SIZE = 32
CLIPPING_THRESHOLD = 2.0
PATIENCE_EPOCHS = 1

"""
Bi-directional RNN with attention
For a sequence of N words, the model makes N punctuation decisions (no punctuation before the first word, but there's a decision after the last word or before </S>)
"""

def get_minibatch(file_name, batch_size, shuffle, with_pauses=False):

    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)

    if shuffle:
        np.random.shuffle(dataset)

    X_batch = []
    Y_batch = []

    if len(dataset) < batch_size:
        lenwarning = (
        f"WARNING: Not enough samples in {file_name}. "
        f"Reduce mini-batch size to {len(dataset)} "
        f"or use a dataset with at least {MINIBATCH_SIZE * data.MAX_SEQUENCE_LEN} words."
        )
        print(lenwarning)

    for subsequence in dataset:

        X_batch.append(subsequence[0])
        Y_batch.append(subsequence[1])

        if len(X_batch) == batch_size:

            # Transpose, because the model assumes the first axis is time
            X = np.array(X_batch, dtype=np.int32).T
            Y = np.array(Y_batch, dtype=np.int32).T

            yield X, Y

            X_batch = []
            Y_batch = []

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = models.cost(y_pred, y)
    gradients = tape.gradient(loss, model.params)
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIPPING_THRESHOLD)
    optimizer.apply_gradients(zip(gradients, model.params))
    return loss

if __name__ == "__main__":
    
    starting_time = time()
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("'Model name' argument missing!")

    if len(sys.argv) > 2:
        num_hidden = int(sys.argv[2])
    else:
        sys.exit("'Hidden layer size' argument missing!")
    
    if len(sys.argv) > 3:
        learning_rate = float(sys.argv[3])
    else:
        sys.exit("'Learning rate' argument missing!")

    model_file_name = "Model_%s_h%d_lr%s.pcl" % (model_name, num_hidden, learning_rate)

    print(num_hidden, learning_rate, model_file_name)

    rng = np.random
    rng.seed(1)

    print("Building model ...")
    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, MINIBATCH_SIZE)).astype(int)
    # Initialize the weights of the model without any real data, comparable to placeholders in earlier Tensorflow version
    net = models.GRU(rng, x, num_hidden)
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=1e-6)

    starting_epoch = 0
    best_ppl = np.inf
    validation_ppl_history = []

    print(f"Total number of trainable parameters: {sum(np.prod([dim for dim in param.get_shape()]) for param in net.params)}")

    print("Training...")
    for epoch in range(starting_epoch, MAX_EPOCHS):
        t0 = time()
        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        iteration = 0 
        for X, Y in get_minibatch(data.TRAIN_FILE, MINIBATCH_SIZE, shuffle=True):
            loss = train_step(net, X, Y)
            total_neg_log_likelihood += loss
            total_num_output_samples += np.prod(Y.shape)
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.write("PPL: %.4f; Speed: %.2f sps\n" % (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100)))
                sys.stdout.flush()
        print(f"Total number of training labels: {total_num_output_samples}")

        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        for X, Y in get_minibatch(data.DEV_FILE, MINIBATCH_SIZE, shuffle=False):
            total_neg_log_likelihood += models.cost(net(X, training=True), Y)
            print(f"Total neg log likelihood in dev iteration {iteration}, epoch {epoch}: {total_neg_log_likelihood}")
            total_num_output_samples += np.prod(Y.shape)
            print(f"Total num output samples in dev iteration {iteration}, epoch {epoch}: {total_num_output_samples}")
        print(f"Total number of validation labels: {total_num_output_samples}")

        ppl = np.exp(total_neg_log_likelihood / total_num_output_samples)
        validation_ppl_history.append(ppl)

        print(f"Validation perplexity is {np.round(ppl, 4)}")

        if ppl <= best_ppl:
            best_ppl = ppl
            models.save(net, model_file_name, learning_rate=learning_rate, validation_ppl_history=validation_ppl_history, best_validation_ppl=best_ppl, epoch=epoch, random_state=rng.get_state())
        elif best_ppl not in validation_ppl_history[-PATIENCE_EPOCHS:]:
            print("Finished!")
            print(f"Best validation perplexity was {best_ppl}")
            print(f"Total time: {time() - starting_time}")
            break
