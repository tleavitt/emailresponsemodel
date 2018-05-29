#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
import warnings
import pdb
from defs import EMBED_SIZE, N_CASES, N_CLASSES

__author__ = 'Tucker Leavitt'

# Ignore the TensorFlow warning
#   Converting sparse IndexedSlices to a dense Tensor of unknown shape.
#   This may consume a large amount of memory.
warnings.filterwarnings("ignore", category=UserWarning)

# class BiRNNConfig():  
#     feature_dim = EMBED_SIZE + N_CASES
#     n_features = 2
#     keep_prob = 0.5
#     max_pool = False
    

class TfBiRNNClassifier(TfModelBase):
    """Defines a Bidirectional RNN in which the final hidden state is used as
    the basis for a softmax classifier predicting a label

    Parameters
    ----------
    vocab : list
        The full vocabulary. `_convert_X` will convert the data provided
        to `fit` and `predict` methods into a list of indices into this
        list of items.
    embedding : 2d np.array or None
        If `None`, then a random embedding matrix is constructed.
        Otherwise, this should be a 2d array aligned row-wise with
        `vocab`, with each row giving the input representation for the
        corresponding word. For instance, to roughly duplicate what
        is done by default, one could do
            `np.array([np.random.randn(h) for _ in vocab])`
        where n is the embedding dimensionality (`embed_dim`).
    embed_dim : int
        Dimensionality of the inputs/embeddings. If `embedding`
        is supplied, then this value is set to be the same as its
        column dimensionality. Otherwise, this value is used to create
        the embedding Tensor (see `_define_embedding`).
    max_length : int
        Maximum sequence length.
    train_embedding : bool
        Whether to update the embedding matrix when training.
    hidden_activation : tf.nn activation
       E.g., tf.nn.relu, tf.nn.relu, tf.nn.selu.
    hidden_dim : int
        Dimensionality of the hidden layer.
    max_iter : int
        Maximum number of iterations allowed in training.
    eta : float
        Learning rate.
    tol : float
        Stopping criterion for the loss.
    """

    def __init__(self,
            vocab,
            embedding=None,
            train_embedding=True,
            max_length=52,
            hidden_activation = tf.nn.relu,
            cell_class=tf.nn.rnn_cell.LSTMCell,

            # model dimensions
            hidden_size = 100,
            bidirectional = True,
            context_size = 50,
            inputs_keep = 0.6,
            state_keep = 0.6,
            **kwargs):

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = EMBED_SIZE
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        self.hidden_activation = hidden_activation

        self.hidden_size = hidden_size
        self.bidirectional = True
        self.context_size = context_size
        self.inputs_keep = inputs_keep
        self.state_keep = state_keep
        super().__init__(**kwargs)
        # self.eta = self.config.lr

        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']

    def define_embedding(self):

        if type(self.embedding) == type(None):
            self.embedding = np.random.uniform(size=[self.vocab_size, self.embed_dim],
                low=-1.0, high=1.0)


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        inputs_batch, lens_batch, labels_batch = self.data_manager.batch_op

        batch_size = tf.shape(inputs_batch)[0]
        self.inputs_placeholder = tf.reshape(inputs_batch, 
            shape=(batch_size, self.max_length, 2), name="inputs") # word ids and case ids

        self.word_ids, self.case_ids = tf.split(self.inputs_placeholder, 
                                                num_or_size_splits=2, axis=2)

        self.lens_placeholder = tf.reshape(lens_batch, 
            shape=(batch_size,), name="inputs_lengths")

        self.outputs = tf.reshape(labels_batch, 
            shape=(batch_size, N_CLASSES), name="labels")

        # dropout params
        self.input_keep_prob = tf.placeholder(tf.float32, shape=(), name="input_keep_prob")
        self.state_keep_prob = tf.placeholder(tf.float32, shape=(), name="state_keep_prob")

    def add_wordvec_features(self):
        """Adds a trainable embedding layer.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_dim)
        """
        assert self.embedding.shape[-1] == self.embed_dim
        all_embeddings = tf.get_variable('embeddings', 
            shape=self.embedding.shape, 
            initializer=tf.constant_initializer(self.embedding),
            trainable=self.train_embedding
        )     

        input_embeddings = tf.nn.embedding_lookup(
            params=all_embeddings, 
            ids=self.word_ids
        )                                                                                                          
        embeddings = tf.reshape(input_embeddings, 
            (-1, self.max_length, self.embed_dim)
        )                                                                                                
        return embeddings

    def add_case_features(self):
        return tf.one_hot(self.case_ids, N_CASES) 

    def get_features(self):
        return tf.concat(
            [self.add_wordvec_features(), self.add_case_features()], 
            axis=2
        )

    def add_prediction_op(self):
        self.n_word_features = self.embed_dim + N_CASES if self.case_features else self.embed_dim
        x = self.get_features()
        # x_static = self.parse_case_features(self.add_static_embedding())
        # x = tf.stack([x_var, x_static], axis=-1) # concatenate the inputs on the last axis
        # x has shape (batch_size, max_length, embed_dim + N_CASES) 

        # This converts the inputs to a list of lists of dense vector
        # representations:

        # Defines the RNN structure:
        outputs = self.run_rnn(x, self.lens_placeholder)

        # Add an attention layer
        self.attn = self.add_attention_layer(outputs)

        # Add a fully connected softmax layer:
        preds = tf.contrib.layers.fully_connected(
            self.attn,
            N_CLASSES,
            activation_fn=None
        )

        return preds


    def create_rnn_cell(self):
        inner_cell = self.cell_class(
            self.hidden_dim, activation=self.hidden_activation,
            initializer=tf.contrib.layers.xavier_initializer()
        )

        return tf.nn.rnn_cell.DropoutWrapper(
            inner_cell,
            input_keep_prob = self.input_keep_prob,
            state_keep_prob = self.state_keep_prob  
        )

    def run_rnn(self, x, lens):
         # Pick out the cell to use here.
        with tf.variable_scope('rnn'):
            # cells = []
            cell_fw = self.create_rnn_cell()
            cell_bw = self.create_rnn_cell()

            # stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            if self.bidirectional:
                (fw_outputs, bw_outputs), state = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, 
                    cell_bw, 
                    x,
                    sequence_length=lens,
                    dtype=tf.float32
                )
                """
                TODO: may need to do finnicky stuff with LSTMStateTuple...?
                """
                all_outputs = tf.concat( [fw_outputs, bw_outputs], axis=2 )
            else:
                all_outputs, state = tf.nn.dynamic_rnn(cell, x,
                    sequence_length=lens,
                    dtype=tf.float32
                )

        return all_outputs

    def add_attention_layer(self, rnn_outputs):
        with tf.variable_scope('attention'):

            hidden_size = 2 * self.hidden_size if self.bidirectional else self.hidden_size
            W = tf.get_variable('W_w',
               shape = (hidden_size, self.context_size),
               initializer=tf.contrib.layers.xavier_initializer()
            )       
            b = tf.get_variable('b_w',
                shape = (self.context_size),
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # broadcast multiplication by W over time over each batch
            # outputs_shape = tf.shape(rnn_outputs)
            # u_flat = tf.tanh( tf.matmul(
            #         tf.reshape(rnn_outputs, [-1, self.hidden_size]), W
            #     )) + b
            # u = tf.reshape(u_flat, (outputs_shape[0], outputs_shape[1], self.context_size))
            u = tf.einsum('aij,jk->aik', rnn_outputs, W) + b

            # u_flat has shape (batch_size * max_length, context_size)
            # u has shape (batch_size, max_length, context_size)
            uw = tf.get_variable('word_context_vector',
               # shape = (self.context_size, 1),
               shape = (self.context_size,),
               initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            )

            self.alphas = tf.nn.softmax(tf.einsum('aij,j->ai', u, uw), dim=1)
            # logits = tf.reshape(tf.matmul(u_flat, uw), (outputs_shape[0], outputs_shape[1]))
            # alphas = tf.nn.softmax(logits)
            # alphas has shape (batch_size, max_length)
            # use the alphas to form a weighted sum of the LSTM states
            # fc_ins = tf.reduce_sum(tf.expand_dims(alphas, 2) * rnn_outputs, axis = 1)
            weighted_rnn_outputs = tf.einsum('ai,aij->aj', self.alphas, rnn_outputs)
            # fc_ins = tf.reduce_sum(tf.expand_dims(alphas, axis=2) * rnn_outputs, axis=1)

            # outputs = self.activation_fn(weighted_rnn_outputs) #TODO: consider removing?
            outputs = weighted_rnn_outputs

        return outputs

    def build_graph(self):
        self.define_embedding()
        self.add_placeholders()
        self.model = self.add_prediction_op()


    def train_dict(self):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, , and gets the true length of each example
        and passes it to `fit` as well. `y` is fed to `outputs`.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        return {
            self.input_keep_prob: self.inputs_keep,
            self.state_keep_prob: self.state_keep,
        }

    def test_dict(self):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, and gets the true length of each example and
        passes it to `fit` as well.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        return {
            self.input_keep_prob: 1.0,
            self.state_keep_prob: 1.0,
        }

    # override to use Adam
    def get_optimizer(self):
        return tf.train.AdamOptimizer(
            self.eta).minimize(self.cost)


    def predict_proba(self, init_dm=True, dataset='dev'):
        """Return probabilistic predictions.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array of predictions, dimension m x n, where m is the length
        of X and n is the number of classes

        """
        if init_dm:
            sess.run(self.dm.initializer, feed_dict=self.dm.get_init_feed_dict(dataset))
        self.probs = tf.nn.softmax(self.model)
        return self.sess.run(
            self.probs, self.attn, self.inputs_placeholder, self.outputs, feed_dict=self.test_dict())

    def predict(self, init_dm=True, dataset='dev'):
        """Return classifier predictions, as the class with the
        highest probability for each example, for a single batch

        Returns
        -------
        list

        """
        probs, attn, inputs, outputs = self.predict_proba(init_dm, dataset)
        return [(np.argmax(p), a, x, y) for p, a, x, y in zip(probs, attn, inputs, outputs)]




def simple_example():
    vocab = ['a', 'b', '$UNK']

    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad']]

    test = [
        [list('aaab'), 'good'],
        [list('baaa'), 'bad']]

    mod = TfCNNClassifier(
        vocab=vocab, max_iter=100, max_length=4)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
