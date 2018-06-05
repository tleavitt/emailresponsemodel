#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
import warnings
import pdb
from defs import EMBED_SIZE, N_CASES, N_CLASSES, MAX_LENGTH

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
    

class TfMLPClassifier(TfModelBase):
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
            embedding=None,
            train_embedding=True,
            max_length = MAX_LENGTH,

            # model parameters,
            hidden_dim = 100,
            hidden_activation = tf.tanh,
            keep_prob = 0.75, 
            # model dimensions
            **kwargs):

        self.embedding = embedding
        self.embed_dim = EMBED_SIZE
        self.max_length = max_length
        self.train_embedding = train_embedding

        self.hidem_dim = hiddem_dim
        self.hidden_activation = hidden_dim
        self.keep_prob_ = keep_prob

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
        inputs_batch, lens_batch, labels_batch, self.ids_batch = self.data_manager.batch_op

        batch_size = tf.shape(inputs_batch)[0]
        self.inputs_placeholder = tf.reshape(inputs_batch, 
            shape=(batch_size, self.max_length, 2), name="inputs") # word ids and case ids

        # word_ids, case_ids = tf.split(self.inputs_placeholder, 
        #                                         num_or_size_splits=2, axis=2)

        self.word_ids = self.inputs_placeholder[:, :, 0]
        self.case_ids = self.inputs_placeholder[:, :, 1]

        self.lens_placeholder = tf.reshape(lens_batch, 
            shape=(batch_size,), name="inputs_lengths")

        self.outputs = tf.reshape(labels_batch, 
            shape=(batch_size, N_CLASSES), name="labels")

        # dropout params
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

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
        self.n_word_features = self.embed_dim + N_CASES
        x = self.get_features()

        # Take the average word vector
        x_avg = tf.reduce_mean(x, axis=1)
        W1= tf.get_variable('W1',
            shape=(self.n_word_features, self.hidden_dim),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b1 = tf.get_variable('b1', shape=(self.hidden_dim))

        h = self.hidden_activation(tf.matmul(x_avg, W1) + b1)
        h_drop = tf.nn.dropout(h, self.keep_prob)

        W2 = tf.get_variable('W2',
            shape=(self.hidden_dim, N_CLASSES),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        b2 = tf.get_variable('b2', shape=(N_CLASSES))

        preds = tf.matmul(h_drop, W2) + b2
        
        return preds


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
        return {self.keep_prob: self.keep_prob_}

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
        return {self.keep_prob: 1.0}

    # override to use Adam
    def get_optimizer(self):
        return tf.train.AdamOptimizer(
            self.eta).minimize(self.cost, global_step=self.global_step)


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
        if not self.sess:
            logger.error("model unitnialized, not running batch.")
            return
        if init_dm:
            self.sess.run(self.data_manager.initializer, feed_dict=self.data_manager.get_init_feed_dict(dataset))
        probs = tf.nn.softmax(self.model)
        return self.sess.run(
            [probs, self.inputs_placeholder, self.lens_placeholder, self.outputs, self.ids_batch], feed_dict=self.test_dict())

    def predict(self, init_dm=True, dataset='dev'):
        """Return classifier predictions, as the class with the
        highest probability for each example, for a single batch

        Returns
        -------
        list

        """
        probs, inputs, lens, outputs, email_ids = self.predict_proba(init_dm, dataset)
        return np.argmax(probs, axis=1), inputs, lens, np.argmax(outputs, axis=1), email_ids


