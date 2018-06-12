#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
from tf_birnn_classifier import TfBiRNNClassifier
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
    

class TfCFClassifier(TfBiRNNClassifier):
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
    max_iter : int
        Maximum number of iterations allowed in training.
    eta : float
        Learning rate.
    tol : float
        Stopping criterion for the loss.
    """

    def __init__(self, n_users,
            embedding=None,
            train_embedding=True,
            max_length=MAX_LENGTH,
            hidden_activation = tf.nn.relu,
            cell_class=tf.nn.rnn_cell.LSTMCell,

            # model dimensions
            hidden_dim = 50,
            bidirectional = True,
            use_attn = False,
            context_size = 25,
            inputs_keep = 0.6,
            state_keep = 0.6,
            **kwargs):

        self.n_users = n_users
        self.embedding = embedding
        self.email_embedding = None
        self.embed_dim = EMBED_SIZE
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        self.hidden_activation = hidden_activation

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_attn = use_attn
        self.context_size = context_size
        self.inputs_keep = inputs_keep
        self.state_keep = state_keep
        super().__init__(**kwargs)
        # self.eta = self.config.lr

        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']

    def define_embedding(self):

       if type(self.email_embedding) == type(None):
            self.email_embedding = np.random.uniform(size=[self.n_users, self.embed_dim],
                low=-1.0, high=1.0) 


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        inputs_batch, lens_batch, labels_batch, self.ids_batch, to_batch, from_batch\
             = self.data_manager.batch_op

        batch_size = tf.shape(inputs_batch)[0]
        self.inputs_placeholder = tf.reshape(inputs_batch, 
            shape=(batch_size, self.max_length, 2), name="inputs") # word ids and case ids

        self.word_ids = self.inputs_placeholder[:, :, 0]
        self.case_ids = self.inputs_placeholder[:, :, 1]

        self.lens_placeholder = tf.reshape(lens_batch, 
            shape=(batch_size,), name="inputs_lengths")

        self.outputs = tf.reshape(labels_batch, 
            shape=(batch_size, N_CLASSES), name="labels")

        self.to_ids = to_batch
        self.from_ids = from_batch

        # dropout params
        self.input_keep_prob = tf.placeholder(tf.float32, shape=(), name="input_keep_prob")
        self.state_keep_prob = tf.placeholder(tf.float32, shape=(), name="state_keep_prob")

    def add_email_features(self):
        """Adds a trainable embedding layer.

        Returns:
           email_embeddings: tf.Tensor of shape (None, max_length, n_features*embed_dim)
        """
        assert self.email_embedding.shape[-1] == self.embed_dim
        all_email_embeddings = tf.get_variable('email_embedding', 
            shape=self.email_embedding.shape, 
            initializer=tf.constant_initializer(self.email_embedding),
            trainable=True
        )     

        to_embeddings = tf.nn.embedding_lookup(
            params=all_email_embeddings, 
            ids=self.to_ids
        )                                                                                                          
        from_embeddings = tf.nn.embedding_lookup(
            params=all_email_embeddings, 
            ids=self.from_ids
        )    
        return to_embeddings, from_embeddings

    def add_cf_model(self):
        u_to, u_from = self.add_email_features()
        q_email = self.fc_ins 

        item_vec = tf.contrib.layers.fully_connected(
            tf.concat([u_from, q_email], axis=1),
            self.embed_dim,
            activation_fn=None
        )
        print("item_vec: ", item_vec.get_shape())
        phi_user = tf.multiply(u_to, item_vec)

        phi_item = q_email

        preds =  tf.contrib.layers.fully_connected(
            tf.concat([phi_user, phi_item], axis=1),
            N_CLASSES,
            activation_fn=None
        )
        return preds 


    def build_graph(self):
        self.define_embedding()
        self.add_placeholders()
        self.add_prediction_op()
        self.model = self.add_cf_model()


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
        ops = [self.probs, self.inputs_placeholder, self.lens_placeholder, self.outputs, self.ids_batch]
        if self.use_attn:
            ops.append(self.attn)
        return self.sess.run(ops, feed_dict=self.test_dict())

    def predict(self, init_dm=True, dataset='dev'):
        """Return classifier predictions, as the class with the
        highest probability for each example, for a single batch

        Returns
        -------
        list

        """
        result = self.predict_proba(init_dm, dataset)
        if self.use_attn:
            probs, inputs, lens, outputs, email_ids, attn = result
            return np.argmax(probs, axis=1), inputs, lens, np.argmax(outputs, axis=1), email_ids, attn
        else:
            probs, inputs, lens, outputs, email_ids = result
            return np.argmax(probs, axis=1), inputs, lens, np.argmax(outputs, axis=1), email_ids




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
