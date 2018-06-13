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
    

class TfRandomClassifier(TfModelBase):
    """Defines a Bidirectional RNN in which the final hidden state is used as
    the basis for a softmax classifier predicting a label

    Parameters
    ----------
    None
    """

    def __init__(self,
            # model dimensions
            **kwargs):

        super().__init__(**kwargs)
        # self.eta = self.config.lr

        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        inputs_batch, lens_batch, labels_batch, self.ids_batch = self.data_manager.batch_op

        batch_size = tf.shape(inputs_batch)[0]
        self.inputs_placeholder = tf.reshape(inputs_batch, 
            shape=(batch_size, MAX_LENGTH, 2), name="inputs") # word ids and case ids

        # word_ids, case_ids = tf.split(self.inputs_placeholder, 
        #                                         num_or_size_splits=2, axis=2)

        self.word_ids = self.inputs_placeholder[:, :, 0]
        self.case_ids = self.inputs_placeholder[:, :, 1]

        self.lens_placeholder = tf.reshape(lens_batch, 
            shape=(batch_size,), name="inputs_lengths")

        self.outputs = tf.reshape(labels_batch, 
            shape=(batch_size, N_CLASSES), name="labels")

        # dropout params


    def add_prediction_op(self):
        batch_size = tf.shape(self.inputs_placeholder)[0]
        return tf.random_uniform(
            shape=(batch_size, N_CLASSES)
        )


    def build_graph(self):
        self.add_placeholders()
        self.model = self.add_prediction_op()

    def get_optimizer(self):
        return tf.no_op();

    def train_dict(self):
        return {self.data_manager.handle: self.data_manager.handle_train}

    def dev_dict(self):
        return {self.data_manager.handle: self.data_manager.handle_dev}

    def test_dict(self):
        return {self.data_manager.handle: self.data_manager.handle_test}


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

    mod = TfLinearClassifier(
        vocab=vocab, max_iter=100, max_length=4)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
