#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import sys
import tensorflow as tf
from defs import BATCH_LIM, PROJECT_DIR, LBLS
from data_util import tfDatasetManager, tfConfig
from util import sysprint, check_dirs
from sklearn import metrics
import time
import logging
import os
from glob import glob
import pdb

__author__ = 'Tucker Leavitt, Chris Potts'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def most_recent_meta_graph_fn(ckpts_prefix):
    metas = sorted( glob(
       '{}-*.meta'.format(ckpts_prefix) 
    ))
    return metas[0] if len(metas) > 0 else None


class TfModelBase(object):
    """
    Parameters
    ----------
    hidden_dim : int
    hidden_activation : tf.nn function
    max_iter : int
    eta : float
        Learning rate
    tol : float
        Stopping criterion for the loss.
    display_progress : int
        For value i, progress is printed every ith iteration.

    Attributes
    ----------
    errors : list
        Tracks loss from each iteration during training.
    """
    def __init__(self, ckpts_prefix = None, hidden_dim=50, hidden_activation=tf.nn.relu, 
            batch_size=32, eta=0.005, tol=1e-4, display_progress=1,
            summaries_dir = os.path.abspath('{}/summaries'.format(PROJECT_DIR)) ):
        self.hidden_dim = hidden_dim
        self.hidden_activation = tf.nn.tanh
        self.batch_size = batch_size
        self.eta = eta
        self.tol = tol
        self.display_progress = display_progress
        self.errors = []
        self.dev_predictions = []
        self.summaries_dir = summaries_dir
        self.ckpts_prefix = ckpts_prefix
        self.data_manager = None

        if self.summaries_dir is not None:
            check_dirs(self.summaries_dir + '/train')
            check_dirs(self.summaries_dir + '/dev')

        if self.ckpts_prefix is not None:
            check_dirs(os.path.dirname(ckpts_prefix))
            
        self.params = [
            'hidden_dim', 'hidden_activation', 'max_iter', 'eta']

    def build_graph(self):
        """Define the computation graph. This needs to define
        variables on which the cost function depends, so see
        `cost_function` below unless you are defining your own.

        """
        raise NotImplementedError

    def train_dict(self):
        """This method should feed `X` to the placeholder that defines
        the inputs and `y` to the placeholder that defines the output.
        For example:

        {self.inputs: X, self.outputs: y}

        This is used during training.

        """
        raise NotImplementedError

    def test_dict(self):
        """This method should feed `X` to the placeholder that defines
        the inputs. For example:

        {self.inputs: X}

        This is used during training.

        """
        raise NotImplementedError


    # Load the session variables
    def try_restore(self, sess, saver):
        if self.ckpts_prefix is None:
            return None
        latest_ckpt_path = tf.train.latest_checkpoint(os.path.dirname(self.ckpts_prefix))
        # latest_meta_path = most_recent_meta_graph_fn(self.ckpts_prefix)
        if latest_ckpt_path is not None:
            # saver = tf.train.import_meta_graph(latest_meta_path)
            saver.restore(sess, latest_ckpt_path)
            return True
        else:
            return False

    def fit(self, tfconfig, restore_weights = False, batches_to_eval=5000, max_iter=10, n_val_batches = 40):
        """ Trains using a tf DatasetManager

        Parameters
        ----------
        tfconfig: config object for the tf DatasetManager
        kwargs : dict
            For passing other parameters, e.g., a test set that we
            want to monitor performance on.

        Returns
        -------
        self

        """

        tf.reset_default_graph()
        # with tf.Graph().as_default():

        sess = tf.InteractiveSession()
        self.sess = sess

        if self.data_manager is None:
            logger.info("Initializing data manager...",)
            start = time.time()
            self.data_manager = tfDatasetManager(tfconfig)
            logger.info("took %d s", time.time() - start)


        # Build the computation graph. This method is instantiated by
        # individual subclasses. It defines the model.

        logger.info("Building model...",)
        start = time.time()
        self.build_graph()
        logger.info("took %d s", time.time() - start)

        # Optimizer set-up:
        self.global_step = tf.train.create_global_step()
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()

        # Set up summaries
        self.add_metrics()
        self.summary = tf.summary.merge_all()

        self.saver = tf.train.Saver( var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) )



        restored = self.try_restore(sess, self.saver) if restore_weights else False
        if restored:
            logger.info("-- Restored model")
        if not restored:
            logger.info("-- Initializingt new model")
            self.sess.run(tf.global_variables_initializer()) 


        train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', sess.graph)
        dev_writer = tf.summary.FileWriter(self.summaries_dir + '/dev')

        for i in range(1, max_iter+1):
            logger.info("Starting epoch %d", i)

            sess.run(self.data_manager.initializer, feed_dict=self.data_manager.get_init_feed_dict('train')) 
            batch_cnt = 0
            loss = 0
            try:
                for it in range(BATCH_LIM):

                    _, summary, batch_loss = sess.run(
                        [self.optimizer, self.summary, self.cost],
                        feed_dict=self.train_dict()
                    )

                    train_writer.add_summary(summary,
                        tf.train.global_step(sess, self.global_step)
                    )
                    train_writer.flush()

                    batch_cnt += 1
                    if (batch_cnt > 0 and batch_cnt % batches_to_eval == 0):
                        logger.info(" == Evaluating on development data after batch %d", batch_cnt)

                        dev_labels, dev_preds, _word_ids, _email_ids = self.evaluate_tfdata(sess, 'dev', 
                                        batch_lim = n_val_batches, writer = dev_writer)
                        prec, rec, f1, _ = metrics.precision_recall_fscore_support(dev_labels, dev_preds, average='micro') 
                        logger.info("== Results for %d batches of size %d", n_val_batches, self.data_manager.config.batch_size)
                        logger.info("== Precision: %.3f, Recall: %.3f, F1: %.3f", prec, rec, f1 )

                        # Save model
                        if self.ckpts_prefix is not None:
                            logger.info("-- Saving model")
                            self.saver.save(sess, self.ckpts_prefix, global_step=self.global_step) 

                if (batch_cnt > 0 and batch_cnt % 10 == 0):
                    logger.info(" = batch %d", batch_cnt)

            except tf.errors.OutOfRangeError:
                pass

            logger.info("Finished epoch %d, ran %d batches", i + 1, batch_cnt)

        return self


    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it.
        """
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.model, labels=self.outputs))
        tf.summary.scalar('loss', cost)
        return cost

    def add_metrics(self):
        """
        Adds summary ops for precision, recall, accuracy. 
        """
        pred_labels = tf.cast(tf.argmax(self.model, axis=1), tf.float32)
        true_labels = tf.cast(tf.argmax(self.outputs, axis=1), tf.float32)

        for cl, cl_name in enumerate(LBLS):

            with tf.variable_scope('metrics-{}'.format(cl_name)) as vs:
                true_labels_cl = tf.equal(true_labels, tf.constant(float(cl)))
                pred_labels_cl = tf.equal(pred_labels, tf.constant(float(cl)))

                # rep_prec, rp_update = tf.metrics.precision(true_labels, pred_labels)
                prec = tf.count_nonzero(tf.logical_and(true_labels_cl, pred_labels_cl), dtype=tf.float32) / (1e-8 + tf.count_nonzero(pred_labels_cl, dtype=tf.float32))
                rep = tf.count_nonzero(tf.logical_and(true_labels_cl, pred_labels_cl), dtype=tf.float32) / (1e-8 + tf.count_nonzero(true_labels_cl, dtype=tf.float32))
                # rep_rec, rr_update = tf.metrics.recall(true_labels, pred_labels)
                f1 = 2 * prec * rep / (prec + rep + tf.constant(1e-8))

                tf.summary.scalar('precision-{}'.format(cl_name), prec)
                tf.summary.scalar('recall-{}'.format(cl_name), rep)
                tf.summary.scalar('f1-{}'.format(cl_name), f1)

    def get_optimizer(self):
        return tf.train.GradientDescentOptimizer(
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
        if init_dm:
            sess.run(self.dm.initializer, feed_dict=self.dm.get_init_feed_dict(dataset))
        self.probs = tf.nn.softmax(self.model)
        return self.sess.run(
            self.probs, self.inputs_placeholder, self.outputs, feed_dict=self.test_dict())

    def predict(self, init_dm=True, dataset='dev'):
        """Return classifier predictions, as the class with the
        highest probability for each example, for a single batch

        Returns
        -------
        list

        """
        probs, inputs, outputs = self.predict_proba(init_dm, dataset)
        return [(np.argmax(p), x, y) for p, x, y in zip(probs, inputs, outputs)]


    def evaluate_tfdata(self, sess, dataset_name='dev', batch_lim=20, writer=None):

        sess.run(self.data_manager.initializer, feed_dict=self.data_manager.get_init_feed_dict(dataset_name))

        preds = []
        labels = []
        word_ids = []
        email_ids = []
        batch_cnt = 0
        try:
            for it in range(batch_lim):

                summary, preds_batch_, labels_batch_, word_ids_batch, email_ids_batch = sess.run(
                    [self.summary, self.model, self.outputs, self.word_ids, self.ids_batch],
                    feed_dict=self.test_dict()
                )

                if writer is not None:
                    writer.add_summary(summary, tf.train.global_step(sess, self.global_step))
                    # writer.add_summary(summary)

                labels_batch = np.argmax(labels_batch_, axis=1)
                preds_batch = np.argmax(preds_batch_, axis=1)

                preds += list(preds_batch)
                labels += list(labels_batch)
                word_ids += list(word_ids_batch)
                email_ids += list(email_ids_batch)

                batch_cnt += 1
                if (batch_cnt > 0 and batch_cnt % 500 == 0):
                    logger.info(" - batch %d", batch_cnt)

        except tf.errors.OutOfRangeError:
            pass

        logger.info("Ran %d batches", batch_cnt) 

        return labels, preds, word_ids, email_ids


    def _onehot_encode(self, y):
        classmap = dict(zip(self.classes, range(self.output_dim)))
        y_ = np.zeros((len(y), self.output_dim))
        for i, cls in enumerate(y):
            y_[i][classmap[cls]] = 1.0
        return y_

    def _progressbar(self, msg, index):
        if self.display_progress and index % self.display_progress == 0:
            sysprint("Iteration {}: {}".format(index, msg))

    def weight_init(self, m, n, name):
        """
        Uses the Xavier Glorot method for initializing
        weights. This is built in to TensorFlow as
        `tf.contrib.layers.xavier_initializer`, but it's
        nice to see all the details.
        """
        x = np.sqrt(6.0/(m+n))
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.random_uniform(
                    [m, n], minval=-x, maxval=x), name=name)

    def bias_init(self, dim, name, constant=0.0):
        """Default all 0s bias, but `constant` can be
        used to specify other values."""
        with tf.name_scope(name) as scope:
            return tf.Variable(
                tf.constant(constant, shape=[dim]), name=name)

    def prepare_output_data(self, y):
        """Format `y` so that Tensorflow can deal with it, by turning
        it into a vector of one-hot encoded vectors.

        Parameters
        ----------
        y : list

        Returns
        -------
        np.array with length the same as y and each row the
        length of the number of classes

        """
        self.classes = sorted(set(y))
        self.output_dim = len(self.classes)
        y = self._onehot_encode(y)
        return y

    def get_params(self, deep=True):
        """Gets the hyperparameters for the model, as given by the
        `self.params` attribute. This is called `get_params` for
        compatibility with sklearn.

        Returns
        -------
        dict
            Map from attribute names to their values.

        """
        return {p: getattr(self, p) for p in self.params}

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self
