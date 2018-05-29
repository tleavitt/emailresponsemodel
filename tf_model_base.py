#!/usr/bin/env python3

import numpy as np
import pandas as pd
import random
import sys
import tensorflow as tf
from defs import BATCH_LIM, PROJECT_DIR
from data_util import tfDatasetManager, tfConfig
from util import sysprint
from sklearn import metrics
import time

__author__ = 'Tucker Leavitt, Chris Potts'

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class TfModelBase(object):
    """
    Subclasses need only define `build_graph`, `train_dict`, and
    `test_dict`. (They can redefine other methods as well.)

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
    def __init__(self, hidden_dim=50, hidden_activation=tf.nn.relu,
            batch_size=32, max_iter=100, eta=0.005, tol=1e-4, display_progress=1,
            summaries_dir = os.path.abspath('{}/summaries'.format(PROJECT_DIR)) ):
        self.hidden_dim = hidden_dim
        self.hidden_activation = tf.nn.tanh
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eta = eta
        self.tol = tol
        self.display_progress = display_progress
        self.errors = []
        self.dev_predictions = []
        self.summaries_dir = summaries_dir
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


    def fit_tfdata(self, tfconfig, **kwargs):
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
        sess = tf.InteractiveSession()
        self.sess = sess

        # Set up record keepers
        saver = tf.train.Saver()

        # Build the computation graph. This method is instantiated by
        # individual subclasses. It defines the model.
        logger.info("Building model...",)
        start = time.time()
        self.build_graph()
        logger.info("took %d s", time.time() - start)

        # Optimizer set-up:
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()

        # Set up summaries
        self.add_metrics()
        train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', sess.graph)
        dev_writer = tf.summary.FileWriter(self.summaries_dir + '/dev')
        self.summary = tf.summary.merge_all()

        # Initialize the session variables:
        self.sess.run(tf.global_variables_initializer())

        # Initialize datamanager
        logger.info("Initializing data manager...",)
        start = time.time()
        tfdm = tfDatasetManager(tfconfig)
        self.data_manager = tfdm
        logger.info("took %d s", time.time() - start)

        # Training, full dataset for each iteration:
        for i in range(1, self.max_iter+1):
            self._progressbar("starting epoch", i)
            sess.run(tfdm.initializer, feed_dict=tfdm.get_init_feed_dict('train'))

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
                if (batch_cnt > 0 and batch_cnt % 10 == 0):
                    logger.info("Evaluating on development data")

                    dev_labels, dev_preds = self.evaluate_tfdata(sess, 'dev', 
                                    batch_lim = 20, writer = dev_writer)
                    logger.info("++ Development set results:")
                    logger.info("++ Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}".format(
                        metrics.precision(dev_labels, dev_preds),
                        metrics.recall(dev_labels, dev_preds),
                        metrics.f1_score(dev_labels, dev_preds)
                    ))

                if (batch_cnt > 0 and batch_cnt % 10 == 0):
                    logger.info(" = batch %d", batch_cnt)

            except tf.errors.OutOfRangeError:
                pass

            logger.info("Finished epoch %d, ran %d batches", epoch + 1, batch_cnt)

        return self

    def fit(self, X, y, **kwargs):
        """Standard `fit` method.

        Parameters
        ----------
        X : np.array
        y : array-like
        kwargs : dict
            For passing other parameters, e.g., a test set that we
            want to monitor performance on.

        Returns
        -------
        self

        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Incremental performance:
        X_dev = kwargs.get('X_dev')
        if X_dev is not None:
            dev_iter = kwargs.get('test_iter', 10)

        # One-hot encoding of target `y`, and creation
        # of a class attribute.
        y = self.prepare_output_data(y)

        self.input_dim = len(X[0])

        # Start the session:
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()

        # Build the computation graph. This method is instantiated by
        # individual subclasses. It defines the model.
        self.build_graph()

        # Optimizer set-up:
        self.cost = self.get_cost_function()
        self.optimizer = self.get_optimizer()

        # Initialize the session variables:
        self.sess.run(tf.global_variables_initializer())

        # Training, full dataset for each iteration:
        for i in range(1, self.max_iter+1):
            loss = 0

            for X_batch, y_batch in self.batch_iterator(X, y):
                _, summary, batch_loss = self.sess.run(
                    [self.optimizer, self.summary, self.cost],
                    feed_dict=self.train_dict(X_batch, y_batch))
                loss += batch_loss
            self.errors.append(loss)
            if X_dev is not None and i > 0 and i % dev_iter == 0:
                self.dev_predictions.append(self.predict(X_dev))
            if loss < self.tol:
                self._progressbar("stopping with loss < self.tol", i)
                break
            else:
                self._progressbar("loss: {}".format(loss), i)
        return self

    def batch_iterator(self, X, y):
        dataset = zip(X, y)
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i: i+self.batch_size]
            X_batch, y_batch = zip(*batch)
            yield X_batch, y_batch

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

        with tf.variable_scope('metrics') as vs:
            # rep_prec, rp_update = tf.metrics.precision(true_labels_rep, pred_labels_rep)
            prec = tf.count_nonzero(tf.logical_and(true_labels_rep, pred_labels_rep), dtype=tf.float32) / (1e-8 + tf.count_nonzero(pred_labels_rep, dtype=tf.float32))
            rep = tf.count_nonzero(tf.logical_and(true_labels_rep, pred_labels_rep), dtype=tf.float32) / (1e-8 + tf.count_nonzero(true_labels_rep, dtype=tf.float32))
            # rep_rec, rr_update = tf.metrics.recall(true_labels_rep, pred_labels_rep)
            f1 = 2 * prec * rep / (prec + rep + tf.constant(1e-8))

        tf.summary.scalar('precision', prec)
        tf.summary.scalar('recall', rec)
        tf.summary.scalar('f1', f1)

    def get_optimizer(self):
        return tf.train.GradientDescentOptimizer(
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
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Returns:
            the labels and predictions on the given dataset
        """
        tfdm = self.data_manager

        sess.run(tfdm.initializer, feed_dict=tfdm.get_init_feed_dict(dataset_name))

        preds = []
        labels = []
        batch_cnt = 0
        try:
            for it in range(batch_lim):

                summary, preds_batch_, labels_batch_ = sess.run(
                    [self.summary, self.model, self.outputs],
                    feed_dict=self.test_dict()
                )

                if writer is not None:
                    writer.add_summary(summary, tf.train.global_step(sess, self.global_step))
                    # writer.add_summary(summary)

                labels_batch = np.argmax(labels_batch_, axis=1)
                preds_batch = np.argmax(preds_batch_, axis=1)

                preds += list(preds_batch)
                labels += list(labels_batch)

                batch_cnt += 1
                if (batch_cnt > 0 and batch_cnt % 500 == 0):
                    logger.info(" - batch %d", batch_cnt)

        except tf.errors.OutOfRangeError:
            pass

        logger.info("Ran %d batches", batch_cnt) 

        return labels, pred


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
