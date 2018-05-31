import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
import warnings
import pdb

__author__ = 'Chris Potts'

N_CASES = 4

# Ignore the TensorFlow warning
#   Converting sparse IndexedSlices to a dense Tensor of unknown shape.
#   This may consume a large amount of memory.
warnings.filterwarnings("ignore", category=UserWarning)

class CNNConfig():  

    in_channels = 2
    # filter_word_widths = [3, 4, 7, 10]
    # filter_word_widths = [1, 3, 4, 7]
    filter_word_widths = [1, 2]
    # filter_word_widths = [[3, 4, 5], [5, 6, 7]]
    time_stride = 1
    out_channels = 150
    # out_channels = [100, 100]
    # padding = 'SAME'
    padding = 'VALID'
    keep_prob = 0.5
    max_pool = False
    

class TfCNNClassifier(TfModelBase):
    """Defines an CNN in which the final hidden state is used as
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
            embed_dim=50,
            max_length=52,
            hidden_activation = tf.nn.relu,
            filter_widths=None,
            out_channels=None,
            keep_prob=None,
            case_features=False,
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.train_embedding = True
        self.config = CNNConfig()
        self.hidden_activation = hidden_activation
        super().__init__(**kwargs)
        # self.eta = self.config.lr
        if filter_widths is not None:
            self.config.filter_word_widths = filter_widths
        if out_channels is not None:
            self.config.out_channels = out_channels
        if keep_prob is not None:
            self.config.keep_prob = keep_prob
        self.case_features = case_features
        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']

    def define_embedding(self):

        if type(self.embedding) == type(None):
            self.embedding = np.random.uniform(size=[self.vocab_size, self.embed_dim], low=-1.0, high=1.0)


    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors
        """
        self.inputs_placeholder = tf.placeholder(tf.int64, 
            shape=(None, self.max_length), name="inputs")
        self.cases_placeholder = tf.placeholder(tf.float32, 
            shape=(None, self.max_length, N_CASES), name="inputs_cases")
        self.outputs = tf.placeholder(tf.float32, 
            shape=(None, self.output_dim), name="labels")
        self.keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

    def add_embedding(self):
        """Adds a trainable embedding layer.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_dim)
        """
        all_embeddings = tf.get_variable('embeddings', 
            shape=self.embedding.shape, 
            initializer=tf.constant_initializer(self.embedding),
            trainable=True
        )     
        input_embeddings = tf.nn.embedding_lookup(
            params=all_embeddings, 
            ids=self.inputs_placeholder
        )                                                                                                          
        embeddings = tf.reshape(input_embeddings, 
            (-1, self.max_length, self.embed_dim)
        )                                                                                                
        return embeddings

    def add_static_embedding(self):
        """Adds a static, non-trainable embedding layer.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_dim)
        """
        static_embeddings = tf.get_variable('static_embeddings', 
            shape=self.embedding.shape, 
            initializer=tf.constant_initializer(self.embedding),
            trainable=False
        )     
        input_embeddings = tf.nn.embedding_lookup(
            params=static_embeddings, 
            ids=self.inputs_placeholder
        )                                                                                                          
        embeddings = tf.reshape(input_embeddings, 
            (-1, self.max_length, self.embed_dim)
        )                                                                                                
        return embeddings

    def pool_over_time(self, h, height, n_channels):
        # do max pooling over time:
        batch_size = tf.shape(h)[0]
        h_pool = tf.nn.max_pool(h,
            ksize = [1, height, 1, 1],
            strides = [1, 1, 1, 1],
            padding = 'VALID'
        )
        # now, h_pool will have shape (batch_size, 1, 1, self.config.out_channels) 
        print('h_pool:', h_pool.get_shape())
        h_out = tf.reshape(h_pool, [batch_size, n_channels])
        print('h_out:', h_out.get_shape())

        return h_out

    # applies a convolution layer widh width @window_size and name @i to the input @x
    # output has shape (batch_size, self.config.out_channels)
    def add_conv_layer(self, x, window_size, i):

        print("x: ", x.get_shape())
        with tf.variable_scope('conv{}'.format(i)):
            filter_shape = [window_size, self.n_word_features, self.config.in_channels, self.config.out_channels]
            w = tf.get_variable('filter',
                shape=filter_shape,
                initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            )
            b = tf.get_variable('bias',
                shape=(self.config.out_channels,),
                initializer=tf.constant_initializer(0)
            )

            if self.config.padding == "VALID":
                conv_outs = tf.nn.conv2d(x, w,
                    strides=[1, self.config.time_stride, 1, 1],
                    padding='VALID',
                ) + b
            else:
                # need to pad the time axis but not the word vector axis.
                batch_size = tf.shape(x)[0]
                num_pad =  np.max([window_size - self.config.time_stride, 0])
                x_pad = tf.concat([
                    tf.zeros([batch_size, num_pad, self.embed_dim, self.config.in_channels]),
                    x,
                    tf.zeros([batch_size, num_pad, self.embed_dim, self.config.in_channels])
                ], axis = 1)

                print("x_pad: ", x_pad.get_shape())
                conv_outs = tf.nn.conv2d(x_pad, w,
                    strides=[1, self.config.time_stride, 1, 1],
                    padding='VALID',
                ) + b

            # with valid, conv_outs has shape (batch_size, (max_length - window_size + 1)/time_stride, 1, self.config.out_channels)
            # with same, conv_outs has shape (batch_size, (max_length)/time_stride, 1, self.config.out_channels)
            # see: https://www.tensorflow.org/versions/r1.4/api_guides/python/nn#Convolution
            # consider: batch normalization?
            h = self.hidden_activation(conv_outs)
            print("h: ", h.get_shape())

        return h

    def parse_case_features(self, x):
        if self.case_features:
            return tf.concat([x, self.cases_placeholder], axis=-1)
        else:
            return x

    def add_prediction_op(self):
        self.n_word_features = self.embed_dim + N_CASES if self.case_features else self.embed_dim
        x_var = self.parse_case_features(self.add_embedding())
        x_static = self.parse_case_features(self.add_static_embedding())
        x = tf.stack([x_var, x_static], axis=-1) # concatenate the inputs on the last axis
        # x has shape (batch_size, max_length, n_word_features, 2) 

        # lens = self.lens_placeholder         
        # note: we assume that all word features beyond the length of the sentence are zero,
        # i.e. x[:, lens:, :, :] == 0 

        filter_outs = []
        for i, window_size in enumerate(self.config.filter_word_widths):
            h = self.add_conv_layer(x, window_size, i)
            if self.config.padding == "VALID":
                conv_height = np.ceil((self.max_length - window_size + 1.0)/self.config.time_stride)
            else:
                conv_height = np.ceil(self.max_length/self.config.time_stride) + window_size - self.config.time_stride


            filter_outs.append(
                self.pool_over_time(h, conv_height, self.config.out_channels)
            )

        h_concat = tf.concat(filter_outs, axis=1) 
        h_drop = tf.nn.dropout(h_concat, self.keep_prob) 


        fc_dim = [len(filter_outs) * self.config.out_channels, self.output_dim]
        with tf.variable_scope('fc'):
            W = tf.get_variable('W',
                shape=fc_dim,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.get_variable('b',
                shape=self.output_dim,
                initializer=tf.constant_initializer(0)
            )
        preds = tf.matmul(h_drop, W) + b
        # preds = tf.matmul(h_concat, W) + b

        return preds



    def build_graph(self):
        self.define_embedding()
        self.add_placeholders()
        self.model = self.add_prediction_op()


    def train_dict(self, X, y):
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
        X, ex_lengths, cases = self._convert_X(X)
        return self._feed_parse_cases({
            self.inputs_placeholder: X, 
            self.outputs: y,
            self.keep_prob: self.config.keep_prob
        }, cases)

    def test_dict(self, X):
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
        X, ex_lengths, cases = self._convert_X(X)

        return self._feed_parse_cases({
            self.inputs_placeholder: X,
            self.keep_prob: 1.0
        }, cases)

    def _convert_X(self, X):
        """Convert `X` to a list of list of indices into `self.vocab`,
        where all the lists have length `self.max_length`, which
        truncates the beginning of longer sequences and zero-pads the
        end of shorter sequences.

        Parameters
        ----------
        X : array-like
            The rows must be lists of objects in `self.vocab`.

        Returns
        -------
        np.array of int-type objects
            List of list of indices into `self.vocab`
        """
        new_X = np.zeros((len(X), self.max_length), dtype='int')
        cases = np.zeros((len(X), self.max_length, N_CASES), dtype='float')
        ex_lengths = []
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_tok = ('$UNK', 3) if self.case_features else '$UNK'
        unk_index = index[unk_tok]
        for i in range(new_X.shape[0]):
            ex_lengths.append(len(X[i]))
            vals = X[i][-self.max_length: ]
            if self.case_features:
                vals, cur_case = zip(*vals)
                cases[i][range(len(vals)), cur_case] = 1.0
            vals = [index.get(w, unk_index) for w in vals]
            temp = np.zeros((self.max_length,), dtype='int')
            temp[0: len(vals)] = vals
            new_X[i] = temp

        return new_X, ex_lengths, cases


    def _feed_parse_cases(self, feed, cases):
        if self.case_features:
            feed[self.cases_placeholder] = cases
        return feed

    # override to use Adam
    def get_optimizer(self):
        return tf.train.AdamOptimizer(
            self.eta).minimize(self.cost)




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
