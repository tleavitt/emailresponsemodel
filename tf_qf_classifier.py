import tensorflow as tf
import numpy as np
from tf_model_base import TfModelBase

__author__ = 'Tucker Leavitt'


class TfQfClassifier(TfModelBase):
    """
    Parameters
    ----------
    hidden_dims : [int]
    max_iter : int
    eta : float
    tol : float

    """
    def __init__(self,
            vocab,
            hidden_dim=50, 
            hidden_layers=[],
            embedding=None,
            embed_dim=50,
            n_words=2,
            train_embedding=True,
            pos_features=True,
            reg=0.1, keep_prob=0.8,
            **kwargs):

        # super(TfMlpClassifier, self).__init__(**kwargs)
        super().__init__(**kwargs)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.embedding=embedding
        self.embed_dim = embed_dim
        self.n_words = n_words
        self.train_embedding = train_embedding
        self.reg = reg
        self._keep_prob = keep_prob
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=reg)

    def linear_layer(self, x, x_dim, y_dim, scope):
        '''
        y = dropout(activation(Wx + b))
        '''
        with tf.variable_scope(scope):
            W = self.weight_init(
                x_dim, y_dim, name='W')
            b = self.bias_init(
                y_dim, name='b')
        y = tf.matmul(x, W) + b
        return y

    def build_graph(self):
        """Builds a graph

        hidden = relu(xW_xh + b_h)
        model = softmax(hW_hy + b_y)
        """
        self._define_embedding()
        self._define_parameters()

        x = self.add_embedding()

        # The graph:
        x_dim = self.embed_dim
        x_prem = tf.squeeze(tf.slice(x, [0, 0, 0], [-1, 1, -1]), axis=1)
        x_hypo = tf.squeeze(tf.slice(x, [0, 1, 0], [-1, 1, -1]), axis=1)
        print('x_prem:', x_prem.get_shape())                                                                                        
        print('x_hypo:', x_hypo.get_shape())      

        W = tf.get_variable("W", 
            shape=[x_dim, self.hidden_dim, x_dim], 
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self.regularizer
        )
        b = tf.get_variable("b", shape=[self.hidden_dim])

        xW = tf.einsum('ij,jkl->ikl', x_prem, W)
        xWxt = tf.einsum('ikj,ij->ik', xW, x_hypo)
        h = tf.nn.dropout(self.hidden_activation(xWxt + b), self.keep_prob)
        print('h:', xWxt.get_shape())      

        dims = [self.hidden_dim] + self.hidden_layers
        cur_h = h
        for i, (x_dim, y_dim) in enumerate(zip(dims[:-1], dims[1:])):
            y = self.linear_layer(cur_h, x_dim, y_dim, 'layer{}'.format(i))
            cur_h = tf.nn.dropout(self.hidden_activation(y), self.keep_prob)

        # softmax layer
        self.model = self.linear_layer(cur_h, dims[-1], self.output_dim, 'softmax')

    def _define_parameters(self):
        # Input and output placeholders
        self.inputs = tf.placeholder(
            tf.int32, shape=[None, self.n_words])
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])
        self.keep_prob = tf.placeholder(
            tf.float32, shape=())

    
    def _define_embedding(self):
        """Build the embedding matrix. If the user supplied a matrix, it
        is converted into a Tensor, else a random Tensor is built. This
        method sets `self.embedding` for use and returns None.
        """
        if type(self.embedding) == type(None):
            self.embedding = np.random.uniform(size=[self.vocab_size, self.embed_dim], low=-1.0, high=1.0)


    def add_embedding(self):
        """Adds a trainable embedding layer.

        Returns:
            embeddings: tf.Tensor of shape (None, n_words, embed_dim)
        """
        all_embeddings = tf.get_variable('embeddings', 
            shape=self.embedding.shape, 
            initializer=tf.constant_initializer(self.embedding),
            trainable=self.train_embedding
        )     
        input_embeddings = tf.nn.embedding_lookup(
            params=all_embeddings, 
            ids=self.inputs
        )

        print('input_embeddings:', input_embeddings.get_shape())                                                                                        
        return input_embeddings




    def _convert_X(self, X):
        """Convert `X` to a list of list of indices into `self.vocab`,
        where all the lists have length `self.n_words`, which
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
        new_X = np.zeros((len(X), self.n_words), dtype='int')
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['$UNK']
        for i in range(new_X.shape[0]):
            vals = X[i][-self.n_words: ]
            vals = [index.get(w, unk_index) for w in vals]
            temp = np.zeros((self.n_words,), dtype='int')
            temp[0: len(vals)] = vals
            new_X[i] = temp
        return new_X

    # def feed_parse_pos_features(self, feed):

    def train_dict(self, X, y):
        X = self._convert_X(X)
        return {self.inputs: X, 
                self.outputs: y,
                self.keep_prob: self._keep_prob
        }

    def test_dict(self, X):
        X = self._convert_X(X)
        return {self.inputs: X, self.keep_prob: 1.0}


    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it.
        """
        data_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.model, labels=self.outputs))
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
        return data_loss + reg_loss

    def get_optimizer(self):
        return tf.train.AdamOptimizer(self.eta).minimize(self.cost)

def simple_example():
    vocab = ['a', 'b', '$UNK']

    train = [
        [list('ab'), 'good'],
        [list('ab'), 'good'],
        [list('aa'), 'good'],
        [list('ba'), 'bad'],
        [list('ba'), 'bad'],
        [list('bb'), 'bad'],
        [list('bb'), 'bad']]

    test = [
        [list('aa'), 'good'],
        [list('ba'), 'bad']]

    mod = TfQfClassifier(hidden_dim=10, embed_dim=10,
        vocab=vocab, max_iter=100, n_words=2)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()

