#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
from __future__ import print_function
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import os
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import nltk
import logging
from collections import Counter
import gzip
import json
import csv
from glob import glob
import copy

from scipy.stats import mstats
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf
from glove import loadWordVectorsIntoMemory
from util import one_hot, ConfusionMatrix, read_records
from glove import load_word_vector_mapping
from defs import LBLS, NONE, LMAP, NUM, UNK, EMBED_SIZE, MAX_LENGTH, PROJECT_DIR, N_CLASSES
from defs import FDIM, P_CASE, CASES, CASE2ID, START_TOKEN, END_TOKEN
from tfdata_helpers import parse_tfrecord, tf_record_parser, tf_filename_func

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class DMConfig(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_classes = N_CLASSES
    batch_size = 32
    collapse_classes = True
    embed_size = EMBED_SIZE
    n_embeddings = 10000
    len_cutoff = 500 # only consider emails of less than 500 tokens.
    max_length = MAX_LENGTH # all emails will be extended or truncated to have this length.
    train_filepath = '{}/processed-data/records0_train.pkl.gz'.format(PROJECT_DIR)
    # train_filepath = '{}/processed-data/bl_records0_train.pkl.gz'.format(PROJECT_DIR)
    # train_filepath = '{}/processed-data/buy_records0_train.pkl.gz'.format(PROJECT_DIR)
    dev_filepath = '{}/processed-data/records0_dev.pkl.gz'.format(PROJECT_DIR)
    # dev_filepath = '{}/processed-data/bl_records0_dev.pkl.gz'.format(PROJECT_DIR)
    # dev_filepath = '{}/processed-data/buy_records0_dev.pkl.gz'.format(PROJECT_DIR)
    # test_filepath = '{}/processed-data/email_records0_test.pkl.gz'.format(PROJECT_DIR)
    test_filepath = '{}/processed-data/records0_test.pkl.gz'.format(PROJECT_DIR)
    # test_filepath = '{}/processed-data/buy_records0_test.pkl.gz'.format(PROJECT_DIR)

    dev_log_file = '{}/lstm/devpredictions.log'.format(PROJECT_DIR)
    dm_save_dir = '{}/lstm/'.format(PROJECT_DIR)
    tok2id_path = '{}/data/tok2id_20k.pkl.gz'.format(PROJECT_DIR)
    id2tok_path = '{}/data/id2tok_20k.pkl.gz'.format(PROJECT_DIR)

    vocab_filepath = '{}/data/vocab.txt'.format(PROJECT_DIR)
    vectors_filepath = '{}/data/wordVectors.txt'.format(PROJECT_DIR)
    one_hot = True # whether labels should be one hot vectors or not
    should_tokenize = True # whether we should tokenize vectors as we load them.

def casing(word):
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def featurize(embeddings, word):
    """
    Featurize a word given embeddings.
    """
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))

class DataManager(object):

    def log_predictions(self, dev_true_labels, dev_pred_labels):
        with open(self.config.dev_log_file, 'w') as f:
            writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['True', 'Pred', 'Message Id', 'Email body'])
            for l_true, l_pred, email in zip(dev_true_labels, dev_pred_labels, self.dev_emails):
                writer.writerow( ('{} '.format(l_true), l_pred, email['Id'], json.dumps(
                    email['Body'] if self.config.should_tokenize else ''.join(email['Body'])
                )) )
                # writer.writerow([l_true, l_pred, json.dumps(email.tolist())])

    def get_dev_examples(self):
        raise "Override me!"

    # returns (features_list, labels_list)
    def get_dev_feats_and_labels(self):
        raise "Override me!"

    def get_train_examples(self):
        raise "Override me!"

    # returns (features_list, labels_list)
    def get_train_feats_and_labels(self):
        raise "Override me!"

class StaticDataManager(DataManager):

    def __init__(self, config, tok2id=None):
        self.config = config
        self.train_emails, self.dev_emails, self.test_emails = load_and_preprocess_data(config)
        pdb.set_trace()
        if tok2id is not None:
            self.tok2id = tok2id
        else:
            self.tok2id = build_tok2id(self.train_emails, self.config.n_embeddings, self.config.should_tokenize)

        # now process all the input data.
        self.vectorizer = EmailVectorizer(config, self.tok2id)
        self.train_data = self.vectorizer.vectorize_emails(self.train_emails)
        self.dev_data = self.vectorizer.vectorize_emails(self.dev_emails)
        self.test_data = self.vectorizer.vectorize_emails(self.test_emails)

    def get_dev_examples(self):
        return self.dev_data

    def get_dev_feats_and_labels(self):
        return zip(*self.dev_data) 

    def get_train_examples(self):
        return self.train_data

    def get_train_feats_and_labels(self):
        return zip(*self.train_data) 

    def get_test_examples(self):
        return self.test_data


    def save(self):
        # Make sure the directory exists.
        path = self.config.dm_save_dir
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with gzip.open(os.path.join(path, "dm.pkl.gz"), "w") as f:
            cPickle.dump(self, f)


def build_tok2id(email_examples, n_embeddings=10000, should_tokenize=False):
    # Preprocess data to construct an embedding
    # Reserve 0 for the special UNK token.
    body_parser = nltk.word_tokenize if should_tokenize else lambda b: b
    tok2id = build_dict(
        (normalize(token) for email in email_examples for token in body_parser(email['Body'])), 
        offset=1, max_words=n_embeddings
    )
    tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
    tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
    assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
    logger.info("Built dictionary for %d features.", len(tok2id))

    return tok2id

def dm_load(config):
    # Make sure the directory exists.
    path = config.dm_save_dir
    assert os.path.exists(path) and os.path.exists(os.path.join(path, "dm.pkl.gz"))
    # Save the tok2id map.
    with gzip.open(os.path.join(path, "dm.pkl.gz")) as f:
        dm = cPickle.load(f)
    return dm

def load_and_preprocess_data(config):
    logger.info("Loading training data...")
    train_emails = read_records(config.train_filepath)
    logger.info("Done. Read %d emails", len(train_emails))
    logger.info("Loading dev data...")
    dev_emails = read_records(config.dev_filepath)
    logger.info("Done. Read %d emails", len(dev_emails))
    logger.info("Loading test data...")
    test_emails = read_records(config.test_filepath)
    logger.info("Done. Read %d emails", len(test_emails))
    return train_emails, dev_emails, test_emails


def load_embeddings(config, tok2id):
    embeddings = np.array(np.random.randn(len(tok2id) + 1, config.embed_size), dtype=np.float32)
    embeddings[0] = 0.
    with open(config.vocab_filepath, 'r') as vocf, open(config.vectors_filepath, 'r') as vecf:
        for word, vec in load_word_vector_mapping(vocf, vecf).items():
            word = normalize(word)
            if word in tok2id:
                embeddings[tok2id[word]] = vec
    logger.info("Initialized embeddings.")

    return embeddings

def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}


class TestDataManager(DataManager):

    def __init__(self, config, min_length, max_length):
        self.config = config
        self.min_length = min_length
        self.max_length = max_length
        self.train_data = self.generate_sequence(1000)
        self.dev_data = self.generate_sequence(100)
   
    def log_predictions(self, dev_true_labels, dev_pred_labels):
        pass

    def get_dev_examples(self):
        return zip(*self.dev_data)

    # returns (features_list, lens_list, labels_list)
    def get_dev_feats_and_labels(self):
        return self.dev_data

    def get_train_examples(self):
        return zip(*self.train_data)

    # returns (features_list, lens_list, labels_list)
    def get_train_feats_and_labels(self):
        return self.train_data

    def generate_sequence(self, n_samples=10000):
        """
        Generates a sequence like a [0]*n a
        """
        seqs = []
        lens = []
        labels = []
        for _ in range(int(n_samples/2)):
            len1 = np.random.randint(self.min_length, self.max_length)
            ft1 = [[1.,1.]] + ([[1.,1.]] * len1) + ([[0.,0.]] * (self.max_length - len1 - 1))
            seqs.append(ft1); lens.append(len1); labels.append([1., 0.])

            len2 = np.random.randint(self.min_length, self.max_length)
            ft2 = [[2.,2.]] + ([[1.,1.]] * len2) + ([[0.,0.]] * (self.max_length - len2 - 1))

            seqs.append(ft2); lens.append(len2); labels.append([0., 1.])
        return np.asarray(seqs), np.asarray(lens), np.asarray(labels)

    def test_generate_sequence(self):
        max_length = 20
        for seq, y in generate_sequence(20):
            assert len(seq) == max_length
            assert seq[0] == y


class EmailVectorizer(object):
    def __init__(self, config, tok2id):
        self.config = config
        self.tok2id = tok2id
        self.case2id = CASE2ID
        self.START = [self.tok2id[START_TOKEN], self.case2id[casing("a")]]
        self.END = [self.tok2id[END_TOKEN], self.case2id[casing("a")]]
        self.UNK = self.tok2id[UNK]

    def featurize_email(self, body_toks, pad=True):
        body_toks = body_toks[:self.config.max_length - 2] # truncate if necessary.
        if len(body_toks) == 0:
            return [], 0

        features = [self.START]
        features += [[self.tok2id.get(normalize(word), self.tok2id[UNK]) , self.case2id[casing(word)]] for word in body_toks]
        features += [self.END]
        num_feats = len(features)
        if pad:
            features += [[0, 0]] * (self.config.max_length - num_feats) # pad with NILs
        return features, num_feats

    def email_vectors_generator(self, email_examples):
        for email in email_examples:
            body, label = email['Body'], email['Label']
            body_toks = nltk.word_tokenize(body) if self.config.should_tokenize else body
            if len(body_toks) + 2 > self.config.len_cutoff:
                continue
            features, num_feats = self.featurize_email(body_toks)

            if label:
                if self.config.collapse_classes and label == 2:
                    label = 1
            else:
                label = 0
            yield features, num_feats, one_hot(self.config.n_classes, label) if self.config.one_hot else label

    def vectorize_emails(self, email_examples):
        return [ex for ex in self.email_vectors_generator(email_examples)]



class tfConfig(DMConfig):
    shuffle_buffer_size = 5000
    epochs_per_init = 1
    train_fn_tag = tf_filename_func('train', '*')
    dev_fn_tag = tf_filename_func('dev', '*')
    test_fn_tag = tf_filename_func('test', '*')

class tfDatasetManager(DataManager):
    filenames_placeholder = None
    iterator = None

    train_filenames = []
    dev_filenames = []
    test_filenames = []

    def initialize_iterators(self, sess):
        sess.run(
           [self.train_iterator.initializer, 
            self.dev_iterator.initializer, 
            self.test_iterator.initializer]
        ) 

        self.handle_train, self.handle_dev, self.handle_test =\
            sess.run(
                [self.train_iterator.string_handle(), 
                 self.dev_iterator.string_handle(),
                 self.test_iterator.string_handle()]
            ) 

    def init_filenames(self):
        self.train_filenames = glob(self.config.train_fn_tag)
        self.dev_filenames = glob(self.config.dev_fn_tag)
        self.test_filenames = glob(self.config.test_fn_tag)

    def init_dataset(self, dataset):
        dataset = dataset.map(tf_record_parser(one_hot=self.config.one_hot))  # Parse the record into tensors.
        dataset = dataset.repeat(self.config.epochs_per_init)  # Repeat the input indefinitely.
        dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer_size)
        dataset = dataset.batch(self.config.batch_size)
        return dataset

    def init_datasets(self):
        # with tf.Graph().as_default():
        self.train_fn_tensor = tf.constant(self.train_filenames)
        self.dev_fn_tensor = tf.constant(self.dev_filenames)
        self.test_fn_tensor = tf.constant(self.test_filenames)

        train_dataset = tf.data.TFRecordDataset(self.train_fn_tensor)
        self.train_dataset = self.init_dataset(train_dataset)
        self.train_iterator = self.train_dataset.make_initializable_iterator()

        dev_dataset = tf.data.TFRecordDataset(self.dev_fn_tensor)
        self.dev_dataset = self.init_dataset(dev_dataset)
        self.dev_iterator = self.dev_dataset.make_initializable_iterator()

        test_dataset = tf.data.TFRecordDataset(self.test_fn_tensor)
        self.test_dataset = self.init_dataset(test_dataset)
        self.test_iterator = self.test_dataset.make_initializable_iterator()

        # make feedable iterator
        self.handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
                self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.iterator = iterator

        self.next_batch = self.iterator.get_next()


    @property
    def batch_op(self):
        return self.next_batch

    def __init__(self, config):
        self.config = config
        self.init_filenames()
        self.init_datasets()

def test_main4():
    config = tfConfig()
    tfdm = tfDatasetManager(config)
    with gzip.open(config.id2tok_path, 'r') as f:
        id2tok = cPickle.load(f)
    with gzip.open(config.tok2id_path, 'r') as f:
        tok2id = cPickle.load(f)

    class_counts = np.zeros(config.n_classes)
    feature_counters = [Counter() for _ in range(config.n_classes)]

    with tf.Session() as sess:
        sess.run(tfdm.initializer, feed_dict=tfdm.get_init_feed_dict('train'))
        it = 0
        try:
            while True:
                feats_batch, lens_batch, labels_batch, ids_batch = sess.run(tfdm.batch_op)                
                for feats, length, label in zip(feats_batch, lens_batch, labels_batch):
                    word_ids = feats[1:length-1][:,0] # strip start, end tok, last character, and case features
                    l = np.argmax(label)
                    feature_counters[l].update(word_ids)
                    class_counts[l] += word_ids.shape[0]
                it += 1
                if (it % 100 == 0):
                    print("it: {}, num tokens seen: {}".format(it,
                        reduce(lambda cum, fc: cum + len(fc), feature_counters, 0)
                    ))
        except tf.errors.OutOfRangeError:
            print("Finished reading training set.")


    print("Total number of tokens counted: {}".format( 
        reduce(lambda cum, fc: cum + len(fc), feature_counters, 0)
    ))
    common_counts = reduce(lambda a, b: a & b, feature_counters) # intersection of all counts

    n_top_words = 100
    if config.n_classes == 3:
        feature_counters[1] = feature_counters[1] + feature_counters[2] 
        del feature_counters[2]
        class_counts[1] += class_counts[2]

    common_counts = [Counter() for fc in feature_counters]
    for i, fc in enumerate(feature_counters):
        '''
        Most unique words of the ith class
        '''
        # other_idxs = [j for j in range(config.n_classes) if j != i]
        # characteristic_words = reduce(lambda cur, other_idx: cur - feature_counters[other_idx], other_idxs, feature_counters[i])
        # if config.n_classes > 2:
        #     # we subtracted common counts twice, add them back.
        #     characteristic_words = characteristic_words + common_counts
        # remove stop words
        for word in stopwords.words('english'):
            wid = tok2id.get(word, -1)
            if wid in fc:
                fc[wid] = 0

        print("Top {} most common words for class {}:".format(n_top_words, i))
        it = 0
        n_words = 0
        for wid, count in fc.most_common():
            it += 1
            tok = id2tok[wid]
            if tok == UNK or tok == NUM or not tok.isalnum():
                # print("unk or punct")
                continue
            # if tok in stopwords.words('english'):
            #     print("stop word")
            #     continue
            print("== {}: {}".format(tok, count))

            common_counts[i][tok] = float(count)/class_counts[i]

            n_words += 1
            if n_words >= n_top_words:
                break
        print("it: {} n_words: {}".format(it, n_words))

    return common_counts

def test_main3():
    config = tfConfig()
    tfdm = tfDatasetManager(config)

    with tf.Session() as sess:
        sess.run(tfdm.initializer, feed_dict=tfdm.get_init_feed_dict('train'))
        len_lim = 2000
        num_outliers = 0
        lens_list = []
        class_counts = np.zeros(config.n_classes)
        num_train = 0
        it = 0
        try:
            while True:
                feats, lens, labels, ids_batch = sess.run(tfdm.batch_op)                
                lens_list += filter(lambda l: l < len_lim, lens)
                num_outliers += reduce(lambda cum, l: cum + 1 if l >= len_lim else cum, lens, 0)
                num_train += feats.shape[0]
                for i in range(config.n_classes):
                    class_counts[i] += np.count_nonzero(labels[:, i])
                it += 1
                if (it % 100 == 0):
                    print("it: {}, counted {} examples, class counts: {}".format(it, num_train, class_counts))
        except tf.errors.OutOfRangeError:
            print("Finished reading training set.")

        m = mstats.mquantiles(lens_list)
        # pdb.set_trace()

        print("Total num_train: {}".format(num_train))
        print("Class counts:")
        for i in range(config.n_classes):
            print("= {}: {}".format(i, class_counts[i]))
        print("Instances with len geq {}: {}".format(len_lim, num_outliers))
        plot_lens_hist(lens_list)

    # pdb.set_trace()
    # # plt.hist(train_ids, bins=40)
    # # plt.title("Dev word_id distribution")
    # plt.hist(dev_labels, bins=40)
    # plt.title("Dev label distribution")
    # plt.show()

def plot_lens_hist(lens_list, filename='lens_dist.png'):
    plt.rc('font', size=24)
    fig, ax = plt.subplots()
    ax.hist(lens_list, bins=100)
    plt.title('Length frequencies over the training dataset')
    # ax.set_yticklabels([])
    ax.set_ylabel('Relative frequency')
    ax.set_xlabel('Email length in tokens.')
    fig.savefig(filename, format='png') if filename else plt.show() 

def create_and_save_tok2id():
    config = DMConfig()
    train_emails, dev_emails, test_emails = load_and_preprocess_data(config)

    tok2id = build_tok2id(train_emails, config.n_embeddings, True)
    id2tok = {tok2id[k]:k for k in tok2id}

    with gzip.open(config.tok2id_path, 'w') as f:
        cPickle.dump(tok2id, f)

    with gzip.open(config.id2tok_path, 'w') as f:
        cPickle.dump(id2tok, f)

def test_main1():
    config = DMConfig()
    with gzip.open(config.tok2id_path, 'r') as f:
        tok2id = cPickle.load(f)
    sdm = StaticDataManager(DMConfig(), tok2id)
    # tdm = TestDataManager(2, 50)
    # print(sdm.get_dev_examples()[0])
    fig, ax = plt.subplots()
    ax.hist(map(lambda ex: len(ex[0]), sdm.get_train_examples()), bins=40)
    plt.title()
    ax.set_yticklabels([])
    ax.set_ylabel('Relative frequency')
    ax.set_ylabel('Email length in tokens.')
    # plt.hist(map(lambda ex: ex[1], sdm.get_train_examples()), bins=40)
    # plt.title("Data length distribution")
    # plt.hist(map(lambda ex: np.argmax(ex[2]), sdm.get_dev_examples()), bins=40)
    # plt.title("Train class distribution")
    # plt.hist(map(lambda email: len(email['Id']), sdm.dev_emails), bins=40)
    # plt.title("Dev label len distribution")
    plt.show()


if __name__ == "__main__":
    # test_main4()
    # test_main3()
    # test_main1()
    create_and_save_tok2id()
