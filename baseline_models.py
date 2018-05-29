#!/usr/bin/env python3
from __future__ import print_function

try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import os
import pdb
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics
import numpy as np
from scipy import stats
import nltk
from util import get_minibatches
from glove import loadWordVectorsIntoMemory
from preprocess import read_records
try:
    from io import StringIO 
except ImportError:
    from io import StringIO

from data_util import StaticDataManager, DMConfig, load_embeddings


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_classes = 3
    n_features = 2

    n_embeddings = 50000
    C = 0.1
    gamma = 1.0
    train_filepath = './processed-data/email_records0_train.pkl.gz'
    dev_filepath = './processed-data/email_records0_dev.pkl.gz'
    test_filepath = './processed-data/email_records0_test.pkl.gz'

def get_word_vectors(email_feats, embeddings):
    return map(
        lambda feat_list: np.mean([embeddings[feat] for feat in feat_list], axis=0),
        email_feats
    )
    
def get_avg_word_vectors(feats_list, embeddings):
    return map(
        lambda embedding_list: np.mean(embedding_list, axis = 0),
        map( lambda email_feats: get_word_vectors(email_feats, embeddings),
            feats_list
        )
    )

class AvgModel(object):
    embeddings = None
    model = None
    config = None

    def fit(self, word_feats, labels):
        avg_word_vectors = get_avg_word_vectors(word_feats, self.embeddings)
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        return self.model.fit(avg_word_vectors, labels)

    def predict(self, word_feats):
        avg_word_vectors = get_avg_word_vectors(word_feats, self.embeddings)
        return self.model.predict(avg_word_vectors)

class AvgLogisticRegressionModel(AvgModel):
    def __init__(self, embeddings, config, class_weight=None):
        self.embeddings = embeddings
        self.config = config
        self.model = linear_model.LogisticRegression(class_weight=class_weight)

class AvgRandomForestModel(AvgModel):
    def __init__(self, embeddings, config):
        self.embeddings = embeddings
        self.config = config
        self.model = ensemble.RandomForestClassifier()

class AvgSvmModel(AvgModel):
    def __init__(self, embeddings, config):
        self.embeddings = embeddings
        self.config = config
        self.model = svm.SVC(C=config.C, gamma=config.gamma)



# if n_features is not none, standardize all feature vectors to have the same length.
def email_records_to_word_ids(email_examples, token_mapping, n_features=None):
    labels = [None] * len(email_examples)
    body_ids = [None] * len(email_examples)
    for i, email in enumerate(email_examples):
        labels[i] = int(email['Label'])
        cur_ids = map(lambda word: token_mapping[word], nltk.word_tokenize(email['Body']) )
        # Pad or cut body to have right length:
        if n_features is not None:
            body_ids[i] = cur_ids[:n_features] + [0] * (max(0, n_features - len(cur_ids)))
        else:
            body_ids[i] = cur_ids
    return body_ids, labels

def get_email_batch_loader(token_mapping, n_features):

    def email_minibatches(email_examples, batch_size):
        body_ids, labels = email_records_to_word_ids(email_examples, token_mapping, n_features)
        return get_minibatches([body_ids, labels], batch_size)

    return email_minibatches

def train_and_eval_model(model, train_word_ids, train_labels, test_word_ids, dev_labels):
    model.fit(train_word_ids, train_labels)

    predictions = model.predict(test_word_ids)
    dev_labels = np.asarray(dev_labels)
    n_classes = np.unique(dev_labels).shape[0]
    score_labels=range(0, n_classes);
    print("= Accuracy on dev set: {}".format(np.mean(dev_labels == predictions)))
    # print(predictions)
    # precision = 1.0 * np.sum(np.logical_and(predictions == 1, dev_labels == 1)) / (np.sum(predictions == 1) + 1e-8)
    # recall = 1.0 * np.sum(np.logical_and(predictions == 1, dev_labels == 1)) / (np.sum(dev_labels == 1) + 1e-8)
    # f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # print "= Precision (True positives / Predicted Positives) {}".format(precision)
    # print "= Recall (True positives / Actual Positives) {}".format(recall)
    print("= Precision (averaged per class) {}".format(
        metrics.precision_score(dev_labels, predictions, labels=score_labels, average='weighted')
    ))
    print("= Recall (averaged per class) {}".format(
        metrics.recall_score(dev_labels, predictions, labels=score_labels, average='weighted')
    ))
    print("= F1 score (averaged per class): {}".format(
        metrics.f1_score(dev_labels, predictions, labels=score_labels, average='weighted') #
    ))

def main(debug=True):
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    config = Config()
    dm_conf = DMConfig()
    sdm = StaticDataManager(dm_conf)
    embeddings = load_embeddings(dm_conf, sdm.tok2id)

    # print("-- Loading GloVe embeddings into memory...", end="")
    # token_mapping, embeddings = loadWordVectorsIntoMemory(N = config.n_embeddings)
    # print("done")

    # print("-- Loading email dataset into memory...", end="")
    # train_records = read_records(config.train_filepath)
    # dev_records = read_records(config.dev_filepath)
    # print("done")

    # print("-- Parsing email dataset into embedding indexes...", end="")
    # train_word_ids, train_labels = email_records_to_word_ids(train_records, token_mapping) 
    # dev_word_ids, dev_labels = email_records_to_word_ids(dev_records, token_mapping) 
    # train_freqs = stats.itemfreq(train_labels)
    # print("done")
    # for c, freq in train_freqs:
    #     print("class {} in training: {} points".format(c, freq))

    # return 
    print(80 * "=")
    print("TRAINING")
    print(80 * "=")

    # print("-- Initializing and training SVM on average word vectors...")
    # svm = AvgSvmModel(embeddings, config)
    # train_and_eval_model(svm, train_word_feats, train_labels, dev_word_ids, dev_labels)
    train_word_feats, _, train_labels = sdm.get_train_feats_and_labels()
    dev_word_feats, _, dev_labels = sdm.get_dev_feats_and_labels()

    train_labels = np.argmax(np.asarray(train_labels), axis=1)
    dev_labels = np.argmax(np.asarray(dev_labels), axis=1)

    print("-- Initializing and training RandomForest on average word vectors...")
    rf = AvgRandomForestModel(embeddings, config)
    train_and_eval_model(rf, train_word_feats, train_labels, dev_word_feats, dev_labels)

    print("-- Initializing and training LogisticRegression on average word vectors...")
    # for w in [1, 2, 3, 4, 8, 16, 32]:
    #     print("--- w = {}".format(w))
    logreg = AvgLogisticRegressionModel(embeddings, config, {0:1, 1:2})
    train_and_eval_model(logreg, train_word_feats, train_labels, dev_word_feats, dev_labels)

if __name__ == '__main__':
    main()

