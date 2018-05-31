#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from util import one_hot

LBLS = [
    "NORESPONSE",
    "REPLY",
    "FORWARD",
    ]
NONE = "NORESPONSE"
LMAP = {k: one_hot(len(LBLS),i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
CASE2ID = {cs: i for i, cs in enumerate(CASES)}
START_TOKEN = "<s>"
END_TOKEN = "</s>"
N_CASES = len(CASES)

N_CLASSES = 2
EMBED_SIZE = 50
MAX_LENGTH = 300 # maximum number of words to consider in an email.
BATCH_LIM = 1000 # maximum number of batches per epoch.
# BATCH_LIM = 1000 # maximum number of batches per epoch.

PROJECT_DIR = os.path.abspath('..')
BASE_DIR = [os.path.abspath(dr) for dr in [
   '../maildir/kean-s', '../mail/dir/jones-t',
    '../maildir/taylor-m', '../maildir/sgermany-c', '../maildir/beck-s' 
]]