#!/usr/bin/env python3
from __future__ import print_function

import os, sys
from glob import glob
import pdb
import re
import dateparser
import csv
import gzip
import nltk
# import tensorflow as tf
import collections
from pprint import pprint
try:
    import _pickle as cPickle
except ImportError:
    import cPickle
import numpy as np
from sklearn.model_selection import train_test_split

from data_util import DMConfig, EmailVectorizer
from tfdata_helpers import write_to_tfrecords, tf_filename_func
from util import read_records

from defs import N_CLASSES

# DATA_DIR = './processed-data/test_data'
# DATA_DIR = '../maildir/buy-r/sent_items'
# BASE_DIR = '../maildir/*'
# BASE_DIR = '../maildir/buy-r'
# BASE_DIR = map(os.path.abspath, [
#     '../maildir/buy-r', '../maildir/lay-k', '../maildir/shackleton-s', '../mail/dir/nemec-g',
#     '../maildir/dasovich-j', '../maildir/symes-k', 'farmer-d' 
# ])
# BASE_DIR = map(os.path.abspath, [
#     '../maildir/kean-s', '../maildir/kean-s', '../mail/dir/jones-t',
#     '../maildir/taylor-m', '../maildir/sgermany-c', '../maildirbeck-s' 
# ])
BASE_DIR = map(os.path.abspath, [
   '../maildir/kean-s', '../mail/dir/jones-t',
    '../maildir/taylor-m', '../maildir/sgermany-c', '../maildir/beck-s' 
])
# BASE_DIR = map(os.path.abspath, [
#     '../maildir/beck-s' , '../maildir/scott-s', '../maildir/rogers-b',
#      '../maildir/bass-e', '../maildir/sanders-r', '../maildir/campbell-l',      
#      '../maildir/shapiro-r ', '../maildir/guzman-m', '../maildir/lokay-m',      
# ])
# BASE_DIR = ['../maildir/buy-r', '../maildir/lay-k']
# DATA_DIR = '{}/*'.format(BASE_DIR)
DATA_DIR = ['{}/*'.format(d) for d in BASE_DIR]

SHOULD_USE_TFRECORDS = True
SHOULD_TOKENIZE = True
SHOULD_FEATURIZE = True
SHOULD_PAD = True
SHOULD_SHUFFLE_VAL = True
RECORDS_PER_LOOP_ESTIMATE = 10000

# Load tok2id
config = DMConfig()
with gzip.open(os.path.abspath(config.tok2id_path)) as f:
    tok2id = cPickle.load(f)

email_vectorizer = EmailVectorizer(config, tok2id)

def strformat_fn(path, start=BASE_DIR):
    return os.path.relpath(path, start) 

def get_file_contents(dr):
    if type(dr) == str:
        dr = [dr]
    for dir in dr:
        for fn in glob('{}/*/*'.format(os.path.abspath(dir))):
            if not os.path.isfile(fn):
                continue
            with open(fn, 'r') as fp:
                contents = fp.read()
            yield contents, strformat_fn(fn, dir)

def match_length(match):
    return match.end(0) - match.start(0)

def get_last_match(pattern, string, flags=0):
    matches_iter = re.finditer(pattern, string, flags)
    matches = [match for match in matches_iter]
    return matches[-1] if matches else None

# Fields needed:
# From, To, Subject, Timestamp, message text.


EMAIL_RE = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
FROM_RE = r'From:\s*([^\n]+)\n'
TO_RE = r'To:\s*([^\n]+)\n'
SUBJECT_RE = r'Subject:[\t ]*(re:|fwd:|fw:)?[\t ]*([^\n\r\f\v]+)[\n\r\f\v]'

REPLY_RE = r'-+[^-]*?Original Message[^-]*?-+'
FORWARD_RE = r'-+[^-]*?Forwarded by[^-]+?-+'


HEADER_RE = r'Message-ID:\s*([^\n]+)\n'
HDATE_RE = r'Date:\s*([^\n]+[0-9]{2}:[0-9]{2}(:[0-9]{2})?)[^\n]*\n'
HFROM_RE =r'(?<!X-){}'.format(FROM_RE)
HTO_RE = r'(?<!X-)To:\s*(([^,:]+?,[\r\t\f\v ]*\n?)*[^,:\n]+?[\r\t\f\v ]*\n)'
HSUBJECT_RE = SUBJECT_RE
HEND_RE = r'X-FileName:[^\n]+\n'

# HEADER_RE = r'Message-ID:([^\n]+)\n\s*Date:([^\n]+)\n\s*From:([^\n]+)\n\s*{}\s*{}'.format(
#     HEADER_TO, SUBJECT_RE)

def get_message_id(contents):
    id_match = re.search(HEADER_RE, contents)
    if id_match:
        return id_match.group(1).strip()
    else:
        return None

# def parse_body_text(message_data, body_text):
#     str_body = body_text.strip()

#     if SHOULD_TOKENIZE:
#         toks = nltk.word_tokenize(str_body)
#         if SHOULD_FEATURIZE:
#             word_feats, length = email_vectorizer.featurize_email(toks, SHOULD_PAD)
#             message_data['Body'] = word_feats
#             message_data['Length'] = length
#         else:
#             message_data['Body'] = toks
#             message_data['Length'] = len(toks) 
#     else:
#         message_data['Body'] = str_body
#         message_data['Length'] = len(str_body)

def parse_body_text(message_data, body_text):
    str_body = body_text.strip()
    toks = nltk.word_tokenize(str_body)
    word_feats, length = email_vectorizer.featurize_email(toks, SHOULD_PAD)
    message_data['Body'] = word_feats
    message_data['Length'] = length


def remove_leading_carats(text):
    next_char_to_copy = 0
    clean_text = ""
    carat_re = r'^>[\r\t\f >]*'
    matches_iter = re.finditer(carat_re, text, flags=re.MULTILINE)
    for match in matches_iter:
        clean_text += text[next_char_to_copy:match.start(0)] 
        next_char_to_copy = match.end(0)
    clean_text += text[next_char_to_copy:] 
    return clean_text 

def parse_metadata_header(content, content_start, message_data):
    return apply_metadata_parsers(content, message_data,
        [
            (parse_metadata_format_3, content_start),
            (parse_metadata_format_3, 0)
        ]
    )

def parse_metadata_forward(content, content_start, message_data):
    return apply_metadata_parsers(content, message_data,
        [
            (parse_metadata_format_2, content_start),
            (parse_metadata_format_1, content_start),
            (parse_metadata_format_3, 0)
        ]
    )

def parse_metadata_reply(content, content_start, message_data):
    return apply_metadata_parsers(content, message_data,
        [
            (parse_metadata_format_1, content_start),
            (parse_metadata_format_2, content_start),
            (parse_metadata_format_3, 0)
        ]
    )

def apply_metadata_parsers(content, message_data, parsers):
    for parser, content_start in parsers:
        body_start = parser(content[content_start:], message_data)
        if body_start >= 0:
            return body_start + content_start
    return -1

class ParseError(IndexError): pass # simple exception
def skip_emptychars(cur_char, content):
    while (cur_char < len(content) and content[cur_char].isspace()):
        cur_char += 1
    if cur_char >= len(content):
        raise ParseError
    return cur_char

def parse_metadata_format_1(content, message_data):
    cur_char = 0
    try:
        cur_char = skip_emptychars(cur_char, content)
        fromMatch = re.search(FROM_RE, content[cur_char:], flags=re.IGNORECASE)
        if fromMatch:
            message_data['From'] = fromMatch.group(1).strip()
            cur_char += fromMatch.end(0)
            cur_char = skip_emptychars(cur_char, content)

        date_re = r'(Date|Sent):\s*(.+)\n'
        dateMatch = re.search(date_re, content[cur_char:], flags=re.IGNORECASE) 
        if dateMatch:
            message_data['Date'] = dateMatch.group(2).strip()
            cur_char += dateMatch.end(0)
            cur_char = skip_emptychars(cur_char, content)

        toMatch = re.search(TO_RE, content[cur_char:], flags=re.IGNORECASE) 
        # match phrases delimitted by semicolons, and potentially spanning multiple lines
        if toMatch:
            message_data['To'] = toMatch.group(1).strip()
            cur_char += toMatch.end(0)
            cur_char = skip_emptychars(cur_char, content)
        # Skip over BCC and CC.
        # ccMatch = re.search(r'(cc|bcc):\s*(([^,]+?,\s*?\n)*.+\s*?\n)', content[cur_char:], flags=re.IGNORECASE)
        # if ccMatch:
        #     cur_char += ccMatch.end(0)
        #     cur_char = skip_emptychars(cur_char, content)
        # bccMatch = re.search(r'(cc|bcc):\s*(([^,]+?,\s*?\n)*.+\s*?\n)', content[cur_char:], flags=re.IGNORECASE)
        # if bccMatch:
        #     cur_char += bccMatch.end(0)
        #     cur_char = skip_emptychars(cur_char, content)

        subjectMatch = re.search(SUBJECT_RE, content[cur_char:], flags=re.IGNORECASE) 
        if subjectMatch:
            message_data['Subject'] = subjectMatch.group(2).strip()
            cur_char += subjectMatch.end(0)
            cur_char = skip_emptychars(cur_char, content)

    except ParseError as e:
        return -cur_char

    return cur_char # success, return the final character position


forward_date_regex = r'[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\s*[0-9]{1,2}:[0-9]{2}(:[0-9]{2})?(\s*(P|A)M)?'
def parse_metadata_format_2(content, message_data):
    # idea: from address can have a lot of formats... lets match it more generally
    # Match To
    # then, match Date starting from the start of To
    # then, set From to be content[:date.start]
    # Then, match CC, Subject off of To.end
    cur_end_char = 0
    toMatch = re.search(TO_RE, content[cur_end_char:], flags=re.IGNORECASE) 
    if toMatch:
        message_data['To'] = toMatch.group(1).strip() 
        cur_end_char += toMatch.end(0)

        # look for dateMatch up to the beginning of toMatch
        dateMatch = get_last_match(forward_date_regex, content[:toMatch.start(0)], flags=re.IGNORECASE) 
        if dateMatch:
            message_data['Date'] = dateMatch.group(0).strip()
            message_data['From'] = content[:dateMatch.start(0)].strip() # grab everything down to Date
    try:
        cur_end_char = skip_emptychars(cur_end_char, content)
    except ParseError as e:
        return -cur_end_char
    subjectMatch = re.search(SUBJECT_RE, content[cur_end_char:], flags=re.IGNORECASE)
    if subjectMatch:
        message_data['Subject'] = subjectMatch.group(2).strip()
        cur_end_char += subjectMatch.end(0)

    return cur_end_char


def parse_metadata_format_3(content, message_data):
    cur_end_char = 0
    dateMatch = re.search(HDATE_RE, content[cur_end_char:], flags=re.IGNORECASE) 
    if dateMatch:
        message_data['Date'] = dateMatch.group(1).strip()
        cur_end_char += dateMatch.end(0)
    fromMatch = re.search(HFROM_RE, content[cur_end_char:], flags=re.IGNORECASE)
    if fromMatch:
        message_data['From'] = fromMatch.group(1).strip()
        cur_end_char += fromMatch.end(0)
    toMatch = re.search(HTO_RE, content[cur_end_char:], flags=re.IGNORECASE)
    if toMatch:
        message_data['To'] = toMatch.group(1).strip()
        cur_end_char += toMatch.end(0)
    subjectMatch = re.search(HSUBJECT_RE, content[cur_end_char:], flags=re.IGNORECASE)
    if subjectMatch:
        message_data['Subject'] = subjectMatch.group(2).strip()
        cur_end_char += subjectMatch.end(0)

    endMatch = re.search(HEND_RE, content[cur_end_char:], flags=re.IGNORECASE)
    if endMatch:
        cur_end_char += endMatch.end(0)

    return cur_end_char


def parse_message(content, content_start, metadata_parser, label, id_, conservative):
    message_data = collections.defaultdict(str);
    message_data['Id'] = id_
    message_data['From'] = ''
    message_data['To'] = ''
    message_data['Date'] = ''
    message_data['Subject'] = ''
    message_data['Body'] = ''
    message_data['Length'] = 0
    message_data['Label'] = label

    body_start = metadata_parser(content, content_start, message_data)
    if body_start >= 0:
        parse_body_text(message_data, content[body_start:])
    else:
        if conservative:
            message_data = None
        else:
            parse_body_text(message_data, content)
    return message_data

CLASS_NO_ACTION = 0
CLASS_REPLY = 1
CLASS_FORWARD = 2

fieldnames = ['Id', 'Label', 'From', 'To', 'Date', 'Subject', 'Body', 'Length']
def write_records_to_csv(records, filename, compress=False):
    opener = gzip.open if compress else open
    with opener(filename, 'w+') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_records_to_pickle(records, filename, compress=True):
    opener = gzip.open if compress else open
    with opener(filename, 'w+') as f:
        cPickle.dump(records, f, -1)


def extract_records(file_names, conservative, fn_limit=50000):
    processed_count = 0
    for contents, file_name in file_names:
        message_id = get_message_id(contents)
        if not message_id:
            print("{}: could not extract message id, skipping record".format(file_name))
            continue

        contents = remove_leading_carats(contents)
        reply_match = get_last_match(REPLY_RE, contents, flags=re.IGNORECASE|re.DOTALL) 
        forward_match = get_last_match(FORWARD_RE, contents, flags=re.IGNORECASE|re.DOTALL) 
        header_match = re.search(HEADER_RE, contents, flags=re.IGNORECASE|re.DOTALL) 

        if reply_match and forward_match:
            # uh oh, figure out which one is last...
            if reply_match.start() > forward_match.start():
                label = CLASS_REPLY
                content_start = reply_match.end(0)
                parser = parse_metadata_reply
            else:
                label = CLASS_FORWARD if N_CLASSES > 2 else CLASS_REPLY
                content_start = forward_match.end(0)
                parser = parse_metadata_forward
        elif reply_match:
            label = CLASS_REPLY
            content_start = reply_match.end(0)
            parser = parse_metadata_reply
        elif forward_match:
            label = CLASS_FORWARD if N_CLASSES > 2 else CLASS_REPLY
            content_start = forward_match.end(0)
            parser = parse_metadata_forward
        elif header_match:
            label = CLASS_NO_ACTION
            content_start = header_match.end(0)
            parser = parse_metadata_header
        else:
            print("--{}: could not identify type, skipping record".format(file_name))
            continue

        message_data = parse_message(contents, content_start, parser, label, message_id, conservative)
        if message_data is None:
            print("--{}: parsing failed, skipping record".format(file_name))
            continue
        if len(message_data['Body']) == 0:
            print("--{}: message body has zero length, skipping record".format(file_name))
            continue

        # records.append(message_data)
        processed_count += 1
        if processed_count >= fn_limit:
            break
        yield message_data

def undersample(record_generator, total_records = 50000):
    num_classes = {
        CLASS_NO_ACTION: 0,
        CLASS_REPLY: 0,
        CLASS_FORWARD: 0
    }

    record_samples = {
        CLASS_NO_ACTION: [],
        CLASS_REPLY: [],
        CLASS_FORWARD: []
    }

    count_limits = {
        # CLASS_NO_ACTION: 4 * total_records // 10,
        CLASS_NO_ACTION: total_records,
        CLASS_REPLY: total_records,
        CLASS_FORWARD: total_records
    }

    for record in record_generator:
        c = record['Label']
        num_classes[c] += 1
        if num_classes[c] <= count_limits[c]:
            record_samples[c].append(record)
        else:
            j = np.random.randint(0, num_classes[c] + 1)
            if j < len(record_samples[c]):
                record_samples[c][j] = record 

    all_records = []
    for k, v in record_samples.items():
        all_records += v
    return all_records, record_samples, num_classes

def write_records(data_dir = './processed-data/test_data', start_it = 0, 
                  record_limit = 50000, loop_limit = 50000, split_ratio = 0.1, conservative = True):
    it = start_it 
    num_train_records = 0
    file_name_generator = get_file_contents(data_dir)
    # file_name_generator = get_file_contents('test_data/30')
    np.random.seed(42) # for train_test_split

    class_counts = {
        CLASS_NO_ACTION: 0,
        CLASS_REPLY: 0,
        CLASS_FORWARD: 0
    }

    while (True):
        print("it: {}".format(it))
        print("== Starting parse loop: records {} to {}".format(num_train_records, num_train_records + loop_limit))
        extractor = extract_records(file_name_generator, fn_limit=loop_limit, conservative=conservative)
        # records = undersample(extractor, record_limit)
        records, record_samples, num_classes = undersample(extractor, loop_limit)
        if len(records) == 0:
            print('-- Exhausted file list.')
            break
        # write_records_to_csv(records, os.getcwd() + "/email_records{}.csv".format(it), False)
        # write_records_to_tfexamples(records, os.getcwd() + "/email_records{}.tfrecord".format(it), False)
        train, val = train_test_split(records, test_size = 0.2, shuffle=SHOULD_SHUFFLE_VAL)
        dev, test = train_test_split(val, test_size = 0.5, shuffle=SHOULD_SHUFFLE_VAL)
        if not SHOULD_SHUFFLE_VAL:
            np.random.shuffle(train) # randomize training data, but not dev or test.
        print("== writing {} records to disk".format(len(records)))
        print("== class distribution: ")
        for k, v in record_samples.items():
            count = len(v)
            print("=== {}: {}".format(k, count))
            class_counts[k] = class_counts[k] + count

        if SHOULD_USE_TFRECORDS:
            write_func = write_to_tfrecords
            filename_func = tf_filename_func
        else:
            write_func = write_records_to_pickle
            filename_func = lambda dataset, it: os.path.abspath(
                "./processed-data/buy_records{}_{}.pkl.gz".format(dataset, it)
            )

        write_func(train, filename_func('train', it))
        write_func(dev, filename_func('dev', it))
        write_func(test, filename_func('test', it))

        it += 1
        num_train_records += len(train)
        if num_train_records >= record_limit:
            break

    print("== Parse loop finished, processed {} train records".format(num_train_records))
    print("== Overall class distribution: ")
    for k, count in class_counts.items():
        print("=== {}: {}".format(k, count))



if __name__ == '__main__':
    write_records(start_it = 0, data_dir = BASE_DIR, record_limit = 100000, loop_limit=RECORDS_PER_LOOP_ESTIMATE)
    # records = read_records('./processed-data/skilling_records0_train.pkl.gz')
    # pdb.set_trace()
    # num_ones = reduce(lambda cum, r: cum + r['Label'], records, 0)
    #(print "num_ones: {}, num_records: {}".format(num_ones, len(records)))
    # for r in records:
    #     if len(r['Body']) == 0:
    #         pdb.set_trace()
    #(        print r['Id'], 'has zero length!')
