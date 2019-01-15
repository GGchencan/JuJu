#!/usr/bin/env python
# !-*-coding:utf-8 -*-
"""
@version: python3.7
@author: ‘v-enshi‘
@license: Apache Licence
@contact: 123@qq.com
@site:
@software: PyCharm
@file: 1233.py
@time: 2019/1/4 10:45
"""

from __future__ import division, print_function, unicode_literals

from collections import defaultdict
import logging


def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g.
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)


def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g.
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True
    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)
    chunkEnd = False
    chunkEnd = (((prefix1 == "B") & (prefix2 == "B"))
                | ((prefix1 == "I") & (prefix2 == "B"))
                | ((prefix1 == "B") & (prefix2 == "O"))
                | ((prefix1 == "I") & (prefix2 == "O"))
                | ((prefix2 == "O") & (chunk_type1 != chunk_type2)))
    return chunkEnd


def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)
    chunkStart = False
    if prefix2 == 'B':
        chunkStart = True
    return chunkStart


def count_chunks(true_seqs, pred_seqs):
    """
    true_seqs: a list of true tags
    pred_seqs: a list of predicted tags
    return:
    correct_chunks: a dict (counter),
                    key = chunk types,
                    value = number of correctly identified chunks per type
    true_chunks:    a dict, number of true chunks per type
    pred_chunks:    a dict, number of identified chunks per type
    correct_counts, true_counts, pred_counts: similar to above, but for tags
    """
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag

    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks)


def get_result(correct_chunks, true_chunks, pred_chunks):
    """
    if verbose, print overall performance, as well as preformance per chunk type;
    otherwise, simply return overall prec, rec, f1 scores
    """
    # sum counts
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())

    tp = sum_correct_chunks
    p = sum_pred_chunks
    t = sum_true_chunks

    prec = tp / p if p else 0
    rec = tp / t if t else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    res = (prec, rec, f1)
    # print overall performance, and performance per chunk type
    logging.info("processed {} phrases; found: {} phrases; correct: {}."
                 .format(sum_true_chunks, sum_pred_chunks, sum_correct_chunks))
    logging.info("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f%%" % (prec * 100, rec * 100, f1 * 100))

    return res


def evaluate(trueseqs, predseqs):
    # tagString = {0: "O", 1: "I-ORG", 2: "I-MISC", 3: "I-PER", 4: "I-LOC", 5: "B-LOC", 6: "B-MISC", 7: "B-ORG"}
    tagString = {0: "O", 1: "B-ORG", 2: "B-MISC", 3: "B-PER", 4: "I-PER", 5: "B-LOC", 6: "I-ORG", 7: "I-MISC",
                 8: "I-LOC"}
    true_seqs = []
    pred_seqs = []
    for i in trueseqs:
        true_seqs.append(tagString[i])
    for j in predseqs:
        pred_seqs.append(tagString[j])

    (correct_chunks, true_chunks, pred_chunks) = count_chunks(true_seqs, pred_seqs)

    result = get_result(correct_chunks, true_chunks, pred_chunks)
    return result

####usage

# true_seqs = [1, 7, 2, 3, 8, 10, 10, 10, 10, 10, 2, 6, 9, 2, 3, 1, 7, 10, 10, 10]
# pred_seqs = [1, 1, 10, 3, 3, 1, 10, 3, 3, 2, 2, 6, 6, 2, 3, 10, 10, 10, 10, 10]
# result = evaluate(true_seqs, pred_seqs)
