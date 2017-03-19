import json
import collections
import nltk
import re
import math
import random
from util import *

regex1 = re.compile(r"[^a-zA-Z-\s]")
regex2 = re.compile(r"^[\W_]+$")

def read_datasets(training_set_path, test_set_path, test_set_limit):
    training_set = json.load(open(training_set_path))["documents"]
    tlog("Training set read.")

    if test_set_limit !=-1:
        test_set = json.load(open(test_set_path))["documents"][0:test_set_limit]
    else:
        test_set = json.load(open(test_set_path))["documents"]

    tlog("Test set read.")

    return training_set, test_set

def build_vocabulary(dataset, vocabulary_size):
    """
        Builds a vocabulary of length 'vocabulary_size'.
        The vocabulary consists of the most common words.
        UNK describes an unknown token. Its used later for building Skip-Gram
        pairs of words that do not exist in the vocabulary.
    """

    threshold = 0.0001

    #Regex to remove special characters, keeping '-' for phrases like 'skip-gram'
    vocabulary_all = nltk.word_tokenize(regex1.sub(r"", dataset))
    index = 0

    #Remove one letter tokens
    vocab_without_minus = [x.lower() for x in vocabulary_all if not regex2.match(x)]
    total_voc_size = len(vocab_without_minus)

    vocabulary_frequencies = [["UNK", 0]]
    vocabulary_frequencies.extend(collections.Counter(vocab_without_minus).most_common(vocabulary_size-1))

    deleted_words = []

    for i in range(1, vocabulary_size):
        freq = vocabulary_frequencies[i][1] / total_voc_size
        p =  ((freq - threshold) / freq) - math.sqrt(threshold/freq)
        if p > 0 and random.random() <= p:
            deleted_words.append(vocabulary_frequencies[i][0])

    vocabulary_frequencies_words = [x for (x,y) in vocabulary_frequencies if x not in deleted_words]

    vocabulary = dict()
    for word in vocabulary_frequencies_words:
        vocabulary[word] = len(vocabulary)

    return vocabulary, dict(zip(vocabulary.values(), vocabulary.keys())), vocabulary_frequencies, total_voc_size

def build_skip_gram_dataset(texts, vocabulary, window_size):
    """
        Builds pairs of words and returns a list containing them.
        Words that do not exist in the vocabulary are replaced with 'UNK'.
        Pairs that consist only of UNK are removed.
    """

    indices = []
    inputs = []
    labels = []

    for i in range(window_size):
        indices.append(i-window_size)
    for i in range(window_size):
        indices.append(i+1)

    words = texts.split(" ")
    for i in range(len(words)):
        if words[i] not in vocabulary:
            words[i] = "UNK"

    for i in range(len(words)):
        word = words[i]
        for index in indices:
            if i + index >= 0 and i + index < len(words):
                if word not in "UNK" and words[i+index] not in "UNK":
                    inputs.append(vocabulary[word])
                    labels.append(vocabulary[words[i+index]])

    return inputs, labels

def get_batch(inputs, labels, batch_size, index):
    """
        Returns batch no. index.
        I.E. for batch_size = 128:
            for index = 0 returns pairs 0-127
            for index = 1 returns pairs 128-255, and so on.
        Last batch is trimmed if there are less than 'batch_size' pairs.
    """

    start = batch_size*index
    end = batch_size*(index+1)

    if end > len(inputs):
        end = len(inputs)

    if start >= end:
        return None, None

    return inputs[start:end], labels[start:end]
