import json
import collections
import nltk
import re
from util import *

regex = re.compile(r"[^a-zA-Z-\s]")

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

    #Regex to remove special characters, keeping '-' for phrases like 'skip-gram'
    dataset = regex.sub(r"", dataset)
    vocabulary_all = nltk.word_tokenize(regex.sub(r"", dataset))
    index = 0

    #Remove one letter tokens
    vocab_without_minus = [x.lower() for x in vocabulary_all if len(x) > 1]

    vocabulary_temp = [["UNK", 0]]
    vocabulary_temp.extend(collections.Counter(vocab_without_minus).most_common(vocabulary_size-1))

    vocabulary_temp_words = [x for (x,y) in vocabulary_temp]

    vocabulary = dict()
    for word in vocabulary_temp_words:
        vocabulary[word] = len(vocabulary)

    return vocabulary, dict(zip(vocabulary.values(), vocabulary.keys()))

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
            for index = 1 returns pairs 0-127
            for index = 2 returns pairs 128-255, and so on.
        Last batch is trimmed if there are less than 'batch_size' pairs.
    """

    start = batch_size*index
    end = batch_size*(index+1)

    if end > len(inputs):
        end = len(inputs)

    if start >= end:
        return None, None

    return inputs[start:end], labels[start:end]
