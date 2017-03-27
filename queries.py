import nltk
import numpy as np
import operator
import math

from evaluator import Evaluator
from metrics import Metrics
from filewriter import FileWriter


def dot_product(v1, v2):
    return sum(map(operator.mul, v1, v2))

def cosine_similarity(v1, v2):
    prod = dot_product(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)

def make_vector(embeddings, document, vocabulary, vec_size):
    words = nltk.word_tokenize(document)
    vector = np.zeros(vec_size)

    for word in words:
        if word in vocabulary:
            vector += embeddings[vocabulary[word]]

    return (vector/vec_size)

def make_vector_lists(training_set, test_set, embeddings, vocabulary, vec_size):
    train_vectors = []
    test_vectors = []

    for doc in training_set:
        train_vectors.append(make_vector(embeddings, doc["abstractText"], vocabulary, vec_size))

    for doc in test_set:
        test_vectors.append(make_vector(embeddings, doc["abstractText"], vocabulary, vec_size))

    return train_vectors, test_vectors

def query(training_set, query_doc, threshold):
    scores = []
    i = 0

    for doc in training_set:
        scores.append([i, cosine_similarity(query_doc, doc)])
        i += 1

    scores.sort(key = lambda tup: tup[1], reverse = True)
    return scores

def queries(training_set, test_set, train_vectors, test_vectors, path):
    threshold_start = 10
    threshold_end = 200
    thresholds = []
    metrics_obj_list = []

    for i in range(threshold_start, threshold_end+1):
        thresholds.append(i)
        metrics_obj_list.append(Metrics())

    fw = FileWriter(path)
    eval = Evaluator(training_set)

    for i in range(len(test_vectors)):
        scores = query(train_vectors, test_vectors[i], threshold_end)
        query_doc = test_set[i]

        for j in range(len(thresholds)):
            threshold = thresholds[j]

            eval.query([training_set[x] for (x,y) in scores[0:threshold]], query_doc)
            eval.calculate()

            metrics_obj_list[j].updateConfusionMatrix(eval.getTp(), eval.getTn(), eval.getFp(), eval.getFn())
            metrics_obj_list[j].updateMacroAverages(eval.getAccuracy(), eval.getF1Score(), eval.getPrecision(), eval.getRecall())

    for obj in metrics_obj_list:
        obj.calculate(len(test_set))

    fw.writeToFiles(metrics_obj_list, thresholds)

def word2vec_queries(embeddings, training_set, test_set, vocabulary, vec_size, path):
    train_vectors, test_vectors = make_vector_lists(training_set, test_set, embeddings, vocabulary, vec_size)
    queries(training_set, test_set, train_vectors, test_vectors, path)
