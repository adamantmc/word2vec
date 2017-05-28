import ijson
import logging
import datetime
import os.path
import numpy as np
from evaluator import Evaluator
from metrics import Metrics
from filewriter import FileWriter
from math import sqrt

word_file_path = "./types.txt"
vector_file_path = "./vectors.txt"

test_set_path = "./testSet"
training_set_path = "./trainingSet"

test_set_limit = 200
threshold_start = 1
threshold_end = 10

thresholds = []
metrics_obj_list = []

fw = FileWriter()

for i in range(threshold_start, threshold_end+1):
    thresholds.append(i)
    metrics_obj_list.append(Metrics())

def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)

def cossim(v1, v2):
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    product = np.dot(v1,v2)

    cosine_sim = product / (v1_mag * v2_mag)
    return cosine_sim

def doc2vec(document, word_vectors):
    doc_vec = np.zeros(200)
    counter = 0
    empty = 0

    for word in document["abstractText"].split():
        if word in word_vectors:
            doc_vec += word_vectors[word]
            counter += 1

    if counter != 0:
        doc_vec /= counter
        print(doc_vec)

    return doc_vec

def read_vectors(word_file_path, vector_file_path):
    word_embeddings = {}
    i = 0
    with open(word_file_path) as words, open(vector_file_path) as vectors:
        for x, y in zip(words, vectors):
            vector = []
            for value in y.split():
                vector.append(float(value))
            word_embeddings[x] = vector
            i+=1
            if i % 100000 == 0:
                break

    return word_embeddings

start_time = getTime()

tlog("Reading vectors from disk.")
word_vectors = read_vectors(word_file_path, vector_file_path)
tlog("Vectors read.")

#Reading both sets
training_set = open(training_set_path, "r", encoding="ISO-8859-1")
test_set = open(test_set_path, "r")

#Transforming documents to vectors
tlog("Processing training set.")
training_set_docs = ijson.items(training_set, "documents.item")
training_set_vectors = []
for doc in training_set_docs:
    training_set_vectors.append(doc2vec(doc, word_vectors))
tlog("Training set processed.")

tlog("Processing test set.")
test_set_docs = ijson.items(test_set, "documents.item")
test_set_vectors = []

i=0
for doc in test_set_docs:
    if i == test_set_limit:
        break

    test_set_vectors.append(doc2vec(doc, word_vectors))

    i+=1

tlog("Test set processed.")
