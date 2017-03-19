import numpy as np
import tensorflow as tf
import math
from util import *
from data_processing import *
from operator import itemgetter

training_set, test_set = read_datasets("trainingSet","testSet",-1)

texts = ""
for doc in training_set:
    texts += doc["abstractText"] + " "

vocabulary_size = 100000

tlog("Building vocabulary.")
vocabulary, reverse_vocabulary, voc_count, total_size = build_vocabulary(texts, vocabulary_size)
tlog("Vocabulary built.")

tlog("Sorting list.")
voc_count.sort(key = itemgetter(1), reverse = True)
tlog("List sorted.")

print(total_size)

for i in range(100):
    print(voc_count[i][0], "{0:.2f}".format(100*voc_count[i][1]/total_size)+"%",voc_count[i][1])
