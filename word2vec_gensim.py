import os
import re
from util import *
from data_processing import *
from queries import *
from nltk import tokenize
from gensim import models
from args import *

args = getParser().parse_args()

embedding_size = int(args.embedding_size)
window_size = int(args.window_size)
num_samples = int(args.num_samples)

model_filename = "./word2vec_model"

threshold = 0.0001

nonword_regex = re.compile(r"\W") #Expression to match non-word characters (characters not in[a-zA-Z0-9_])
num_regex = re.compile(r"^\d+$") #Expression to match start of string (^), numbers (\d+) and end of string ($). Removes number-only tokens.

def clean_text(text):
    words = tokenize.word_tokenize(text)

    clean_words = []
    for word in words:
        word_clean = nonword_regex.sub(r"", word.lower())
        if len(word_clean) > 1 and not num_regex.match(word_clean):
            clean_words.append(word_clean)

    return clean_words

def generateModel(sentences, embedding_size, num_samples, window_size):
    tlog("Generating Word2Vec model.")
    model = models.Word2Vec(sentences, size=embedding_size, negative = num_samples, sample = threshold, window = window_size)
    tlog("Model generated.")
    return model

def make_vectors(word_vectors, doc_set, embedding_size):
    word_percent = 0
    vectors = []
    for doc in doc_set:
        vector = np.zeros(embedding_size)
        words = tokenize.word_tokenize(doc["abstractText"])
        existing_words = 0

        for word in words:
            if len(word) > 1 and word in word_vectors:
                existing_words += 1
                vector += word_vectors[word]

        vector /= len(words)
        word_percent += existing_words / len(words)
        vectors.append(vector)

    print(word_percent/len(doc_set))
    return vectors

training_set, test_set = read_datasets("trainingSet","testSet",-1,-1)

for doc in test_set:
    doc["abstractText"] = ' '.join(clean_text(doc["abstractText"]))

if os.path.isfile(model_filename):
    tlog("Loading saved word2vec model from disk.")
    word2vec = models.Word2Vec.load(fname=model_filename)
else:
    tlog("Tokenizing into sentences.")

    sentences = []
    for x in training_set:
        temp = tokenize.sent_tokenize(x["abstractText"])
        for sentence in temp:
            sentences.append(clean_text(sentence))

    word2vec = generateModel(sentences, embedding_size, num_samples, window_size)
    tlog("Saving model to disk.")
    word2vec.save(model_filename)

word_vectors = word2vec.wv
del word2vec

tlog("Making document vectors.")
train_vectors = make_vectors(word_vectors, training_set, embedding_size)
test_vectors = make_vectors(word_vectors, test_set, embedding_size)

tlog("Making queries.")
queries(training_set, test_set[0:10], train_vectors, test_vectors[0:10], "word2vec_gensim_" + str(embedding_size) + "_" + str(window_size) + "_" + str(num_samples))
tlog("Done.")
