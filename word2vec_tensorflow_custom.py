import numpy as np
import tensorflow as tf
import math
from util import *
from data_processing import *
from queries import *
from args import *

args = getParser().parse_args()

embedding_size = int(args.embedding_size)
batch_size = 128
num_sampled = int(args.num_samples)
skip_window = int(args.window_size)
training_set, test_set = read_datasets("trainingSet","testSet",-1,-1)

texts = ""
for doc in training_set:
    texts += doc["abstractText"] + " "

vocabulary_size = 100000

tlog("Building vocabulary.")
vocabulary, reverse_vocabulary, _, _ = build_vocabulary(texts, vocabulary_size)
tlog("Vocabulary built.")

tlog("Bulding Skip-Gram pairs.")
inputs, labels = build_skip_gram_dataset(texts, vocabulary, (skip_window*2)+1)
input_size = len(inputs)
tlog("Skip-Gram pairs built.")

#Pairs as (input, label)
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[None, 1])

#Random initialization of embedding matrix.
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

#To avoid building one-hot representations and then multiplying with embedding
#matrix, we use tf.nn.embedding_lookup.
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#Output layer weights and biases.
output_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
output_biases = tf.Variable(tf.zeros([vocabulary_size]))

#Loss function using Noise Contrastive Estimation
loss = tf.reduce_mean(
    tf.nn.nce_loss(
        weights=output_weights,
        biases=output_biases,
        labels=train_labels,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=vocabulary_size))

#Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tlog("Total number of pairs: " + str(len(inputs)))
tlog("Training:")

i = 0 #Index of current batch
training_steps = int(len(inputs) / batch_size) + 1 #Only used for printing progr-ess bar
t_inputs, t_labels = get_batch(inputs, labels, batch_size, i) #Initializing inputs

#get_batch returns None for both lists if there are no data left, so we loop till we reach that.
while t_inputs != None and t_labels != None:
    i += 1
    feed_dict = {train_inputs:t_inputs, train_labels:np.reshape(t_labels, [len(t_labels), 1])}
    sess.run(optimizer, feed_dict = feed_dict)
    printProgressBar(i, training_steps)

    t_inputs, t_labels = get_batch(inputs, labels, batch_size, i)

tlog("Done training.")

w = sess.run(embeddings)

word2vec_queries(w, training_set, test_set[0:10], vocabulary, embedding_size, "word2vec_tensorflow_custom_" + str(embedding_size) + "_" + str(skip_window) + "_" + str(num_sampled))

tlog("Visualizing embeddings.")
#Visualization of embeddings using TSNE
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(w[:plot_only, :])
    labels = [reverse_vocabulary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
