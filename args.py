from argparse import ArgumentParser

def getParser():
    parser = ArgumentParser()

    parser.add_argument("-s", "--size", dest = "embedding_size", help = "Size of word vectors", default = 128)
    parser.add_argument("-w", "--window", dest = "window_size", help = "Size of window (total, with target word).", default = 5)
    parser.add_argument("-n", "--nsamples", dest = "num_samples", help = "Number of negative samples", default = 64)
    
    return parser
