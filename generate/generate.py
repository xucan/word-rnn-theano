import numpy as np
import cPickle
import rnn as rnned
from util import * 
import sys

model_path = '../best.mdl'
dict_path = '../data/dict.pkl'

print 'load model...'
best = cPickle.load(open(model_path, "r"))
rParameters = best[0]
word_to_index = cPickle.load(open(dict_path, "r"))
index_to_word = {value:key for key, value in word_to_index.items()}

#parameters == train.py

nhidden = 512
vobsize = len(word_to_index)
emb_dimension = 200

rnn = rnned.RNNED(nh=nhidden, nc=vobsize, de=emb_dimension, model= rParameters)


for i in range(10):
    generate_sentence(rnn, index_to_word, word_to_index)



