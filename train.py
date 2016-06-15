import pickle
import numpy as np
import rnn as rnned
import time
import sys
from util import *

def minibatch(l, bs):
    for i in xrange(0, len(l), bs):
        yield l[i:i+bs]

def saveModel(rnn):
    rParameters = rnn.getParams()
    print 'save model...'
    with open("./best.mdl","wb") as m:
        pickle.dump([rParameters], m)

# Hyperparameters
s = {
  'data_path':'./data/val.txt',
  'lr': 0.0827, # The learning rate
  'bs':64, # size of the mini-batch
  'nhidden':512, # Size of the hidden layer
  'seed':324, # Seed for the random number generator
  'emb_dimension':200, # The dimension of the embedding
  'nepochs':300, # The number of epochs that training is to run for
  'vobsize':10000 # The frequency threshold for histogram pruning of the vocab
}
#load train data
train, word_to_index = load_data(filename=s['data_path'],vocabulary_size=s['vobsize']) 

#load dev data
dev = train

#update the size of dict
s['vobsize'] = len(word_to_index)


start = time.time()
rnn = rnned.RNNED(nh=s['nhidden'], nc=s['vobsize'], de=s['emb_dimension'],model=None)
print "--- Done compiling theano functions : ", time.time() - start, "s"

s['clr'] = s['lr']
best_dev_nll = np.inf

#Training
for e in xrange(s['nepochs']):
    s['ce'] = e
    tic = time.time()

    for i, batch in enumerate(minibatch(train, s['bs'])):
        rnn.train(batch, s['clr'])

    print '[learning] epoch', e, '>> completed in', time.time() - tic, '(sec) <<'
    sys.stdout.flush()

    #get the average nll for the validation set
    dev_nlls = rnn.test(dev)
    dev_nll = np.mean(dev_nlls)
    print '[dev-nll]', dev_nll, "(NEW BEST)" if dev_nll < best_dev_nll else ""
    sys.stdout.flush()

    if dev_nll < best_dev_nll:
        best_dev_nll = dev_nll
        s['be'] = e
        saveModel(rnn)

    if abs(s['be'] - s['ce']) >= 3: s['clr'] *= 0.2
    if s['clr'] < 1e-5: break

print '[BEST DEV-NLL]', best_dev_nll
print '[FINAL-LEARNING-RATE]', s['clr']














