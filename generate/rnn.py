import sys
import time
import os
import numpy
import theano
import theano.typed_list
from theano import tensor as T
from collections import OrderedDict

class RNNED(object):
    
    def __init__(self, nh, nc, de, model=None):

        self.y0 = theano.shared(name='y0',value = numpy.zeros(nh,dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',value = numpy.zeros(nh,dtype=theano.config.floatX))

        if model is not None:
            print "*** Loading pre-trained model"
            [self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.O_h, self.O_x, self.G, self.EMB] = [theano.shared(p) for p in model]
        else:
            self.W_e = theano.shared(name='W_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de)).astype(theano.config.floatX))

            self.U_e = theano.shared(name='U_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

            self.W_z_e = theano.shared(name='W_z_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de)).astype(theano.config.floatX))

            self.U_z_e = theano.shared(name='U_z_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

            self.W_r_e = theano.shared(name='W_r_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, de)).astype(theano.config.floatX))

            self.U_r_e = theano.shared(name='U_r_e',value = 0.2 * numpy.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

            self.O_h = theano.shared(name='O_h',value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, nh)).astype(theano.config.floatX))

            self.O_x = theano.shared(name='O_x',value=0.2 * numpy.random.uniform(-1.0, 1.0, (de, de)).astype(theano.config.floatX))

            self.G = theano.shared(name='G',value=0.2 * numpy.random.uniform(-1.0, 1.0, (nc, de)).astype(theano.config.floatX))
            #emb 
            self.EMB = theano.shared(name='EMB',value=0.2 * numpy.random.uniform(-1.0, 1.0, (nc, de)).astype(theano.config.floatX))

        self.params = [self.W_e, self.U_e, self.W_z_e, self.U_z_e, self.W_r_e, self.U_r_e, self.O_h, self.O_x, self.G, self.EMB]
        self.names = ['W_e', 'U_e', 'W_z_e', 'U_z_e', 'W_r_e', 'U_r_e', 'O_h', 'O_x', 'G','EMB']
        
        self.batch_gradients = [theano.shared(value = numpy.zeros((p.get_value().shape)).astype(theano.config.floatX)) for p in self.params]
        self.batch_nll = theano.shared(value = numpy.zeros((1)).astype(theano.config.floatX))

        # Compile training function
        self.prepare_train()

    def prepare_train(self):
        ## Prepare to recieve input and output labels
        X = T.ivector('X')
        Y_IDX = T.ivector('Y_IDX')
        h_d_0 = self.h0
        def decoder(x, y_idx, train_flag=True):
            def decoder_recurrence(x_t, h_tm1):
                
                x_e = self.EMB[x_t,:]

                # Reset gate
                r = T.nnet.sigmoid(T.dot(self.W_r_e, x_e) + T.dot(self.U_r_e, h_tm1))
                # Update gate
                z = T.nnet.sigmoid(T.dot(self.W_z_e, x_e) + T.dot(self.U_z_e, h_tm1))
                # Gated output
                h_prime = T.tanh(T.dot(self.W_e, x_e) + T.dot(self.U_e, r * h_tm1))
                # Compute hidden state
                h_t = z * h_tm1 + (1 - z) * h_prime
                # Compute the final layer
                s = T.dot(self.O_h, h_t) + T.dot(self.O_x, x_e)

                # Softmax to get probabilities over target vocab
                p_t = T.nnet.softmax(T.dot(self.G, s))[0]

                return [h_t, p_t]

            [h, p], _ = theano.scan(fn=decoder_recurrence,
                sequences = x,
                outputs_info = [h_d_0, None],
                n_steps = x.shape[0])

            # predict function
            self.predict = theano.function([x], p)

            # Compute the average NLL for this phrase
            phrase_nll = T.mean(T.nnet.categorical_crossentropy(p, y_idx))
            if train_flag:
                return phrase_nll, T.grad(phrase_nll, self.params)
            else:
                return phrase_nll

        # Learning rate
        lr = T.scalar('lr')
        # example index
        i = T.iscalar('i')

        # Get the average phrase NLL and the gradients
        phrase_train_nll, phrase_gradients = decoder(X, Y_IDX)
        phrase_test_nll = decoder(X, Y_IDX,False)

        train_updates = OrderedDict((p, p + g) for p,g in zip(self.batch_gradients, phrase_gradients))
        test_updates = [(self.batch_nll, T.set_subtensor(self.batch_nll[i], phrase_test_nll))]
        # Compile theano functions for training and testing
        self.phrase_train = theano.function(inputs=[X,Y_IDX], on_unused_input='warn', updates=train_updates)
        # The test function return phrase average NLL
        self.phrase_test = theano.function(inputs=[i,X,Y_IDX], on_unused_input='warn', updates=test_updates)

    def train(self, batch, lr):
        for (x,y_idx) in batch:
            self.phrase_train(x,y_idx)
        # Average gradients
        grad_acc = [g.get_value() for g in self.batch_gradients]
        grad_acc = [g / len(batch) for g in grad_acc]

        # Reset gradients
        for (p,g) in zip(self.params, self.batch_gradients):
            g.set_value(numpy.zeros((p.get_value().shape)).astype(theano.config.floatX))

        # Update shared variables
        for p,g in zip(self.params, grad_acc):
            p.set_value(p.get_value() - lr*g)

    def test(self, batch):
        batch_size = len(batch)
        # Update the size of the shared parameter
        self.batch_nll.set_value(numpy.zeros((batch_size)).astype(theano.config.floatX))
        # Get the average phrase NLL wrt each example in the test/validation set
        for i, (x,y_idx) in enumerate(batch):
            self.phrase_test(i,x,y_idx,)
        return self.batch_nll.get_value()

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())


    def getParams(self):
        return [p.get_value() for p in self.params]
























































