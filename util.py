from collections import Counter
import numpy as np
import cPickle
import sys

SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

def save(path, obj):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_data(filename='./data/smallval.txt', vocabulary_size=2000, min_sent_characters=0):
    SENTENCE_START_TOKEN = "SENTENCE_START"
    SENTENCE_END_TOKEN = "SENTENCE_END"
    UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

    word_to_index = []
    index_to_word = []

    word_counter = Counter()
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    sentences = []
    for line in open(filename, 'r'):
        if line == '\n': continue
        sent = line.strip().split()
        sentences.append( [SENTENCE_START_TOKEN] + sent + [SENTENCE_END_TOKEN])
        word_counter.update(sent)
    
    print 'the numbers of sentences is : %d' % len(sentences)

    vocab_count = word_counter.most_common(vocabulary_size)
    vocab = {SENTENCE_START_TOKEN:1, SENTENCE_END_TOKEN:2, UNKNOWN_TOKEN:0}
    
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 3

    print 'the size of vocabulary: %d' % len(vocab)
    word_to_index = vocab
    index_to_word = {value:key for key, value in word_to_index.iteritems()}


    X = [[word_to_index.get(w,0) for w in sent[:-1]] for sent in sentences]
    Y = [[word_to_index.get(w,0) for w in sent[1:]] for sent in sentences]
    
    i = 0

    X_Y_vocab_vocabindex = []

    for sen in X:
        x_train = np.asarray(sen).astype('int32')
        y_train = np.asarray(Y[i]).astype('int32')
        i = i + 1
        X_Y_vocab_vocabindex.append([x_train, y_train])

    #save dict and trainset
    save('./data/train.pkl',X_Y_vocab_vocabindex)
    save('./data/dict.pkl', word_to_index)

    return X_Y_vocab_vocabindex, word_to_index

def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index_to_word, word_to_index):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]

    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        next_word_probs = model.predict(new_sentence)[-1]
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)

    print_sentence(new_sentence, index_to_word)
#temp = load_data()
#with open('./data/small.pkl','wb') as f:
#    cPickle.dump(temp, f, protocol=cPickle.HIGHEST_PROTOCOL)


