from struct import *
from tensorflow.core.example import example_pb2
import struct
import torch
import numpy as np

from utils import *


class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self.word2id = {}
        self.id2word = {}
        self.count = 0 # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [PAD_TOKEN, UNK_TOKEN, START_DECODING, STOP_DECODING]:
            self.word2id[w] = self.count
            self.id2word[self.count] = w
            self.count += 1
        
        
        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            print('Begin constructing Vocablary')
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    # print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, START_DECODING,STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1

                if max_size != 0 and self.count >= max_size:
                    # print( "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
                    print("vocab size is "+str(max_size) )
                    break
        print('vocab successfully　constructed!')
        print()
        # print( "Finished constructing vocabulary of %i total words. Last word added: %s" % (self.count, self.id2word[self.count-1]))

    def _word2id(self, word):
        if word not in self.word2id:
            return self.word2id[UNK_TOKEN]
        return self.word2id[word]
    
    def _id2word(self, word_id):
        if word_id not in self.id2word:
            return UNK_TOKEN
            # raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def abstract2sents(self, abstract):
        cur = 0
        sents = []
        while True:
          try:
            start_p = abstract.index(BOS_TOKEN, cur)
            end_p = abstract.index(EOS_TOKEN, start_p + 1)
            cur = end_p + len(EOS_TOKEN)
            sents.append(abstract[start_p+len(BOS_TOKEN):end_p])
          except ValueError as e: # no more sentences
            return sents

def generateVocab():
    vocab_file="/home/ryosuke/desktop/data_set/cnn_stories_tokenized/vocab"
    vocab = Vocab(vocab_file, 50000)
    return vocab