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
        
        print('duc', self.id2word)
        
        
        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            print('Begin constructing Vocablary')
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    # print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [UNK_TOKEN, PAD_TOKEN, START_DECODING,STOP_DECODING]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self.word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                self.word2id[w] = self.count
                self.id2word[self.count] = w
                self.count += 1

                if max_size != 0 and self.count >= max_size:
                    # print( "max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self.count))
                    print("vocab size is "+str(self.count) )
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
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def article2ids(self, article_words):
        ids = []
        oovs = []
        unk_id = self._word2id(UNK_TOKEN)
        for w in article_words:
            i = self._word2id(w)
            if i == unk_id: # If w is OOV
                if w not in oovs: # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(self.count + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs

    def abstract2ids(self, abstract_words, article_oovs):
        ids = []
        unk_id = self._word2id(UNK_TOKEN)
        for w in abstract_words:
            i = self._word2id(w)
            if i == unk_id: # If w is an OOV word
                if w in article_oovs: # If w is an in-article OOV
                    vocab_idx = self.count + article_oovs.index(w) # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else: # If w is an out-of-article OOV
                    ids.append(unk_id) # Map to the UNK token id
            else:
                ids.append(i)
        return ids


    def outputids2words(self, id_list,article_oovs):
        words = []
        for i in id_list:
            try:
                w = self._id2word(i) # might be [UNK]
            except ValueError as e: # w is OOV
                assert article_oovs is not None, "Error: model produced a word ID that isn't in the selfulary. This should not happen in baseline (no pointer-generator) mode"
                article_oov_idx = i - self.count
                try:
                    w = article_oovs[article_oov_idx]
                except ValueError as e: # i doesn't correspond to an article oov
                    raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
            words.append(w)
        return words

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