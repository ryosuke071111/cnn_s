from utils import *

#Tensoring
def tensorsFromSent(sentence):
    sentence = sentence.decode()
    sentence = sentence.split(' ')
    indexes = [vocab._word2id(word) for word in sentence]
    indexes.append(EOS)
    return indexes

def judge_inclue_input(sentence):
    sentence = sentence.decode()
    sentence = sentence.split(' ')
    if len(sentence)>MAX_INPUT_LENGTH-1:
        return False
    else:
        return True

def judge_inclue_output(sentence):
    sentence = sentence.decode()
    sentence = sentence.split(' ')
    if len(sentence)>MAX_OUTPUT_LENGTH-1:
        return False
    else:
        return True

def pad_seq_article(seq,max_length=MAX_INPUT_LENGTH):
    res = seq+[PAD for i in range(max_length-len(seq))]
    return res

def pad_seq_abst(seq,max_length=MAX_OUTPUT_LENGTH):
    res = seq+[PAD for i in range(max_length-len(seq))]
    return res