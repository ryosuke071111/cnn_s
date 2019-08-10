import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk import bleu_score
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from vocab_from_file import Vocab,generateVocab
import os
from data import data_generate
from data_loader import DataLoader
from tqdm import tqdm
import time
from model import VanilaEncoderDecoder
from attn_model import AttnEncoderDecoder
# from pg_attn_model import AttnEncoderDecoder
import sys
import  argparse
import math
from torch.nn.utils import clip_grad_norm_

from utils import *

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))
start = time.time()


#parser設定
parser = argparse.ArgumentParser()
parser.add_argument('--model', default="attn", help="type 'attn' or 'vanila'")
parser.add_argument('--datasize', help="datasize used for training",type=int)
parser.add_argument('--epoch', help="params used for training" ,type=int)
parser.add_argument('--bidirectional',help="bi or uni when training",type=str)
parser.add_argument('--nameop',default="", help="name option after param name",type=str)
parser.add_argument('--batch', help="name option after param name",type=int)
parser.add_argument('--inputlen', help="name option after param name",type=int)

args = parser.parse_args()

# ハイパーパラメータの設定
bidirectional = bidirectional if args.bidirectional == None else True if args.bidirectional == "True" else False
num_of_data = num_of_data if args.datasize == None else args.datasize
batch_size = batch_size if args.batch == None else args.batch
num_epochs = num_epochs if args.epoch == None else args.epoch
MAX_INPUT_LENGTH = MAX_INPUT_LENGTH if args.inputlen == None else args.inputlen

print('---------------------Hyper parameter---------------------')
print("device", device)
print('bidirectional', bidirectional)
print('datasize', num_of_data)
print('num_of_epochs', num_epochs)
print('batch size', batch_size)
print('max_input_length',MAX_INPUT_LENGTH)
print('備考: concatの後にreluしたやつ　tanhも入ってる')
print('---------------------------------------------------------')

torch.manual_seed(1)
random_state = 42

PATH="/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/params"


mce = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD)
def masked_cross_entropy(logits, target):
    return mce(logits.view(-1, logits.size(-1)), target.view(-1))


def compute_loss(batch_X, batch_Y, lengths_X, model, optimizer=None, is_train=True):
    # 損失を計算する関数
    model.train(is_train) 

    # 一定確率でTeacher Forcingを行う
    use_teacher_forcing = is_train and (random.random() < teacher_forcing_rate)
    max_length = batch_Y.size(0)

    # 推論
    pred_Y = model(batch_X, lengths_X, max_length, batch_Y, use_teacher_forcing)

    # 損失関数を計算
    loss = masked_cross_entropy(pred_Y.contiguous(), batch_Y.contiguous())
    if is_train:  # 訓練時はパラメータを更新
        optimizer.zero_grad()
        loss.backward()
        norm = clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

    batch_Y = batch_Y.transpose(0, 1).contiguous().data.cpu().tolist()
    pred = pred_Y.max(dim=-1)[1].data.cpu().numpy().T.tolist()


    return loss.item(), batch_Y, pred

def Train():
    iteration = 0
    for epoch in range(1, num_epochs+1):
        
        train_loss = 0.
        train_refs = []
        train_hyps = []
        # train
        for batch in tqdm(data_loader):
            batch_X, batch_Y, lengths_X = batch
            loss, gold, pred = compute_loss(batch_X, batch_Y, lengths_X, model, optimizer, is_train=True)
            loss = loss/MAX_OUTPUT_LENGTH/batch_size
            train_loss+=loss*batch_size
            # if iteration % 100==0:
            #     print("iteration:",iteration,' train_loss:',loss)

            # print("pred", [vocab._id2word(word) for word in pred[0]])
            # train_loss = loss
            # train_refs += gold
            # train_hyps += pred
            print('iteration:', iteration, 'loss:',loss)
            iteration+=1

        # 損失をサンプル数で割って正規化
        # train_loss = train_loss / len(data_loader.data)

        if epoch % 1 == 0:
            print("ref", [vocab._id2word(word) for word in batch_Y.transpose(0,1).tolist()[0]])
            print("pred", [vocab._id2word(word) for word in pred[0]])
            print()
           
        if epoch % 50 == 0:
            state = {
                'iter': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'current_loss': train_loss
            }
            model_save_path = os.path.join(PATH, param+"_epoch_"+str(epoch)+("_bi" if bidirectional else "")+"_"+args.nameop)
            torch.save(state, model_save_path)


#モデル構
model_name = args.model
param = "model_"+model_name+"_"+"datasize_"+str(num_of_data)
# param = "model_attn_datasize_10000_epoch_200_bi_relutanh"

if model_name == "vanila":
    model = VanilaEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    print('model is Vanila seq2seq')
elif model_name == "attn":
    model = AttnEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    # torch.cuda.empty_cache()
    # param_path= "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/params/"
    # print('param',param_path+param)
    # params = torch.load(param_path+param, map_location=lambda storage, loc: storage)
    # model.encoder.load_state_dict(params["encoder_state_dict"])
    # model.decoder.load_state_dict(params["decoder_state_dict"])
    # model.to(device)
    print('model',model)
    print('model is Attention seq2seq')
print('Model successfully constructed !!')
print()

# model = torch.nn.DataParallel(model,device_ids=[0, 1])


vocab = generateVocab()
articles, abstracts = data_generate(vocab, num_of_data)
data_loader = DataLoader(articles, abstracts, batch_size, shuffle=True)
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adagrad(model.parameters(), lr=0.15,initial_accumulator_value=0.1)
Train()




















