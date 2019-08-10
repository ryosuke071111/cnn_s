import random
import numpy as np
from sklearn.utils import shuffle
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
from utils import *
from torch.autograd import Variable


# デバイスの設定

torch.manual_seed(1)
random_state = 42


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=PAD)
        self.gru = nn.GRU(emb_size, hidden_size, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size*2 if bidirectional  else hidden_size , hidden_size) # bidrectionalにした場合のやつ
        self.bi_to_uni = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, seqs, input_lengths, hidden=None):
        emb = self.embedding(seqs)
        packed = pack_padded_sequence(emb, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, outputs_lengths = pad_packed_sequence(outputs)
        outputs = outputs.contiguous()
        outputs = self.linear(outputs)
        if bidirectional:
            hidden = self.bi_to_uni(torch.cat((hidden[0,:,:],hidden[1,:,:]),-1).unsqueeze(0))
            hidden = F.relu(hidden)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size*2,hidden_size)
            self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        #オリジナルと完全に同じ出力がでる
        if self.method == "concat":
            hidden = hidden.unsqueeze(0)
            attn_energies = torch.cat((hidden.expand(max_len,batch_size,hidden_size),encoder_outputs),2).to(device)
            attn_energies = self.attn(attn_energies)
            attn_energies = torch.tanh(attn_energies)
            attn_energies = self.linear(attn_energies)
            attn_energies = attn_energies.squeeze(-1).transpose(0,1)

        #オリジナルと完全に同じ出力がでる
        if self.method == "dot":
            hidden = hidden.unsqueeze(0).transpose(0,1)
            encoder_outputs = encoder_outputs.transpose(0,1)
            attn_energies = torch.bmm(encoder_outputs, hidden.transpose(1,2)).squeeze(-1).to(device)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
            

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):

        super(Decoder, self).__init__()
        #Define Parameters
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.hidden_size = hidden_size
        self.output_size = output_size

        #Define Layers
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=PAD) #padding_idxいらない。。？
        self.dropout = nn.Dropout(dropout_p)
        self.attention = Attention("concat", hidden_size)
        self.gru = nn.GRU(emb_size, hidden_size) 
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size) 

    def forward(self, word_input, last_hidden, encoder_outputs):

        #get embedding
        emb = self.embedding(word_input)
        emb = self.dropout(emb)

        #get current hidden state
        rnn_output, hidden = self.gru(emb, last_hidden)

        #calculcate attn_weigh & apply it to encoder outputs
        attn_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        #attn vector using hidden state & context 
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        # if p_gen:
        #     p_gen_input = torch.cat((context, rnn_output, x), 1)
            

        concat_input = torch.cat((rnn_output, context),1)
        concat_output = torch.tanh(self.linear(concat_input))
        #finnaly predict word 
        output = self.out(concat_output)
        return output, hidden, attn_weights




class AttnEncoderDecoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size):

        super(AttnEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, batch_X, lengths_X, max_length, batch_Y=None, use_teacher_forcing=True):

        # encoderに系列を入力（複数時刻をまとめて処理）
        encoder_outputs, encoder_hidden = self.encoder(batch_X, lengths_X)

        _batch_size = batch_X.size(1)

        # decoderの入力と隠れ層の初期状態を定義
        decoder_input = torch.tensor([START_DECODING_NO] * _batch_size, dtype=torch.long, device=device)
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)
        decoder_hidden = encoder_hidden  # Encoderの最終隠れ状態を取得
        decoder_outputs = torch.zeros(max_length, _batch_size, self.decoder.output_size).to(device)

        # 各時刻ごとに処理
        for t in range(max_length):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[t] = decoder_output

            # 次の時刻のdecoderの入力を決定
            if use_teacher_forcing and batch_Y is not None:  # ターゲット系列を用いる
                decoder_input = batch_Y[t].unsqueeze(0)
            else:  # 自身の出力を用いる
                decoder_input = decoder_output.max(-1)[1].unsqueeze(0)
                # ni = decoder_input[0][0]
                # if ni == STOP_DECODING_NO:
                #     break
        return decoder_outputs





















