import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk import bleu_score

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


# デバイスの設定
torch.manual_seed(1)
random_state = 42


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: int, 入力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, seqs, input_lengths, hidden=None):
#         print("seqs",seqs.shape)
        """
        :param seqs: tensor, 入力のバッチ, size=(max_length, batch_size)
        :param input_lengths: 入力のバッチの各サンプルの文長
        :param hidden: tensor, 隠れ状態の初期値, Noneの場合は0で初期化される
        :return output: tensor, Encoderの出力, size=(max_length, batch_size, hidden_size)
        :return hidden: tensor, Encoderの隠れ状態, size=(1, batch_size, hidden_size)
        """
        
        emb = self.embedding(seqs)
        # print("input_lengths",input_lengths)
        packed = pack_padded_sequence(emb, input_lengths)
        output, hidden = self.gru(packed, hidden)
        output, _ = pad_packed_sequence(output)  #n*batch*dim
        

        # print('Encoder output shape',output.size())
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        """
        :param hidden_size: int, 隠れ層のユニット数
        :param output_size: int, 出力言語の語彙数
        :param dropout: float, ドロップアウト率
        """
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, seqs, hidden):
        """
        :param seqs: tensor, 入力のバッチ, size=(1, batch_size)
        :param hidden: tensor, 隠れ状態の初期値, Noneの場合は0で初期化される
        :return output: tensor, Decoderの出力, size=(1, batch_size, output_size)
        :return hidden: tensor, Decoderの隠れ状態, size=(1, batch_size, hidden_size)
        """
        # print('seqs_shape',seqs.size())
        # print('hidden_shape',hidden.size())
        # print('hidden[-1]',hidden[-1].size())
        # print('hidden size',hidden[:,0].size())
        emb = self.embedding(seqs)
        output, hidden = self.gru(emb, hidden)
        output = self.out(output)
        # output = self.softmax(output)
        return output.squeeze(0), hidden

class VanilaEncoderDecoder(nn.Module):
    """EncoderとDecoderの処理をまとめる"""
    def __init__(self, input_size, output_size, hidden_size):
        """
        :param input_size: int, 入力言語の語彙数
        :param output_size: int, 出力言語の語彙数
        :param hidden_size: int, 隠れ層のユニット数
        """
        super(VanilaEncoderDecoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, batch_X, lengths_X, max_length, batch_Y=None, use_teacher_forcing=True):
        """
        :param batch_X: tensor, 入力系列のバッチ, size=(max_length, batch_size)
        :param lengths_X: list, 入力系列のバッチ内の各サンプルの文長
        :param max_length: int, Decoderの最大文長
        :param batch_Y: tensor, Decoderで用いるターゲット系列
        :param use_teacher_forcing: Decoderでターゲット系列を入力とするフラグ
        :return decoder_outputs: tensor, Decoderの出力,
            size=(max_length, batch_size, self.decoder.output_size)
        """
        # encoderに系列を入力（複数時刻をまとめて処理）
        _, encoder_hidden = self.encoder(batch_X, lengths_X)

        _batch_size = batch_X.size(1)

        # decoderの入力と隠れ層の初期状態を定義
        
        decoder_input = torch.tensor([START_DECODING_NO] * _batch_size, dtype=torch.long, device=device)
        # print('decoder input',decoder_input)
        # print('decoder input',decoder_input.size())
        decoder_input = decoder_input.unsqueeze(0)  # (1, batch_size)
        decoder_hidden = encoder_hidden  # Encoderの最終隠れ状態を取得
        # decoderの出力のホルダーを定義
        
        decoder_outputs = torch.zeros(max_length, _batch_size, self.decoder.output_size, device=device)

        # 各時刻ごとに処理
        ##********************あとで出力が終わっているのにまだ出力しようとする機構を削除する****************
        #手順；abstractの最後にEOSを入れる
        #decodeの最大のtensorの確率のidがEOSのものならdecodeの出力をそこで終わらせる
        
        for t in range(max_length):
            # print('decoder input',decoder_input)
            # print('decoder input',decoder_input.size())
            #vocablaryに対する確率を出力する（e.g.  ...,[-0.5710,  4.0414,  2.7192,  ..., -0.6510,  0.2303, -1.0515],）
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # print('decoder_output',torch.max(decoder_output[0][0])) #これみると面白い（正解データんに対する確率が上がっていく）

            decoder_outputs[t] = decoder_output
            # 次の時刻のdecoderの入力を決定
            if use_teacher_forcing and batch_Y is not None:  # ターゲット系列を用いる
                decoder_input = batch_Y[t].unsqueeze(0)
            else:  # 自身の出力を用いる
                decoder_input = decoder_output.max(-1)[1].unsqueeze(0)
                ni = decoder_input[0][0]
                if ni == EOS:
                    break

        return decoder_outputs























