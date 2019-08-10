import torch
from attn_model import AttnEncoderDecoder
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from vocab_from_file import Vocab,generateVocab

vocab = generateVocab()

class AttnBeamEncoderDecoder(AttnEncoderDecoder):
    def __init__(self, input_size, output_size, hidden_size, beam_size=4):

        super(AttnBeamEncoderDecoder, self).__init__(input_size, output_size, hidden_size)
        self.beam_size = beam_size

    def forward(self, batch_X, lengths_X, max_length=MAX_OUTPUT_LENGTH , batch_Y=None, use_teacher_forcing=True):

        _batch_size = batch_X.size(1)
        encoder_ouputs, encoder_hidden = self.encoder(batch_X, lengths_X)
        seq_length = encoder_ouputs.size()[0]
        
        # decoderの入力と隠れ層の初期状態を定義
        decoder_input = torch.tensor([START_DECODING_NO] * self.beam_size, dtype=torch.long, device=device)
        decoder_input = decoder_input.unsqueeze(0)
        decoder_hidden = encoder_hidden

        # beam_sizeの数だけrepeatする
        decoder_input = decoder_input.expand(1, self.beam_size)
        decoder_hidden = decoder_hidden.expand(1, self.beam_size, -1).contiguous()
        encoder_ouputs = encoder_ouputs.expand(seq_length, self.beam_size, -1)
        

        k = self.beam_size
        finished_beams = []
        finished_scores = []
        prev_probs = torch.zeros(self.beam_size, 1, dtype=torch.float, device=device)  # 前の時刻の各ビームの対数尤度を保持しておく
        output_size = self.decoder.output_size

        a=0
        # 各時刻ごとに処理
        for t in range(max_length):
            # decoder_input: (1, k)
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input[-1:], decoder_hidden, encoder_ouputs[:,:k,:])
   
            # decoder_hidden: (1, k, hidden_size)
            decoder_output_t = decoder_output  # (k, output_size)
            log_probs = prev_probs + F.log_softmax(decoder_output_t, dim=-1)  # (k, output_size)
            scores = log_probs  # 対数尤度をスコアとする
            # print('scores shape',scores.shape)

            # スコアの高いビームとその単語を取得
            flat_scores = scores.view(-1)  # (k*output_size,)
            # print('flat_scores',flat_scores.shape)
            if t == 0:
                flat_scores = flat_scores[:output_size]  # t=0のときは後半の同じ値の繰り返しを除外
            top_vs, top_is = flat_scores.data.topk(k)
            beam_indices = top_is / output_size  # (k,)  #k*output_sizeで1行にしてたのでdatasizeで割ってindicesをゲット（[0~4]）
            word_indices = top_is % output_size  # (k,)  #wordのindex欲しいので[0~50000]
            
            # ビームを更新する
            _next_beam_indices = []
            _next_word_indices = []

            for b, w in zip(beam_indices, word_indices):
                if w.item() == STOP_DECODING_NO:  # EOSに到達した場合はそのビームは更新して終了
                    k -= 1
                    beam = torch.cat([decoder_input.t()[b], w.view(1,)])  # (t+2,)
                    score = scores[b, w].item()
                    finished_beams.append(beam)
                    finished_scores.append(score)
                else:   # それ以外の場合はビームを更新
                    _next_beam_indices.append(b)
                    _next_word_indices.append(w)
            if k == 0:
                break

            # tensorに変換
            next_beam_indices = torch.tensor(_next_beam_indices, device=device)
            next_word_indices = torch.tensor(_next_word_indices, device=device)
            # 次の時刻のDecoderの入力を更新
            decoder_input = torch.index_select(
                decoder_input, dim=-1, index=next_beam_indices)
            decoder_input = torch.cat(
                [decoder_input, next_word_indices.unsqueeze(0)], dim=0)
    
            # 次の時刻のDecoderの隠れ層を更新
            decoder_hidden = torch.index_select(
                decoder_hidden, dim=1, index=next_beam_indices)

            # 各ビームの対数尤度を更新
            flat_probs = log_probs.view(-1)  # (k*output_size,)
            next_indices = (next_beam_indices + 1) * next_word_indices #beamのワードのindex取得
            prev_probs = torch.index_select(
                flat_probs, dim=0, index=next_indices).unsqueeze(1)  # (k, 1) # (k, 1) flat_probの中から選んだ単語の確率を抽出
            
        if k != 0:
            # print('next_beam_indices', next_beam_indices)
            # print('next_word_indices', next_word_indices)
            for i, (b, w) in enumerate(zip(next_beam_indices, next_word_indices)):
                beam = decoder_input.t()[i]
                score = scores[b, w].item()
                finished_beams.append(beam)
                finished_scores.append(score)
        
        # すべてのビームが完了したらデータを整形
        decoder_outputs = [[idx.item() for idx in beam[1:]] for beam in finished_beams]
    
        outputs, scores = zip(*sorted(zip(decoder_outputs, finished_scores), key=lambda x: -x[1]))


        return outputs[0] 