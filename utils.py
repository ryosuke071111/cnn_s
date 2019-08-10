import torch
# 特殊なトークン
PAD_TOKEN = '[PAD]'  # バッチ処理の際に、短い系列の末尾を埋めるために使う （Padding）
BOS_TOKEN = '<s>'  # 系列の始まりを表す （Beggining of sentence）
EOS_TOKEN = '</s>'  # 系列の終わりを表す （End of sentence）
UNK_TOKEN = '[UNK]'  # 語彙に存在しない単語を表す （Unknown）
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'

PAD = 0
UNK = 1
START_DECODING_NO = 2
STOP_DECODING_NO = 3
BOS = 4
EOS = 5


#データサイズや学習回数
num_of_data = 100000
batch_size = 40
num_epochs = 1000

# アーキテクチャ
MAX_INPUT_LENGTH = 400
MAX_OUTPUT_LENGTH = 100
INPUT_VOCAB_SIZE = 50000
OUTPUT_VOCAB_SIZE = 50000
hidden_size = 256
emb_size = 128
bidirectional=True
p_gen_=True

#ハイパーパラメタ
lr = 1e-3
teacher_forcing_rate = 0.4
MIN_COUNT = 2  # 語彙に含める単語の最低出現回数
torch.manual_seed(1)
random_state = 42


#ハードウェア
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PATH="/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/params"