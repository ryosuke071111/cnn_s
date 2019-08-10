import torch
import torch.nn as nn
# from attn_model import EncoderDecoder
from data import data_generate
from vocab_from_file import Vocab,generateVocab
from data_loader import DataLoader
from model import VanilaEncoderDecoder
from attn_model import AttnEncoderDecoder
import sys
import os
import  argparse
from utils import *
from pythonrouge.pythonrouge import Pythonrouge
from beam_decoder_attn import AttnBeamEncoderDecoder
from beam_decoder import VanilaBeamEncoderDecoder


# #parser設定
parser = argparse.ArgumentParser()
parser.add_argument('--model', default="attn", help="type 'attn' or 'vanila'")
parser.add_argument('--datasize', help="datasize used for training", type=int)
parser.add_argument('--epoch', help="num of epochs trained for the params")
parser.add_argument('--decode',default="beam", help="greedy or beam")
parser.add_argument('--bidirectional',help="bi or uni when training",type=str)
parser.add_argument('--nameop',default="", help="bi or uni when training")
parser.add_argument('--batch', help="name option after param name",type=int)
parser.add_argument('--inputlen', help="name option after param name",type=int)

args = parser.parse_args()

model_name = args.model

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
print('---------------------------------------------------------')

# #モデル構築
if model_name == "vanila":
    if args.decode == "beam":
        model = VanilaBeamEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    else:
        model = VanilaEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    print('model is Vanila seq2seq')
elif model_name == "attn":
    if args.decode == "beam":
        model = AttnBeamEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    else:
        model = AttnEncoderDecoder(INPUT_VOCAB_SIZE,OUTPUT_VOCAB_SIZE,hidden_size).to(device)
    print('model is Attention seq2seq')

print('model',model)
#パラメタ読み込み & 流し込み
# param = "model_"+model_name+"_datasize_"+str(args.datasize)+"_epoch_"+str(num_epochs)+("_bi" if bidirectional else "")+"_"+args.nameop
# param = "model_attn_datasize_10000_epoch_200_bi_relutanh"
param = "model_attn_datasize_10000_epoch_150_bi_s_removed"
param_path= "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/params/"
params = torch.load(param_path+param,map_location=lambda storage, loc: storage)

model.encoder.load_state_dict(params["encoder_state_dict"])
model.decoder.load_state_dict(params["decoder_state_dict"])
model.eval()
print('Model successfully constructed !!')
print()


#辞書読み込み
vocab = generateVocab()

#データ構築
articles,abstracts = data_generate(vocab, int(args.datasize))
data_loader = DataLoader(articles,abstracts,1,shuffle=False)

result_path = "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/model_output/"+param+"_"+args.decode
ref_path = "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/ref_summary/"

# result_path = "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/model_output_val/"+param+"_"+args.decode
# ref_path = "/home/ryosuke/desktop/nlp/summarization/cnn_seq2seq/ref_summary_val/"

try:
    os.mkdir(result_path)
except FileExistsError:
    pass


for i in range(num_of_data):
    batch_X, batch_Y,lengths_X = next(data_loader)

    with torch.no_grad():
        output = model(batch_X, lengths_X, max_length=MAX_OUTPUT_LENGTH, batch_Y=None,use_teacher_forcing=False)


    #greedy　decoding
    if args.decode == "greedy":
        output = output.max(dim=-1)[1].view(-1).data.cpu().tolist()

    batch_X = batch_X.view(-1).data.cpu().tolist()
    batch_Y = batch_Y.view(-1).data.cpu().tolist()

    source_text = [vocab._id2word(word)+" " for word in batch_X]
    ground_truth = [vocab._id2word(word)+" " for word in batch_Y]
    
    output_sentence = [vocab._id2word(word)+" " for word in output if word != 0]
    
    with open(result_path+"/output"+str(i)+".txt",mode="w") as f:
        f.write("".join(output_sentence))
    with open(ref_path+"/output"+str(i)+".txt",mode="w") as f:
        f.write("".join(ground_truth))
    
    if i % 100 == 0:
        print('***********iteration***********',i)
        print('groud trugh',ground_truth)
        print()
        print('output',output_sentence)
        print()
# #ROUGE計算
try:
    os.mkdir(result_path)
except FileExistsError:
    pass


rouge = Pythonrouge(summary_file_exist=True,
                    peer_path=result_path, model_path=ref_path,
                    n_gram=3, ROUGE_SU4=False, ROUGE_L=True,
                    recall_only=True,
                    stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=400,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)



