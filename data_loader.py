import torch
from utils import *
from sklearn.utils import shuffle

#batch内最大系列長に合わせる
def pad_seq_article(seq, max_length):
    # 系列(seq)が指定の文長(max_length)になるように末尾をパディングする
    res = seq + [PAD for i in range(max_length - len(seq))]
    return res

def pad_seq_abst(seq, max_length):
    # 系列(seq)が指定の文長(max_length)になるように末尾をパディングする
    res = seq + [PAD for i in range(max_length - len(seq))]
    return res 


class DataLoader(object):
    def __init__(self,X,Y,batch_size,shuffle=False):
        self.data = list(zip(X,Y))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.start_index= 0

        self.reset()
        print('data loader successfully　constructed!')

    def reset(self):
        if self.shuffle:
            self.data = shuffle(self.data,random_state=random_state)
        self.start_index=0

    def __iter__(self):
        return self

    def __next__(self):
        # print('start index', self.start_index)
        if self.start_index >= len(self.data):
            self.reset()
            raise StopIteration()

        seqs_X, seqs_Y = zip(*self.data[self.start_index:self.start_index+self.batch_size])
        #文章の長い順にソートする
        seq_pairs = sorted(zip(seqs_X,seqs_Y),key=lambda p:len(p[0]),reverse=True)
        seqs_X,seqs_Y = zip(*seq_pairs)

        #短い系列のパディング
        lengths_X=[len(s) for s in seqs_X if len(seqs_X)>0]
        # lengths_X = list(map(lambda x: 1 if x==0 else x,lengths_X))
        lengths_Y=[len(s) for s in seqs_Y if len(seqs_Y)>0]
        max_length_X = max(lengths_X)
        max_length_Y = max(lengths_Y)
        padded_X = [pad_seq_article(s,max_length_X) for s in seqs_X]
        padded_Y = [pad_seq_abst(s,max_length_Y) for s in seqs_Y]

        #tensorに変換
        batch_X = torch.tensor(padded_X,dtype=torch.long,device=device).transpose(0,1)
        batch_Y = torch.tensor(padded_Y,dtype=torch.long,device=device).transpose(0,1)

        self.start_index += self.batch_size
   

        return batch_X, batch_Y, lengths_X