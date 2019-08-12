import glob
from tensorflow.core.example import example_pb2
import struct
from nltk.tokenize import word_tokenize, sent_tokenize
from utils import *
from tqdm import tqdm


PATH="/home/ryosuke/desktop/data_set/cnn_stories_tokenized/"


def sent_split(text):
    words =[sent for sent in sent_tokenize(text)]
    words = list(map(lambda x:x.split(),words))
    return [word for inner_list in words for word in inner_list]



def data_generate(vocab, num_of_data):
    file = open(PATH+"train.bin","rb")
    # file = open(PATH+"val.bin","rb")
    #valでやって見る

    articles = []
    abstracts = []
    articles_extend = []
    abstracts_extend = []
    oovs = []
    print('# of data', num_of_data)

    i=0
    pbar = tqdm(total=num_of_data)
    while i<num_of_data:
        len_bytes = file.read(8)
        if not len_bytes:
            print('finishied reading this files')
            break

        #連続するバイト列から単位ごとに区切る
        str_len = struct.unpack('q',len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, file.read(str_len))[0]

        #区切られた単位からarticle/abstractを取り出す
        data = example_pb2.Example.FromString(example_str)
        article = data.features.feature["article"].bytes_list.value[0]
        if len(article)<=0:
          continue
        abstract = data.features.feature["abstract"].bytes_list.value[0]

        #バイトコードを文字列に変換する（articleの場合はセンテンスとの一つずつに<s><\s>を入れている）
        article = sent_split(article.decode())[:MAX_INPUT_LENGTH]
        abstract = " ".join(vocab.abstract2sents(abstract.decode())).split()[:MAX_OUTPUT_LENGTH]
        
        
        
        #IDに変更している（辞書増やし版）
        article_vocab_extend, oov = vocab.article2ids(article)
        abstract_vocab_extend = [START_DECODING_NO]+vocab.abstract2ids(abstract, oov)+[STOP_DECODING_NO]

        #IDに変更している（辞書増やし版）
        article = [vocab._word2id(word) for word in article]
        abstract = [START_DECODING_NO]+[vocab._word2id(word) for word in abstract]+[STOP_DECODING_NO]

    
        articles.append(article)
        abstracts.append(abstract)
        oovs.append(oov)
        articles_extend.append(article_vocab_extend)
        abstracts_extend.append(abstract_vocab_extend)
        i+=1
        pbar.update(1)
    print('data successfully　constructed!')
    print()

    return articles, abstracts, oovs, articles_extend, abstracts_extend