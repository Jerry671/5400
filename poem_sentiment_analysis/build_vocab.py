# -*-coding:utf-8-*-
import pickle

from tqdm import tqdm

from poem_sentiment_analysis import dataset
# from poem_sentiment_analysis.imdb_sentiment.vocab import Vocab
from torch.utils.data import DataLoader

class Vocab:
    UNK_TAG = "<UNK>"  # unknown character
    PAD_TAG = "<PAD>"  # filler character
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # Save words and their matching number
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # count term frequency

    def fit(self, sentence):
        """
        Accept sentences and count term frequency
        :param sentence:[str,str,str]
        :return:None
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  #after fitting all sentences, self.count to get the word frequency of all terms

    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        """
        Construct dictionary according to the requirement
        :param min_count:minimum term frequency
        :param max_count: maximum term frequency
        :param max_features: maximum term number
        :return:
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # Each word corresponds to a number

        # reverse dict
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        Convert sentences into sequences of numbers
        :param sentence:[str,str,str]
        :return: [int,int,int]
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # fill PAD

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        """
        Convert sequences of numbers into characters
        :param incides: [int,int,int]
        :return: [str,str,str]
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)

def collate_fn(batch):
    """
    process batch data
    :param batch: [one getitem result,one getitem result,one getitem result]
    :return: tuple
    """
    reviews, labels = zip(*batch)

    return reviews, labels



def get_dataloader(train=True):
    imdb_dataset = dataset.ImdbDataset(train)
    my_dataloader = DataLoader(imdb_dataset, batch_size=200, shuffle=True, collate_fn=collate_fn)
    return my_dataloader


if __name__ == '__main__':

    ws = Vocab()
    dl_train = get_dataloader(True)
    dl_test = get_dataloader(False)
    for reviews, label in tqdm(dl_train, total=len(dl_train)):
        for sentence in reviews:
            ws.fit(sentence)
    for reviews, label in tqdm(dl_test, total=len(dl_test)):
        for sentence in reviews:
            ws.fit(sentence)
    ws.build_vocab()
    print(len(ws))

    pickle.dump(ws, open("./model/vocab.pkl", "wb"))
