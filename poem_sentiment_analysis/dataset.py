# -*-coding:utf-8-*-
import os
import pickle
import re
import zipfile
import jieba
from zhon.hanzi import punctuation
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        if train == True:
            url = 'data/train.xlsx'
        else:
            url = "data/test.xlsx"
        data = pd.read_excel(url)
        sentence = data.get('sentence')
        label = data.get('label')
        self.sentence_list=sentence
        self.label_list=label



    def __getitem__(self, idx):
        line_text=self.sentence_list[idx]
        #get comments from txt and tokenize
        review = tokenlize(str(line_text))
        #get labels corresponding to comments
        label = int(self.label_list[idx])
        return review, label

    def __len__(self):
        return len(self.sentence_list)


#tokenize poems
def tokenlize(sentence):
    """
    tokenize text 
    :param sentence: str
    :return: [str,str,str]
    """

    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    sentence = re.sub("|".join(fileters), "", sentence)
    punctuation_str = punctuation
    for i in punctuation_str:
        sentence = sentence.replace(i, '')
    sentence=' '.join(sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]
    return result

# Here is the debugging code
def collate_fn(batch):
    """
    process batch data
    :param batch: [one getitem result,one getitem result,one getitem result]
    :return: tuple
    """
    reviews, labels = zip(*batch)

    return reviews, labels



if __name__ == "__main__":
    from poem_sentiment_analysis import Vocab
    imdb_dataset = ImdbDataset(True)
    my_dataloader = DataLoader(imdb_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    for review,label in my_dataloader:
        vocab_model = pickle.load(open("./model/vocab.pkl", "rb"))
        print(review[0])
        result = vocab_model.transform(review[0], 30)
        print(result)
        break
