# -*-coding:utf-8-*-
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from 情感分析.诗词情感分析 import dataset
from 情感分析.诗词情感分析.vocab import Vocab

train_batch_size = 512
test_batch_size = 128
voc_model = pickle.load(open("./model/vocab.pkl", "rb"))
sequence_max_len = 100


def collate_fn(batch):
    """
    Process the batch data to prepare it for the model
    :param batch: A list containing results from dataset's getitem
    :return: Tuple of reviews and labels in tensor format
    """
    reviews, labels = zip(*batch)
    reviews = torch.LongTensor([voc_model.transform(i, max_len=sequence_max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    return reviews, labels


def get_dataloader(train=True):
    """
    Create a DataLoader for the IMDb dataset
    :param train: Boolean indicating whether to get training data or test data
    :return: DataLoader for the specified dataset
    """
    imdb_dataset = dataset.ImdbDataset(train)
    batch_size = train_batch_size if train else test_batch_size
    return DataLoader(imdb_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


class ImdbModel(nn.Module):
    def __init__(self):
        """
        Initialize the ImdbModel with embedding, LSTM, and fully connected layers
        """
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=200, padding_idx=voc_model.PAD).to()
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=6, batch_first=True, bidirectional=True,
                            dropout=0.1)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, input):
        """
        Forward pass of the model
        :param input: Input data [batch_size, max_len]
        :return: Log-softmax output from the model
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)  # # Concatenate the last output of forward and backward


        # Fully connected layer
        out_fc1 = self.fc1(out)

        # Apply ReLU
        out_fc1_relu = F.relu(out_fc1)

        # Another fully connected layer
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)


def device():
    """
    Determine the device to use for training (GPU or CPU)
    :return: Device type
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def train(imdb_model, epoch):
    """
    Train the ImdbModel for a specified number of epochs
    :param imdb_model: The model to train
    :param epoch: Number of epochs for training
    """
    train_dataloader = get_dataloader(train=True)

    optimizer = Adam(imdb_model.parameters())
    for i in range(epoch):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, (data, target) in enumerate(bar):
            optimizer.zero_grad()
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            bar.set_description("epcoh:{}  idx:{}   loss:{:.6f}".format(i, idx, loss.item()))
        # if epoch%5==0:
        #     test(imdb_model)
    torch.save(imdb_model, 'model/lstm_model.pkl')


def test(imdb_model):
    """
    Validate the trained model on the test dataset
    :param imdb_model: The trained model to validate
    """
    test_loss = 0
    correct = 0
    total_samples = 0
    true_positive = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    false_positive = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    false_negative = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

    imdb_model.eval()
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for data, target in tqdm(test_dataloader):
            data = data.to(device())
            target = target.to(device())
            output = imdb_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # 获取最大值的位置,[batch_size,1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            total_samples += len(target)

            # Calculate precision and recall
            for i in range(len(target)):
                true_label = target[i].item()
                predicted_label = pred[i].item()
                if true_label == predicted_label:
                    true_positive[true_label] += 1
                else:
                    false_positive[predicted_label] += 1
                    false_negative[true_label] += 1

    test_loss /= len(test_dataloader.dataset)
    accuracy = 100. * correct / total_samples

    # Calculate precision and recall for each label
    precision = {}
    recall = {}
    for label in range(7):
        precision[label] = true_positive[label] / (true_positive[label] + false_positive[label])
        recall[label] = true_positive[label] / (true_positive[label] + false_negative[label])

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total_samples, accuracy))
    print("Precision:")
    for label, value in precision.items():
        print("Label {}: {:.2f}%".format(label, value * 100))
    print("Recall:")
    for label, value in recall.items():
        print("Label {}: {:.2f}%".format(label, value * 100))



def xlftest():
    """
    Perform a test to analyze the emotion of a given line of poetry
    """
    import numpy as np
    model = torch.load('model/lstm_model.pkl')
    model.to(device())
    from 情感分析.诗词情感分析.dataset import tokenlize
    # Emotions: Joy, Sadness, Anxiety, Thoughtfulness, Happiness, Anger, Fear
    lines = [
        '昔闻洞庭水，今上岳阳楼。吴楚东南坼，乾坤日夜浮。亲朋无一字，老病有孤舟。戎马关山北，凭轩涕泗流。'
    ]
    for line in lines:
        print(line)
        review = tokenlize(line)
        # review=tokenlize(line)
        vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
        result = vocab_model.transform(review, sequence_max_len)
        # print(result)
        data = torch.LongTensor(result).to(device())
        data = torch.reshape(data, (1, sequence_max_len)).to(device())
        # print(data.shape)
        output = model(data)
        data = output.data.cpu().numpy()
        # ['悲', '惧', '乐', '怒', '思', '喜', '忧']
        dit = {}
        sum = 0
        for i in range(len(data[0])):
            sum += abs(float(data[0][i]))
            if i == 0:
                dit['悲'] = abs(float(data[0][i]))
            if i == 1:
                dit['惧'] = abs(float(data[0][i]))
            if i == 2:
                dit['乐'] = abs(float(data[0][i]))
            if i == 3:
                dit['怒'] = abs(float(data[0][i]))
            if i == 4:
                dit['思'] = abs(float(data[0][i]))
            if i == 5:
                dit['喜'] = abs(float(data[0][i]))
            if i == 6:
                dit['忧'] = abs(float(data[0][i]))
        # dit=dict(sorted(dit.items(), key=lambda item: item[1], reverse=True))
        for key, value in dit.items():
            val = round((1 - value / sum) * 100, 2)
            dit[key] = val
        dit = dict(sorted(dit.items(), key=lambda item: item[1], reverse=True))
        for key, value in dit.items():
            print(key + " " + str(value))
        # print(output.data.max(1, keepdim=True)[0].item())
        pred = output.data.max(1, keepdim=True)[1]  # Get the position of the maximum value
        # print(pred.item())
        if pred.item() == 0:
            print("悲")
        elif pred.item() == 1:
            print("惧")
        elif pred.item() == 2:
            print("乐")
        elif pred.item() == 3:
            print("怒")
        elif pred.item() == 4:
            print("思")
        elif pred.item() == 5:
            print("喜")
        elif pred.item() == 6:
            print("忧")


if __name__ == '__main__':
    imdb_model = ImdbModel().to(device())
    train(imdb_model, 15)
    test(imdb_model)