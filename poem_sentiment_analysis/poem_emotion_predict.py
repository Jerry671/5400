import torch, pickle, re
from zhon.hanzi import punctuation
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

class Vocab:
    UNK_TAG = "<UNK>"  # Represents an unknown character
    PAD_TAG = "<PAD>"  # Padding symbol
    PAD = 0
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # Stores words and corresponding numbers
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # Count word frequency
    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  # After fitting all sentences, self.count will have the frequency of all words
    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # Each word corresponds to a number

        # Invert the dictionary
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # 填充PAD

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        """
        Convert number sequence back to characters
        :param incides: [int, int, int]
        :return: [str, str, str]
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)

train_batch_size = 128
test_batch_size = 128
voc_model = pickle.load(open("./models/vocab.pkl", "rb"))
sequence_max_len = 100



class ImdbModel(nn.Module):
    def __init__(self):
        super(ImdbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(voc_model), embedding_dim=200, padding_idx=voc_model.PAD).to()
        self.lstm = nn.LSTM(input_size=200, hidden_size=64, num_layers=6, batch_first=True, bidirectional=True,
                            dropout=0.1)
        self.fc1 = nn.Linear(64 * 2, 64)
        self.fc2 = nn.Linear(64, 7)

    def forward(self, input):
        """
        Forward pass of the model
        :param input: [batch_size, max_len]
        :return: Model output
        """
        input_embeded = self.embedding(input)  # input embeded :[batch_size,max_len,200]

        output, (h_n, c_n) = self.lstm(input_embeded)  # h_n :[4,batch_size,hidden_size]
        # out :[batch_size,hidden_size*2]
        # Concatenate the last output of the forward and backward LSTM
        out = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1) 

        # Fully connected layers
        out_fc1 = self.fc1(out)
        # Aplly relu
        out_fc1_relu = F.relu(out_fc1)

        # Final fully connected layer
        out_fc2 = self.fc2(out_fc1_relu)  # out :[batch_size,2]
        return F.log_softmax(out_fc2, dim=-1)

def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
#诗词分割
def tokenlize(sentence):
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
def poem_emotion_predict(line):
    sequence_max_len=100

    model = torch.load('./models/lstm_model.pkl',map_location=torch.device('cpu'))
    model.to(device())
    print(line)
    review = tokenlize(line)
    # review=tokenlize(line)
    vocab_model = pickle.load(open("./models/vocab.pkl", "rb"))
    result = vocab_model.transform(review,sequence_max_len)
    # print(result)
    data = torch.LongTensor(result).to(device())
    data=torch.reshape(data,(1,sequence_max_len)).to(device())
    # print(data.shape)
    output = model(data)
    data=output.data.cpu().numpy()
    #['悲', '惧', '乐', '怒', '思', '喜', '忧']
    dit={}
    sum=0
    for i in range(len(data[0])):
        sum+=abs(float(data[0][i]))
        if i==0:
            dit['悲']=abs(float(data[0][i]))
        if i==1:
            dit['惧'] = abs(float(data[0][i]))
        if i==2:
            dit['乐']=abs(float(data[0][i]))
        if i==3:
            dit['怒'] = abs(float(data[0][i]))
        if i==4:
            dit['思']=abs(float(data[0][i]))
        if i==5:
            dit['喜'] =abs(float(data[0][i]))
        if i==6:
            dit['忧']=abs(float(data[0][i]))
    #dit=dict(sorted(dit.items(), key=lambda item: item[1], reverse=True))
    for key,value in dit.items():
        val=round((1-value/sum)*100,2)
        dit[key]=val
    dit = dict(sorted(dit.items(), key=lambda item: item[1], reverse=True))
    jsonData=[]
    name=[]
    val=[]
    for key,value in dit.items():
        name.append(key)
        val.append(value)
        #print(key+" "+str(value))
    dit={}
    dit['name']=name
    dit['value']=val
    jsonData.append(dit)
    # print(output.data.max(1, keepdim=True)[0].item())
    pred = output.data.max(1, keepdim=True)[1]  # Get the position of the maximum value,[batch_size,1]
    # print(pred.item())
    #Emotions: ['Sadness', 'Fear', 'Joy', 'Anger', 'Thoughtfulness', 'Happiness', 'Anxiety']
    if pred.item() == 0:
        print("悲")
        return "情感：悲", jsonData
    elif pred.item() == 1:
        print("惧")
        return "情感：惧", jsonData
    elif pred.item() == 2:
        print("乐")
        return "情感：乐", jsonData
    elif pred.item() == 3:
        print("怒")
        return "情感：怒", jsonData
    elif pred.item() == 4:
        print("思")
        return "情感：思", jsonData
    elif pred.item() == 5:
        print("喜")
        return "情感：喜", jsonData
    elif pred.item() == 6:
        print("忧")
        return "情感：忧", jsonData

if __name__ == '__main__':
    def poem_emotion_prediction(line):
        # Add your poetry emotion prediction function here
        result = poem_emotion_predict(line)
        return result

    iface = gr.Interface(
        fn=poem_emotion_prediction,
        inputs="text",
        outputs=["text","json"],
        title="Poem sentiment analysis",
        description="Bozhi Hao, Quan Yuan, Zhuoyan Guo, Xueyi Tan"
    )
    iface.launch()