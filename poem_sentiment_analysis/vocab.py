"""
Text Serialization
"""


class Vocab:
    UNK_TAG = "<UNK>"  # Represents unknown characters.
    PAD_TAG = "<PAD>"  # Padding symbol.
    PAD = 0
    UNK = 1

    def __init__(self):
        self.dict = {  # Stores words and their corresponding numbers.
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.count = {}  # Dictionary for counting word frequencies.

    def fit(self, sentence):
        """
        Accepts a sentence and counts word frequency.
        :param sentence:[str,str,str]
        :return:None
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1  # After processing all sentences, self.count will contain the frequency of each word.

    def build_vocab(self, min_count=1, max_count=None, max_features=None):
        """
        Constructs a vocabulary based on specified conditions.
        :param min_count: Minimum word frequency for inclusion in the vocabulary.
        :param max_count: Maximum word frequency for inclusion.
        :param max_features: Maximum number of words to include in the vocabulary.
        :return: None
        """
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count >= min_count}
        if max_count is not None:
        # Sorts the count dictionary and keeps only the top 'max_features' number of entries.
            self.count = {word: count for word, count in self.count.items() if count <= max_count}
        if max_features is not None:
            # [(k,v),(k,v)....] --->{k:v,k:v}
            self.count = dict(sorted(self.count.items(), lambda x: x[-1], reverse=True)[:max_features])

        for word in self.count:
            self.dict[word] = len(self.dict)  # Each word is assigned a unique number.

        # Inverting the dictionary to map numbers back to words.
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        """
        Transforms a sentence into a sequence of numbers.
        :param sentence: [str, str, str] - List of words in the sentence.
        :param max_len: Maximum length of the transformed sequence.
        :return: [int, int, int] - The transformed sentence as a sequence of numbers.
        """
        if len(sentence) > max_len:
            sentence = sentence[:max_len]
        else:
            sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))  # Padding the sentence with the PAD token if it's shorter than max_len.

        return [self.dict.get(i, 1) for i in sentence]

    def inverse_transform(self, incides):
        """
        Transforms a sequence of numbers back into words.
        :param incides: [int, int, int] - Sequence of numbers.
        :return: [str, str, str] - The corresponding sequence of words.
        """
        return [self.inverse_dict.get(i, "<UNK>") for i in incides]

    def __len__(self):
        return len(self.dict)

# # Debugging code (commented out)
# if __name__ == '__main__':
#     sentences = [["今天", "天气", "很", "好"],
#                  ["今天", "去", "吃", "什么"]]
#     ws = Vocab()
#     for sentence in sentences:
#         # 统计词频
#         ws.fit(sentence)
#     # 构造词典
#     ws.build_vocab(min_count=1)
#     print(ws.dict)
#     # 把句子转换成数字序列
#     ret = ws.transform(["好", "好", "好", "好", "好", "好", "好", "热", "呀"], max_len=13)
#     print(ret)
#     # 把数字序列转换成句子
#     ret = ws.inverse_transform(ret)
#     print(ret)
#     pass
