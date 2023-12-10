import re
from zhon.hanzi import punctuation

# Poem segmentation
def tokenlize(sentence):
    """
    Perform text tokenization.
    :param sentence: str
    :return: [str,str,str]
    """
    
    # Defining a list of characters to be filtered out.
    fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”',
                '“', ]
    # Using regular expression to remove all the defined filter characters from the sentence.
    sentence = re.sub("|".join(fileters), "", sentence)
    # Assigning Chinese punctuation characters to 'punctuation_str'.
    punctuation_str = punctuation
    
    for i in punctuation_str:
        # Removing all Chinese punctuation characters from the sentence.
        sentence = sentence.replace(i, '')
    sentence=' '.join(sentence)
    result = [i for i in sentence.split(" ") if len(i) > 0]
    return result


if __name__ == '__main__':
    res=tokenlize('岱宗夫如何？齐鲁青未了。造化钟神秀，阴阳割昏晓。荡胸生曾云，决眦入归鸟。( 曾 同：层)会当凌绝顶，一览众山小。')
    print(res)
