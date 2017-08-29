import numpy
import string

def init_weight(Mi,Mo):
    return np.random.randn(Mi,Mo)/np.sqrt(Mi+Mo)

def remove_punctuation(s):
    return s.translate(None, string.punctuation)

def get_robert_frost():
    word2idx={'START':0,'END':1}
    currentIdx = 2
    sentences = []
    for line in open('RobertFrost'):
        line = line.strip()
        sentence = []
        if line:
            tokens = remove_punctuation(line).split()
            for token in tokens:
                if token not in word2idx:
                    word2idx[token] = currentIdx
                    currentIdx = currentIdx + 1
                sentence.append(word2idx[token])
            sentences.append(sentence)
    return sentences,word2idx
