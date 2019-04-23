
import pandas as pd
import numpy as np
import io
import re
import sys


def normalize(s):

    s = s.lower()
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    s = s.replace('~', '')
    s = s.replace('!', '')
    s = s.replace('$', '')
    s = s.replace('\"', '')
    s = s.replace('\'', '')
    s = s.replace('/', '')
    s = s.replace('-', '')
    s = s.replace('.', '')
    s = s.replace('%', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace(',', '')
    s = s.replace('?', '')
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')

    return s


def get_vocabulary(df, columns, min_count=2):

    words = []

    for col in columns:
        for item in df[col].tolist():
            for w in normalize(item).split(' '):
                words.append(w)

    print('Total number of words in readfile: %i' % len(words))

    uniqueValues, occurCount = np.unique(words, return_counts=True)

    print('Total number of unique words: %i' % len(uniqueValues))

    vocab = uniqueValues[occurCount > min_count].tolist()

    print('Total number of words in vocabulary: %i' % len(vocab))

    return vocab


def load_vectors(fname, vocab):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    x = 0
    printfreq = 300000
    for line in fin:
        x += 1
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocab:
            data[tokens[0]] = [float(num) for num in tokens[1:]]
        if x % printfreq == 0:
            print('Searched %i words' % x)
    print('Total number of words in vocabDict: %i' % len(data))
    return data


def embed_col(col, vocabDict):

    vocab = vocabDict.keys()
    embedded = []

    for elt in col:
        words = normalize(elt).split(' ')
        embedded.append(
            np.array([vocabDict[w] for w in words if w in vocab]))

    return embedded


def main(embeddingfile, savefile, readfile, *columns):

    df = pd.read_csv(readfile)

    for col in columns:
        assert col in df.columns

    vocab = get_vocabulary(df, columns, min_count=2)

    vocabDict = load_vectors(embeddingfile, vocab)

    embedded = [embed_col(df[col].tolist(), vocabDict) for col in columns]

    np.savez(savefile, *embedded)


if __name__ == '__main__':
    main(*sys.argv[1:])
