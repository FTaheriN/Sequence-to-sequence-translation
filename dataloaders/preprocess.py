import pandas as pd

import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')



def read_file(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None).iloc[:100000, :2]
    df = df.rename(columns={0: 'english', 1: 'spanish'})
    return df #df.iloc[:,0], df.iloc[:,1]


# remove punctuations
def remove_punc(sentence):
    return re.sub(r"[\/\.\:\;\»\«\\\¡\¿\!\?\^\`\~\#\$\&\%\)\(\<\>,\،\^\+\_\-\*\"\t\n+]"," ",sentence)


def remove_numbers(sentences):
  return re.sub(r'\d+','',sentences)


def english_preprocessing(row):
    sentences = sent_tokenize(row)
    new_sentences = []
    for sent in sentences:
        new_sent = remove_punc(sent.lower()) #.lower()
        new_sent = remove_numbers(new_sent)
        # new_sent = word_tokenize(new_sent)
        new_sent = new_sent.split()
        new_sentences += (new_sent)
  
    return new_sentences


def spanish_preprocessing(row):
    sentences = sent_tokenize(row, language='spanish')
    new_sentences = []
    for sent in sentences:
        new_sent = remove_punc(sent.lower()) #.lower()
        new_sent = remove_numbers(new_sent)
        # new_sent = toktok.tokenize(new_sent)
        new_sent = new_sent.split()
        new_sentences += (new_sent)
    
    return new_sentences


def preprocess(file_path):
    df = read_file(file_path)

    df['english'] = df['english'].apply(english_preprocessing)
    df['spanish'] = df['spanish'].apply(spanish_preprocessing)

    df['eng_length'] = df['english'].apply(lambda t: len(t))
    df['spn_length'] = df['spanish'].apply(lambda t: len(t))
    return df