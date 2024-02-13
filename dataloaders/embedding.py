import torch
import numpy as np



def load_embeddings(embed_path):
    embeddings_index = {}
    f = open(embed_path, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def build_embedding(eng_vocab_index, embedding_vectors, unk_index, embedding_size=100):
    vocab_size = len(eng_vocab_index)
    embedding_matrix = torch.zeros((vocab_size, embedding_size))
    for word, index in eng_vocab_index.items():
      try:
        word_embed = embedding_vectors[word]
        embedding_matrix[index] = word_embed
      except:
        eng_vocab_index[word] = unk_index
    return embedding_matrix