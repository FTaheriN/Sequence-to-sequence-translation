import torch
from torch.utils.data import Dataset

from split import data_split
from .preprocess import preprocess
from .vocabulary import vocab_index
from .embedding import load_embeddings, build_embedding



class MTDataset(Dataset):

    SOS = 0
    EOS = 1
    PAD = 2
    UNK = 3

    def __init__(self, df_path, gv_path, mode):
        super(MTDataset, self).__init__()

        self.mode = mode

        self.df = preprocess(df_path)

        self.embedding_vectors = load_embeddings(gv_path)

        self.eng_vocab_index, self.spn_vocab_index, \
              self.eng_index_vocab, self.spn_index_vocab = vocab_index(self.df)

        print(self.eng_vocab_index)


        self.embeddings = build_embedding(self.eng_vocab_index, self.embedding_vectors, self.UNK)

        self.embeddings[self.UNK, :] = torch.mean(self.embeddings, 0)

        self.eng_max_len = max(self.df['eng_length'])
        self.spn_max_len = max(self.df['spn_length'])

        self.train_df, self.test_df = data_split(self.df)



    def get_glove_weights(self):
        return torch.tensor(self.embeddings)


    def get_output_spec(self):
        return len(self.spn_vocab_index) + 4, self.spn_max_len
    

    def __getitem__(self, index):
        
      if self.mode == "train":
          row = self.train_df.iloc[index]
      else: 
          row = self.test_df.iloc[index]

      eng_indexed = []
      eng_indexed.extend([self.eng_vocab_index[token] for token in row[0]])
      eng_indexed.append(self.EOS)
      eng_indexed.extend([self.PAD]*(self.eng_max_len - row[2]))
      
      spn_indexed = [self.SOS]
      spn_indexed.extend([self.spn_vocab_index[token] for token in row[1]])
      spn_indexed.append(self.EOS)
      spn_indexed.extend([self.PAD]*(self.spn_max_len - row[3]))

      return torch.tensor(eng_indexed), torch.tensor(spn_indexed)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_df)
        else:
            return len(self.test_df)