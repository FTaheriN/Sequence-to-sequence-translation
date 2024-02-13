import random

import torch
import torch.nn as nn

from .encoder import EncoderLSTM
from .decoder import DecoderLSTM


class Seq2Seq(nn.Module):
    def __init__(self, embedding, output, pretrained_input_embeddings, teacher_forcing_ratio=0.4):
        super(Seq2Seq, self).__init__()

        self.tchr_frc_ratio = teacher_forcing_ratio
        self.hidden_dim = 100
        self.output = output 
        self.encoder = EncoderLSTM(embedding, self.hidden_dim, 1, pretrained_input_embeddings)
        self.decoder = DecoderLSTM(output, embedding, self.hidden_dim, 1)


    def forward(self, input, target):
      out_len = target.shape[1]
      device = next(self.parameters()).device

      encoder_output, encoder_hidden = self.encoder(input)

      # print(encoder_hidden)
      decoder_input = torch.unsqueeze(target[:, 0], dim=1)
      decoder_hidden = encoder_hidden 
      
      
      output_matrix = torch.zeros((target.shape[0], out_len, self.output))

      output_idx = torch.zeros((target.shape[0], out_len))

      for di in range(1, out_len):
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        topv, topi = decoder_output.topk(1)

        teacher_force = random.random() < self.tchr_frc_ratio

        decoder_input = torch.unsqueeze(target[:, di], dim=1) if teacher_force else topi.squeeze(dim=1).detach()

        output = decoder_output.squeeze().to(device)

        output_matrix[:, di, :] = output

        output_idx[:, di] = topi.squeeze()

      return output_idx.to(device), output_matrix.to(device)