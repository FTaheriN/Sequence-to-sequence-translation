from torchtext.data.metrics import bleu_score
import torch
import numpy as np
from dataloaders import MTDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def index_to_word(sentences, train_set):
    new_sentences = []
    for sentence in sentences:
        new_sent = np.array([train_set.spn_index_vocab[index.item()] for index in sentence if index > 3])
        new_sentences.append(new_sent)

    return np.array(new_sentences)


def BLEU_score(trgts, preds, train_set):
    trgt_stncs, pred_stncs = [], []
    score = 0
    trgt_sentences = index_to_word(trgts, train_set)

    pred_sentences = index_to_word(preds, train_set)

    return bleu_score(pred_stncs, trgt_stncs)

def accuracy_word_wise(batch_target, batch_preds):
  sos_poses = batch_target == MTDataset.SOS
  eos_poses = batch_target == MTDataset.EOS
  pad_poses = batch_target == MTDataset.PAD
  
  ignored_poses = torch.logical_or(pad_poses, torch.logical_or(sos_poses, eos_poses))

  eq_mat = torch.eq(batch_target, batch_preds)
  ignrd_eq_mat = torch.logical_or(eq_mat, ignored_poses)

  trgt_matches = (torch.sum(ignrd_eq_mat, dim=0) - torch.sum(ignored_poses, dim=0))
  num_valid_cmprs = torch.ones(batch_target.shape[1], device=batch_target.device)*batch_target.shape[0] - torch.sum(ignrd_eq_mat, dim=0)
  indices = torch.nonzero(num_valid_cmprs)

  col_acc = torch.divide(trgt_matches[indices], num_valid_cmprs[indices]) # Correct predictions ratio in each column (token position)
  return torch.nanmean(col_acc) 


def calc_bleu(model, loader):
  model.eval()
  total_avg = 0
  for batch_input, batch_target in loader:
    with torch.no_grad():
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)

      out_idx, _ = model(batch_input, batch_target)
      total_avg += BLEU_score(batch_target, out_idx)

  return total_avg/len(loader)


def calc_acc(model, loader):
  model.eval()
  total_avg = 0
  for batch_input, batch_target in loader:
    with torch.no_grad():
      batch_input = batch_input.to(device)
      batch_target = batch_target.to(device)

      out_idx, _ = model(batch_input, batch_target)
      total_avg += accuracy_word_wise(batch_target, out_idx)

  return total_avg/len(loader)