import time
import numpy as np

import torch 
from torch.utils.data import DataLoader
import torch.optim as optim

from dataloaders import MTDataset
from deeplearning.train import *
from utils import read_yaml_config
from model import Seq2Seq
from utils import *

############################## Reading Model Parameters ##############################
config = read_yaml_config()
df_path = config['df_path'] 
epochs = config['epochs']
batch_size = config['batch_size']
save_step = config['save_step']
glove_path = config['glove_path']
train_res_path = config['train_res_path']

####################################      Main     #################################### 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = MTDataset(df_path, glove_path, mode="train")
    test_set  = MTDataset(df_path, glove_path, mode="test")

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # Encoder and Decoder integrated
    pretrained = True
    sq2sq_model = Seq2Seq(100, train_set.get_output_spec()[0], train_set.get_glove_weights(), 0.3).to(device)
    optimizer = optim.Adam(sq2sq_model.parameters(), lr=0.001) 

    if pretrained: 
        sq2sq_model.load_state_dict(torch.load(f'{train_res_path}/model.pt'))
        optimizer.load_state_dict(torch.load(f'{train_res_path}/optimizer.pt'))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=MTDataset.PAD, reduction='sum', label_smoothing=0.2)

    epchs_mstr_bar = master_bar(range(next_epoch, next_epoch+epochs))
    for epoch in epchs_mstr_bar:
      # Iterate through train data set
      sq2sq_model.train()
      epch_trn_ls = dataset_iter(sq2sq_model, optimizer, criterion, True,
                                 train_dataloader, train_iter, epchs_mstr_bar)
      train_loss.append(epch_trn_ls.item())

      # Iterate through test data set
      sq2sq_model.eval()
      with torch.no_grad():
        epch_vld_ls = dataset_iter(sq2sq_model, optimizer, criterion, True,
                                  test_dataloader, batch_iter, epchs_mstr_bar)
        valid_loss.append(epch_vld_ls.item())
    
      # Finally update learning plot
      plot_loss_update(epoch, next_epoch+epochs, epchs_mstr_bar, train_loss, valid_loss)
    
      if epoch % save_step == 0:
        torch.save(sq2sq_model.state_dict(), f'{train_res_path}/model.pt')
        torch.save(optimizer.state_dict(), f'{train_res_path}/optimizer.pt')
    next_epoch = epoch+1

    train_bleu = calc_bleu(sq2sq_model, train_dataloader, train_set)
    test_bleu = calc_bleu(sq2sq_model, test_dataloader, train_set) 
    print(f'Train BLEU: {train_bleu:.2f}, Test BLEU: {test_bleu:.2f}')

    train_acc = calc_acc(sq2sq_model, train_dataloader)
    test_acc  = calc_acc(sq2sq_model, test_dataloader) 

    print(f'Train Accuracy: {train_acc:.5f}, Test Accuracy: {test_acc:.5f}')


main()