from fastprogress import master_bar, progress_bar
import torch
import numpy as np


train_loss, valid_loss = [], [] # put them in the upper cell then the lists won't be cleared at every execution
next_epoch = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_iter(model, optimizer, input, target, criterion, calc_loss_inside):
    loss = 0.0

    input = input.to(device)
    target = target.to(device)

    if not calc_loss_inside:
        out_idx, out_mat = model(input, target)
        for word_idx in range(out_mat.shape[1]):
            loss += criterion(out_mat[:, word_idx, :], target[:, word_idx])

    else:
        out_idx, loss = model(input, target, criterion) 

    if torch.isnan(loss):
        loss = torch.tensor(0.0)

    return loss


def train_iter(model, optimizer, batch_input, batch_target, criterion, calc_loss_inside):
    optimizer.zero_grad()

    loss = batch_iter(model, optimizer, batch_input, batch_target, criterion, calc_loss_inside)
    if loss != 0.0:
        loss.backward()
        optimizer.step()

    return loss


def dataset_iter(model, optimizer, criterion, calc_loss_inside, ldr, train_iter, prnt_bar):
    accum_epch_ls = 0
    num_zero_cnt = 0
    for batch_input, batch_target in progress_bar(ldr, parent=prnt_bar):
        epch_ls = train_iter(model, optimizer, batch_input, batch_target, criterion, calc_loss_inside)
        if epch_ls != 0.0:
            num_zero_cnt += 1
            accum_epch_ls += epch_ls
    return accum_epch_ls/num_zero_cnt


def plot_loss_update(epoch, epochs, mb, train_loss, valid_loss):
    x = range(1, epoch+1)
    y = np.concatenate((train_loss, valid_loss))
    graphs = [[x,train_loss], [x,valid_loss]]
    x_margin = 0.2
    y_margin = 0.05
    x_bounds = [1-x_margin, epochs+x_margin]
    y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]
    mb.update_graph(graphs, x_bounds, y_bounds)