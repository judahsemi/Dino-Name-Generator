import os
import time
import math
import string
import random

import collections
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.functional import F



def test_dataset_batching(x, y, xlen=None):
    """
        Test if x[1:] == y[:-1]. Both x and y are without padding.

    Input:
        x: Batch of x.
        y: Batch of y.
        xlen: True length of each sample in the batch.
        
    Example:
        
        # Assuming dataloader is the same as in the notebook dino-name_batch.ipynb
        >>> for d in dataloader:
        ...     test_dataset_batching(d[0][0], d[1], d[0][1])
    """
    count = 0
    for i, (xi, yi) in enumerate(zip(x, y)):
        xi_nopad = xi[:xlen[i]] if xlen is not None else xi
        yi_nopad = yi[:xlen[i]] if xlen is not None else yi
        
        if (xi_nopad[1:] == yi_nopad[:-1]).all():
            count += 1
    print("{} of {} indexes passed test.".format(count, i+1))


def test_model_batch_vs_nobatch(model, data, pad_ix=-100, e=1e-4):
    """
        Test if model's output, state and loss is the same for batch and nobatch.

    Input:
        model: A Pytorch's nn.Module class.
        data: ((x, xlen), y).
        pad_ix: PAD token index. An error might be raised if it is not given.
        e: Precision of comparision. The default value is recommended.
        
    Example:
        
        # Assuming dataloader is the same as in the notebook dino-name_batch.ipynb
        >>> for d in dataloader:
        ...     test_model_batch_vs_nobatch(model, d, char_to_ix["<PAD>"])
    """
    x, y = data
    n_b, n_s = x[0].shape

    count = {"out": 0, "state": 0, "loss": 0}
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_ix)
    
    # BATCHING: Perform forward-pass with gradient calculation disabled
    model.eval()
    with torch.no_grad():
        batch_out, batch_state = model(x, model.init_state(n_b))
        batch_state = torch.cat(batch_state).transpose(0, 1)
        batch_loss = criterion(batch_out.transpose(1, 2), y)
        
    # NO-BATCHING
    for i in range(n_b):
        xlen_i = int(x[1][i])

        xi = (x[0][i][:xlen_i].unsqueeze(0), x[1][i].unsqueeze(0))
        yi =     y[i][:xlen_i].unsqueeze(0)
        
        # Perform forward-pass with gradient calculation disabled
        model.eval()
        with torch.no_grad():
            nobatch_out_i, nobatch_state_i = model(xi, model.init_state(1))
            nobatch_state_i = torch.cat(nobatch_state_i).transpose(0, 1)
            nobatch_loss_i = criterion(nobatch_out_i.transpose(1, 2), yi)

        # NOTE: Each of the comparision below are calculated upto a certain precision e,
        # due to issues arising from early rounding and floating-point precision

        # check if model outputs are equal
        if ((batch_out[i][:xlen_i] - nobatch_out_i) < e).all():
            count["out"] += 1

        # check if model hidden states are equal
        if ((batch_state[i][:xlen_i] - nobatch_state_i) < e).all():
            count["state"] += 1

        # check if model loss are equal
        if ((batch_loss[i][:xlen_i] - nobatch_loss_i) < e).all():
            count["loss"] += 1

    print("Out: {out}/{n_b} match; State: {state}/{n_b} match; Loss: {loss}/{n_b} match."
        .format(n_b=n_b, **count))

