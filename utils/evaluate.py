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



def sample_next(model, x, prev_state, topk=5, uniform=True):
    """
    Input:
        model: A Pytorch's nn.Module class.
        x: The input to the model.
        prev_state: The previous state of the model.
        topk: The top-k output to sample from. If None, sample from the entire output.
        uniform: Whether to sample from a uniform or a weighted distrubution of topk.
    
    Output:
        sampled_ix: The sampled index.
        state: The current state of the model.
    """
    # Perform forward-prop and get the output of the last time-step
    out, state = model(x, prev_state)
    last_out = out[0, -1, :]

    # Get the top-k indexes and their values
    topk = topk if topk else last_out.shape[0]
    top_logit, top_ix = torch.topk(last_out, k=topk, dim=-1)
    
    # Get the softmax of the topk's and sample
    p = None if uniform else F.softmax(top_logit.detach(), dim=-1).numpy()
    sampled_ix = np.random.choice(top_ix, p=p)
    return sampled_ix, state


def sample(model, seed, topk=5, uniform=True, max_seqlen=18, stop_on=None, batched=True):
    """
    Input:
        model: A Pytorch's nn.Module class.
        seed: List of indexes to intialise model with.
        topk: The top-k output to sample from. If None, sample from the entire output.
        uniform: Whether to sample from a uniform or a weighted distrubution of topk.
        max_seqlen: The maximum sequence length to sample. 'seed' length is included.
        stop_on: Index that signals the end of sequence (sampling).
            If None, max_seqlen determines the end of sampling.
    
    Output:
        sampled_ix_list: List of sampled indexes.
    """
    seed = seed if isinstance(seed, (list, tuple)) else [seed]
    
    model.eval()
    with torch.no_grad():
        sampled_ix_list = seed[:]
        x = (torch.tensor([seed]), torch.tensor([len(seed)]))
        if not batched:
            x = torch.tensor([seed])
        
        prev_state = model.init_state(b_size=1)
        for t in range(max_seqlen - len(seed)):
            sampled_ix, prev_state = sample_next(model, x, prev_state, topk, uniform)

            sampled_ix_list.append(sampled_ix)
            x = (torch.tensor([[sampled_ix]]), torch.tensor([1]))
            if not batched:
                x = torch.tensor([[sampled_ix]])
            
            if sampled_ix==stop_on:
                break
    
    model.train()
    return sampled_ix_list


def originality(n_samp, corpus, sampler, model, ix_list, *args, **kwargs):
    """
    Input:
        n_samp: Number of samples to consider.
        corpus: List of training examples of the format outputted by sampler.
        sampler: A function that returns a list of sampled indexes. Must take in
            model and a (list of) seed as it's first two arguments.
        model: A Pytorch's nn.Module class.
        ix_list: List of seed index, from vocab, to consider.

        *arg: Arguments to pass with sampler.
        **kwargs: Keyword arguments to pass with sampler.
    
    Output:
        samples: List of samples that were considered.
        duplicates: List of samples that are in corpus.
    """
    samples = []
    duplicates = []
    
    for i in range(n_samp):
        samp = sampler(model, random.choice(ix_list), *args, **kwargs)
        samples.append(samp)

        if samp in corpus:
            duplicates.append(samp)
    return samples, duplicates


def keys_to_values(keys, _map, default):
    """
        Converts values in keys to their mapped values in _map. If not found,
        default is used instead.
    """
    return [_map.get(key, default) for key in keys]

