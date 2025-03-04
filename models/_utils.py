from sklearn.utils import sparsefuncs
from scipy.sparse import issparse, csr_matrix
import numpy as np
import os
import torch
import random
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

def complete_masking(batch, p, n_tokens):
    
    padding_token = 1
    cls_token = 3

    indices = batch['tokenized_gene']

    indices = torch.where(indices == 0, torch.tensor(padding_token), indices) # 0 is originally the padding token, we change it to 1
    batch['tokenized_gene'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p) # mask indices with probability p
    spatial_mask = 1 - torch.bernoulli(torch.ones_like(indices), 1)
    
    masked_indices = indices * mask # masked_indices 
    masked_indices = torch.where(indices != padding_token, masked_indices, indices) # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token), mask) # in the model we evaluate the loss of mask position 0
    spatial_mask = torch.where(indices == padding_token, torch.tensor(padding_token), spatial_mask) # in the model we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    
    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices, indices) # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token), mask) # we change the mask so that it doesn't mask any CLS token
    spatial_mask = torch.where(indices == cls_token, torch.tensor(padding_token), spatial_mask) # we change the mask so that it doesn't mask any CLS token
    
    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens)*0.1).type(torch.int64) 

    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices) # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices) # put same tokens just in the previously masked tokens

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask
    batch['spatial_mask'] = spatial_mask
    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch


def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"

def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X



def _sub_tokenize_data(x: csr_matrix, max_seq_len: int = -1, aux_tokens: int = 30):
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    
    for i in range(x.shape[0]):
        start_idx = x.indptr[i]  # Start of the non-zero elements for row i
        end_idx = x.indptr[i + 1]  # End of the non-zero elements for row i
        nonzero_indices = x.indices[start_idx:end_idx]  # Indices of non-zero elements
        nonzero_data = x.data[start_idx:end_idx]  # Values of non-zero elements
        
        # sorted_indices = nonzero_indices[np.argsort(-nonzero_data)][:max_seq_len]
        sorted_idx = np.argsort(-nonzero_data)[:max_seq_len]
        sorted_indices = nonzero_indices[sorted_idx]  # 按排序后的顺序获取索引
        sorted_indices = sorted_indices + aux_tokens  # Adjust for auxiliary tokens
        
        if max_seq_len > 0:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros(x.shape[1], dtype=np.int32)

        # 填充排序后的索引和对应的原始数值
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        
        # 将结果存入最终数组
        scores_final[i, :] = scores
    
    # 返回排序后的索引和对应的原始数值
    return scores_final


def tokenize_data(x: np.array, max_seq_len: int = None, aux_tokens: int = None):
    """Tokenize the input gene vector to a vector of 32-bit integers."""
    if type(x) == np.matrix:
        x = csr_matrix(x)
    scores_final = _sub_tokenize_data(x.tocsr(), max_seq_len, aux_tokens)

    return scores_final.astype('i4')