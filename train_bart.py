# For only training, evaluation, testing BART

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from transformers import BartModel
from tqdm import tqdm, trange
from transformers import AdamW

# instantiate the writer for TensorBoard (defaults to directory /runs)
writer = SummaryWriter()
# instantiate cuda
device = torch.device('cuda')

# do I need the mappings?
def train(model, train_dat, dev_dat, tokenizer):
    raise NotImplemented

def dev(model, dev_dat, tokenizer):
    raise NotImplemented

def eval(model, eval_dat, tokenizer):
    raise NotImplemented
