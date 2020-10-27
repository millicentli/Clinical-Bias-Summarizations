# For only training, evaluation, testing BART

import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from transformers import BartModel, AdamW, BatchEncoding
from tqdm import tqdm, trange

# temporary imports - plan to move to a different file
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# Instantiate the writer for TensorBoard (defaults to directory /runs)
writer = SummaryWriter()
# Instantiate cuda
device = torch.device('cuda')

# Dataset for the doc, atten masks, and labels
class Dataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids = ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        _id = self.ids[index]
        _mask = self.masks[index]
        _label = self.labels[index]

        return _id, _mask, _label

class TestDataset(Dataset):
    def __init__(self, ids, masks):
        self.ids = ids
        self.masks = masks

    def __len(self):
        return len(self.ids)

    def __getitem__(self, index):
        _id = self.ids[index]
        _mask = self.masks[index]

        return _id, _mask

def train(model, train_dat, dev_dat, tokenizer):
    raise NotImplemented

def dev(model, dev_dat, tokenizer):
    raise NotImplemented

def evals(model, tokenizer, inputs, doc, summary):
    print("here are the inputs:", inputs)
    #exit()

    dataset = Dataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    #dataset = TestDataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=16)

    total = 0
    correct = 0
    eval_loss = 0.0
    nb_eval_step = 0

    model.to(device)
    model.eval()
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        summary_ids = model.generate(batch[0], num_beams=4, length_penalty=2.0, max_length=142, min_length=56, no_repeat_ngram_size=3)
        summary_generated = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        for num, i in enumerate(summary_generated):
            print(str(num) + " ")
            print(str(i) + "\n")
        summary_gold = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in batch[2]]
        for num, i in enumerate(summary_gold):
            print(str(num) + " ")
            print(str(i) + "\n")
        exit()
    
    raise NotImplemented

def test():
    print("test successful")

if __name__ == "__main__":
    print("currently testing")
    inp = "this is my medical information"
    label = "this is the summary"

    # instantiate
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    inputs = tokenizer([inp], max_length=1024, return_tensors='pt')
    print("here are the inputs:", inputs)

    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
    print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

    # make this inputs for now
    evals(model, inp, tokenizer)
    #evals(model, eval_dat, tokenizer)


