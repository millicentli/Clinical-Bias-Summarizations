# For training BART with select evaluation methods

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

def train(model, tokenizer, inputs, batch_size, num_trained_epochs, learning_rate, grad_clip):
    dataset = Dataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])

    # change this batch_size to args
    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # change this epochs to args
    num_trained_epochs = num_trained_epochs

    global_step = 0
    epochs_trained = 0
    tr_loss = 0

    model.to(device)
    model.zero_grad()

    params = [p for n,p in model.named_parameters()]
    # change this lr to args
    optimizer = AdamW(params, lr=learning_rate)
    train_iterator = trange(epochs_trained, int(num_trained_epochs), desc="Epoch")

    for idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(device) for t in batch)
            inputs = batch[0]
            atten = batch[1]
            labels = batch[2]

            outputs = model(input_ids=inputs, attention_mask=atten, labels=labels)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            # change this grad clipping args.grad_clip to args
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            model.zero_grad()
            global_step += 1

            if (len(epoch_iterator) == step + 1):
                # writer.add_scalar('train loss', tr_loss / global_step, global_step)
                exit()
                # save the model, then run on the development set
                torch.save(model, "models_std/std_pos_model_" + str(idx))
                print(f"Epoch: {epochs_trained}, Step: {step}, Loss: {tr_loss / global_step}")

                # test on dev set
                # dev(model, dev_dat, dev_mappings, tokenizer, idx)

                # load back the original model
                # model = torch.load("pos_model_" + str(idx), map_location="cpu")
                # model.to(device)
        epochs_trained += 1

def dev(model, dev_dat, tokenizer):
    raise NotImplemented

# Evaluation then is twofold: Extract/save representation, then use it within the FC network
# Eval --> Extract the representation from BART --> save it, use it for FC network
def evals(model, tokenizer, inputs, batch_size, data, labels):
    dataset = Dataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    nb_eval_step = 0
    eval_loss = 0.0

    model.to(device)
    model.eval()
    for batch in tqdm(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            inputs = batch[0]
            atten = batch[1]
            # Save this label in the h5py
            labels = batch[2]

            outputs = model(input_ids=inputs, attention_mask=atten)
            tmp_eval_loss, logits = outputs[:2]

            # Get the representations, save as h5py?


'''
def evals(model, tokenizer, inputs, batch_size):
    print("here are the inputs:", inputs)
    #exit()

    dataset = Dataset(inputs['input_ids'], inputs['attention_mask'], inputs['labels'])
    test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

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

'''

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


