import argparse
import h5py

import utils.train_bart as bart
import utils.parse_data as parse

from hyperparameters import HYPERPARAMETERS as hyper
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# Runs Standard-BART
# Direct evaluation of BART with the data (MIMIC-CXR and MIMIC-III)

# Params
def main():
    parser = argparse.ArgumentParser()
    
    # Parameters from other files
    # parser.add_argument('--params', type=str, required=True, help="set hyperparameters from other file")

    # Everything else (can be manually set)
    parser.add_argument("--train", action="store_true", help="do some training")
    parser.add_argument("--evals", action="store_true", help="do some evaluation")
    parser.add_argument("--train_data_file", default=None, type=str, required=False, help="file for the training data")
    parser.add_argument("--model_type", default='facebook/bart-large-cnn', type=str, required=False, help="pick a model to be fine-tuned")
    parser.add_argument("--train_batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="batch size for evaluation")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--num_trained_epochs", default=1, type=int, help="number of training epochs")
    parser.add_argument("--grad_clip", default=0.25, type=float, help="value for gradient clipping")
    parser.add_argument("--output_dir", default=None, type=str, required=False, help="directory where models and output will be saved/written to")
    parser.add_argument("--output_file_name_data", default=None, type=str, required=False, help="name for the file where the hidden representation data will be saved to")
    parser.add_argument("--output_file_name_labels", default=None, type=str, required=False, help="name for the file where the hidden representation labels will be saved to")
    args = parser.parse_args()

    args.train = hyper["train"]
    args.evals = hyper["evals"]
    args.train_data_file = hyper["train_data_file"]
    args.model_type = hyper["model_type"]
    args.train_batch_size = hyper["train_batch_size"]
    args.eval_batch_size = hyper["eval_batch_size"]
    args.learning_rate = hyper["learning_rate"]
    args.num_trained_epochs = hyper["num_trained_epochs"]
    args.grad_clip = hyper["grad_clip"]
    args.output_dir = hyper["output_dir"]
    args.output_file_name_data = hyper["output_file_name_data"]
    args.output_file_name_labels = hyper["output_file_name_labels"]

    # First, get the data
    if args.train_data_file is not None:
        doc, summary = parse.clean_cxr(args.train_data_file)

        # Next, encode the data with the tokenizer
        tokenizer = BartTokenizer.from_pretrained(args.model_type)

        # test_doc = "this is a test doc"
        # test_sum = "this is a test sum"
        maxlen = parse.find_longest_text_len(doc)
        # inputs = tokenizer(text=test_doc, max_length=maxlen, truncation=True, padding=True, return_tensors='pt')
        inputs = tokenizer.prepare_seq2seq_batch(doc, summary, max_length=maxlen, truncation=True, padding=True, return_tensors='pt')

        # Instantiate the model
        model = BartForConditionalGeneration.from_pretrained(args.model_type)

        # f_data = h5py.File(args.output_file_name_data, 'w')
        # f_label = h5py.File(args.output_file_name_labels, 'w')

        # Evaluate
        bart.train(model, tokenizer, inputs, args.train_batch_size, 
                args.num_trained_epochs, args.learning_rate, args.grad_clip)
        #bart.evals(model, tokenizer, inputs, args.eval_batch_size, f_data, f_label)
        #bart.test()

if __name__ == "__main__":
    print("Starting Standard-BART evaluation")
    main() 

    '''
    # First, get the data
    dst = "mimic-cxr/files/data.txt"
    doc, summary = parse.clean_cxr(dst)
    # print("size of doc:", len(doc))
    # print("size of summary:", len(summary))

    # Next, encode the data with the tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

    # test_doc = "this is a test doc"
    # test_sum = "this is a test sum"
    maxlen = parse.find_longest_text_len(doc)
    # inputs = tokenizer(text=test_doc, max_length=maxlen, truncation=True, padding=True, return_tensors='pt')
    inputs = tokenizer.prepare_seq2seq_batch(doc, summary, max_length=maxlen, truncation=True, padding=True, return_tensors='pt')

    # Instantiate the model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    f_data = h5py.File(args.output_file_name_data, 'w')
    f_label = h5py.File(args.output_file_name_labels, 'w')

    # Evaluate
    bart.train(model, tokenizer, inputs, args.train_batch_size, 
            args.num_trained_epochs, args.learning_rate, args.grad_clip)
    #bart.evals(model, tokenizer, inputs, args.eval_batch_size, f_data, f_label)
    #bart.test()
    '''
