import utils.train_bart as bart
import utils.parse_data as parse

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# Runs Standard-BART
# Direct evaluation of BART with the data (MIMIC-CXR and MIMIC-III)

if __name__ == "__main__":
    print("hello")
    
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
    print("initial doc size:", len(doc))
    print("initial sum size:", len(summary))
    inputs = tokenizer.prepare_seq2seq_batch(doc, summary, max_length=maxlen, truncation=True, padding=True, return_tensors='pt')

    # Instantiate the model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Evaluate
    bart.evals(model, tokenizer, inputs,  doc, summary)
    bart.test()
