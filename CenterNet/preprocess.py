import os
import re
import torch
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel


## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples


if __name__ == "__main__":
    with torch.no_grad():
        bert_model = "bert-base-uncased"
        max_query_len = 128
        textmodel = BertModel.from_pretrained(bert_model)
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        
        datasets = ("flickr_train", "flickr_val", "flickr_test")
        label_dir  = os.path.join("data", "flickr")
        label_file_path = os.path.join(label_dir, "{}.pth")
        new_label_file_path = os.path.join(label_dir, "{}_" + bert_model + "-" + str(max_query_len) + ".pth")

        for dataset in datasets:
            label_file = label_file_path.format(dataset)
            new_label_file = new_label_file_path.format(dataset)
            images = torch.load(label_file)
            new_images = []
            for i in range(len(images)):
                image_file, bbox, phrase = images[i]

                phrase = phrase.lower()

                ## encode phrase to bert input
                examples = read_examples(phrase, i)
                features = convert_examples_to_features(examples=examples, seq_length=max_query_len, tokenizer=tokenizer)
                word_id = np.array(features[0].input_ids, dtype=int).reshape((1, -1))
                word_mask = np.array(features[0].input_mask, dtype=int).reshape((1, -1))

                word_id = torch.from_numpy(word_id)
                word_mask = torch.from_numpy(word_mask)

                all_encoder_layers, _ = textmodel(word_id, token_type_ids=None, attention_mask=word_mask)
                bert_feature = (all_encoder_layers[-1][:,0,:] + all_encoder_layers[-2][:,0,:] + all_encoder_layers[-3][:,0,:] + all_encoder_layers[-4][:,0,:])/4
                
                bert_feature = bert_feature.data.cpu().numpy().flatten()
                new_images.append((image_file, bbox, bert_feature))
            torch.save(new_images, new_label_file)
