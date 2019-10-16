import os
import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

if __name__ == "__main__":
    bert_model = "bert-base-uncased"
    max_query_len = 128

    datasets = ("flickr_train", "flickr_val", "flickr_test")
    label_dir  = os.path.join("data", "flickr")
    label_file_path = os.path.join(label_dir, "{}.pth")

    for dataset in datasets:
        label_file = label_file_path.format(dataset)
        images = torch.load(label_file)
        for i in range(len(images)):
            print(images[i])
            pass

    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)