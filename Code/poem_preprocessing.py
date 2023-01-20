from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import pandas as pd
from tqdm import tqdm, trange
import json
import torch
from torch.utils.data import Dataset, DataLoader


class Poems(Dataset):
    def __init__(self, control_code, truncate=False, max_length=32):

        self.tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
        self.poems = []

        for row in tqdm(train['content']):
            self.poems.append(torch.tensor(
                self.tokenizer.encode(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
            ))
            print(f"<|{control_code}|>{row[:max_length]}<|endoftext|>")
        if truncate:
            self.poems = self.poems[:20000]
        self.poems_count = len(self.poems)

    def __len__(self):
        return self.poems_count

    def __getitem__(self, item):
        return self.poems[item]


def preprocess(file_name):
    with open(file_name, 'r', encoding="UTF-8") as f:
        Lines = f.readlines()
    contents = []
    for line in Lines:
        data = json.loads(line)
        content = data["content"]
        content = content.replace("|", "，", 1)
        content = content.replace("|", "。", 1)
        content = content.replace("|", "，", 1)
        content += "。"
        contents.append(content)
    poems = pd.DataFrame({"content":contents})
    return poems


if __name__ == "__main__":
    train = preprocess('./Dataset/Datasets/CCPC/ccpc_train_v1.0.json')
    test = preprocess('./Dataset/Datasets/CCPC/ccpc_test_v1.0.json')
    train_dataset = Poems(train['content'], truncate=False)
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
    model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")


