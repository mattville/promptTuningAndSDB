from torch.utils.data import Dataset
import json
import random

SEED = 314
LABELS_DICT = {'anti-stereotype': '<antistereo>', 'stereotype': '<stereo>', 'unrelated': '<nonseq>'}


class StereoData(Dataset):
    def __init__(self, path:str, tokenizer):
        random.seed(SEED)

        self.data = json.load(open(path, "r"))

        # Process StereoSet data
        self.X = []
        for i in self.data['data']['intersentence']:
            context = i['context']
            for j in i['sentences']:
                label = j['gold_label']
                completion = j['sentence']
                toAppend = "<startofstring> " + context + " " + LABELS_DICT[label] + ": " + completion + " <endofstring>"
                self.X.append(toAppend)
        random.shuffle(self.X)

        self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])