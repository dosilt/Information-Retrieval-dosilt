from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np
import json
import torch


def data_load(args):
    with open(args.data_path + '/paragraph_context.json', 'r', encoding='utf-8') as f:
        texts = json.load(f)

    with open(args.data_path + '/question_context.json', 'r', encoding='utf-8') as f:
        questions = json.load(f)

    with open(args.data_path + '/train_labels.json', 'r', encoding='utf-8') as f:
        labels_dummy = json.load(f)
        labels = defaultdict(str)

        for i in labels_dummy.keys():
            for qu in labels_dummy[i]:
                labels[qu] = i

    with open(args.data_path + '/train_bm25_space_result.json', 'r', encoding='utf-8') as f:
        bm25_result = json.load(f)

    candidates = []
    for i in bm25_result.keys():
        check = False
        for j in bm25_result[i][:10]:
            if labels[i] == j:
                label = 1
                check = True
            else:
                label = 0
                check = False
            candidates.append([i, j, label])

        if check is False:
            candidates.append([i, labels[i], 1])

    train_dataset = CreateDataset(candidates, texts, questions)
    infer_dataset = InferDataset(candidates, texts, questions)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, infer_loader


class CreateDataset(Dataset):
    def __init__(self, candidates, texts, questions):
        self.candidates = candidates
        self.texts = texts
        self.questions = questions

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, item):
        question, context, label = self.candidates[item]
        return {'question': self.questions[question], 'context': self.texts[context], 'labels': torch.FloatTensor([label])}


class InferDataset(Dataset):
    def __init__(self, candidates, texts, questions):
        self.candidates = candidates
        self.texts = texts
        self.questions = questions

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, item):
        question, context, label = self.candidates[item]
        return {'question': self.questions[question], 'context': self.texts[context], 'labels': torch.FloatTensor([label]),
                'question_id': question, 'context_id': context}


if __name__ == '__main__':
    import easydict
    from transformers import ElectraTokenizer
    args = easydict.EasyDict({
        'data_path': '../dataset',
        'model_name': 'monologg/koelectra-base-v3-discriminator',
        'batch_size': 2
    })

    tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
    data_loader, infer_loader = data_load(args)
    print(len(data_loader))

    for x in data_loader:
        print(x['question'])
        print(x['context'])
        X = tokenizer(x['question'], x['context'], padding=True, truncation=True, max_length=512,
                      return_tensors='pt')
        break