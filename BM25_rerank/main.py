from utils.dataload import data_load
from models.model import Model
import easydict
import numpy as np
import torch
from tqdm import tqdm
from torch import nn, optim
from transformers import AdamW, get_linear_schedule_with_warmup

args = easydict.EasyDict({
    'data_path': 'dataset',
    'model_name': 'monologg/koelectra-small-v3-discriminator',
    'batch_size': 16,
    'device': 'cuda:0'
})

data_loader, _ = data_load(args)
model = Model(args).to(args.device)

LM_param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
LM_optimizer_grouped_parameters = [
    {'params': [p for n, p in LM_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in LM_param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(LM_optimizer_grouped_parameters, lr=2e-5)

total_step = len(data_loader) * 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_step*0.1),
                                            num_training_steps=total_step)

for epoch in range(4):
    avg_loss = []
    for features in tqdm(data_loader):
        optimizer.zero_grad()

        label = features['labels'].squeeze(-1).to(args.device)
        output = model(features)

        loss = criterion(output, label)
        avg_loss += [loss.item()]
        loss.backward()

        optimizer.step()
        scheduler.step()

    with open('base_model.txt', 'a') as f:
        f.write(str(np.mean(avg_loss)) + '\n')
    print(np.mean(avg_loss))
    torch.save(model, f'base_model_{epoch}.pt')
