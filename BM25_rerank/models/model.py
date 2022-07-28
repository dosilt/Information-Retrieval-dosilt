from transformers import ElectraModel, ElectraTokenizer, ElectraConfig
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.config = ElectraConfig.from_pretrained(args.model_name)
        self.auto_model = ElectraModel.from_pretrained(args.model_name, config=self.config)
        self.tokenizer = ElectraTokenizer.from_pretrained(args.model_name)
        self.hidden_size = self.config.hidden_size
        self.act = nn.Tanh()
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.device = args.device

    def forward(self, features):
        X = self.tokenizer(features['question'], features['context'], padding=True, truncation=True,
                      max_length=512, return_tensors='pt')
        attention_mask = X['attention_mask'].to(self.device)
        input_X = {'input_ids': X['input_ids'].to(self.device),
                   'attention_mask': attention_mask,
                   'token_type_ids': X['token_type_ids'].to(self.device)}
        output = self.auto_model(**input_X)['last_hidden_state'][:, 0]
        # output = torch.mean(self.auto_model(**input_X)['last_hidden_state'], dim=1)
        # output = output / torch.sum(attention_mask, dim=-1).unsqueeze(-1)
        output = self.fc_out(self.act(output)).squeeze(-1)
        return output


if __name__ == '__main__':
    import sys
    sys.path.append('../utils')
    from dataload import data_load
    import easydict

    args = easydict.EasyDict({
        'data_path': '../dataset/train.json',
        'bm25_result': '../train_mrr10.npy',
        'model_name': 'monologg/koelectra-base-v3-discriminator',
        'batch_size': 2,
        'device': 'cuda:0'
    })

    data_loader, _ = data_load(args)
    model = Model(args).cuda()

    for features in data_loader:
        output = model(features)
        print(output.shape)
        break
