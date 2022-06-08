import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        #self.top = nn.Linear(769, 1)
        self.top = nn.Linear(768, 1)
        #self.reinit_n_layers = 1
        #for n in range(1, self.reinit_n_layers + 1): 
          #self.model.encoder.layer[-n].apply(self.model._init_weights)

    #def forward(self, ids, mask, fts):
    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        #x = torch.cat((x[:, 0, :], fts), 1)
        #x = self.top(x)
        x = self.top(x[:, 0, :])
        return x
