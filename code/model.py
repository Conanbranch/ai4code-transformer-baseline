import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

class MarkdownModel(nn.Module):
    def __init__(self, model_path, re_init = False, reinit_n_layers = 0):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        #self.config = AutoConfig.from_pretrained(model_path)
        #self.top = nn.Linear(self.config.hidden_size, 1)
        self.top = nn.Linear(768, 1)
        #if reinit_n_layers != 0: 
        if re_init == True: 
            #self.model.pooler.dense.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            #self.model.pooler.dense.bias.data.zero_()
            #for param in self.model.pooler.parameters():
            #    param.requires_grad = True
            self.reinit_n_layers = reinit_n_layers
            for n in range(1, self.reinit_n_layers + 1):
                self.model.encoder.layer[-n].apply(self.model._init_weights)
                
    def forward(self, ids, mask):
        x = self.model(ids, mask)[0]
        x = self.top(x[:, 0, :])
        x = torch.sigmoid(x) 
        return x
