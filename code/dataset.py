from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

class MarkdownDataset(Dataset):

    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, fts, code_sep_token = True, pad_between_code = True, vbl_code = False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts
        self.code_sep_token = code_sep_token
        self.pad_between_code = pad_between_code
        self.vbl_code = vbl_code

    def __getitem__(self, index):
        row = self.df.iloc[index]
        num_samples = self.fts[row.id]["num_samples"]  
        num_sampled = self.fts[row.id]["num_sampled"]
        
        if self.code_sep_token == True:
            add_tokens = True
            if self.vbl_code == True:
                code_max_length = int((self.total_max_len - self.md_max_len)/num_sampled) + 1
            else:
                code_max_length = int((self.total_max_len - self.md_max_len)/num_samples) + 1 
        else:
            add_tokens = False
            if self.vbl_code == True:
                code_max_length = int((self.total_max_len - self.md_max_len - 1)/num_sampled)
            else:
                code_max_length = int((self.total_max_len - self.md_max_len - 1)/num_samples)
            
        if self.pad_between_code == True:
            code_padding = "max_length"
        else:
            code_padding = "do_not_pad"  

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=add_tokens,
            max_length=code_max_length,
            padding=code_padding,
            truncation=True
        )
        
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_code"]
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])
        
        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            if self.code_sep_token == True:
                ids.extend(x[1:])
            else:
                ids.extend(x)
        if self.code_sep_token == False: 
            ids.extend([self.tokenizer.sep_token_id, ]) 
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            if self.code_sep_token == True:
                mask.extend(x[1:])
            else:
                mask.extend(x) 
        if self.code_sep_token == False: 
            mask.extend([self.tokenizer.sep_token_id, ])        
        mask = mask[:self.total_max_len] 
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]
