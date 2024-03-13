#DataLoader
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm 
import os

from torch import device 

print('dir is' +str(os.getcwd()))
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

from masking_process import masking
from transformers.models.bert.modeling_bert import BertModel


model = BertModel.from_pretrained('bert-base-uncased')

  

text = "A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"

class Dataset:
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        inputs = self.tokenizer.__call__(text,
                                         None,
                                         add_special_tokens=True,
                                         max_length=self.max_len,
                                         padding="max_length",
                                         truncation=True,
                                         )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        gabriel_mask = masking(text)

        #print(gabriel_mask)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
			"gabriel_mask": torch.tensor(gabriel_mask, dtype=torch.long)
            }
    
	
train_dataset = Dataset(texts=list(text), tokenizer=tokenizer, max_len=512)		
train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)


for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Loading:',disable=True):
	ids = d["ids"].to(device, dtype=torch.long)
	mask = d["mask"].to(device, dtype=torch.long)
	targets = d["labels"].to(device, dtype=torch.long)
	gabriel_mask = d['gabriel_mask'].to(device, dtype=torch.long)
    #outputs = model(ids=ids, mask=mask, gabriel_mask=gabriel_mask) #aggiungere modifiche

    #print(outputs)
