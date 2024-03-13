import torch
from torch.utils.data import DataLoader
from torch import device 
from tqdm import tqdm 
from transformers import AutoTokenizer
from masking_process import masking
from transformers.models.bert.modeling_bert import BertModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained('bert-base-uncased')

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
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "gabriel_mask": torch.tensor(gabriel_mask, dtype=torch.long)
        }

def main():
    texts = ["A BERT tokenizer uses something known BERT tokenizer which is BERT case sensitive"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = Dataset(texts=texts, tokenizer=tokenizer, max_len=20)	#mettere text in lista perchè Dataset prende texts (si suppone essere una lista di strionghe)
    print('\n\n')
    # Stampa degli elementi del dataset
    print(train_dataset[0]) #ids
    print(train_dataset[1]) #padding/mask
    print(train_dataset[2]) #gabriel_mask
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    
    for bi, d in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc='Loading:', disable=True):
        ids = d["ids"].to(device, dtype=torch.long,non_blocking=True)
        mask = d["mask"].to(device, dtype=torch.long,non_blocking=True)
        gabriel_mask = d['gabriel_mask'].to(device, dtype=torch.long,non_blocking=True)
        outputs = model(input_ids=ids, attention_mask=mask, gabriel_mask=gabriel_mask) # aggiungere modifiche
        print(outputs)

if __name__ == '__main__':
    main()
