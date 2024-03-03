import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
import pandas as pd
import os
from tqdm import tqdm

class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=512):
        # ANSI formatında (ISO-8859-9) dosyayı okuma
        self.data = pd.read_csv(file_path, encoding='ISO-8859-9')
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.data.iloc[idx]['input_text']
        target = self.data.iloc[idx]['target_text']
        source_enc = self.tokenizer(source, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        target_enc = self.tokenizer(target, padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        return source_enc, target_enc

def train(epoch, model, tokenizer, device, loader, optimizer):
    model.train()
    for _, (source_enc, target_enc) in enumerate(tqdm(loader, desc=f'Epoch {epoch} Training')):
        input_ids = source_enc['input_ids'].squeeze().to(device)
        labels = target_enc['input_ids'].squeeze().to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

def main():
    cache_dir = './cache'  
    os.makedirs(cache_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    print(f"Training on: {'GPU' if device.type == 'cuda' else 'CPU'}")

    tokenizer = T5Tokenizer.from_pretrained('t5-small', cache_dir=cache_dir)
    model = T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir=cache_dir).to(device)

    dataset = ParaphraseDataset(tokenizer, 'dataset.csv')
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):  
        train(epoch, model, tokenizer, device, loader, optimizer)

    model.save_pretrained('./model')
    tokenizer.save_pretrained("./model")


if __name__ == '__main__':
    main()
