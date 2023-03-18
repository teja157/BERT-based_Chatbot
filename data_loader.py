message4:
# train loader, val loader
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer

class CustomTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        targets = torch.tensor(int(target_text), dtype=torch.long)
        return inputs, targets

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare your data
data = [
    ("How are you?", 0),
    ("What's the weather like?", 1),
    ...
]  # List of tuples (input_text, target_text)
train_data_len = int(0.8 * len(data))  # 80% for training
val_data_len = len(data) - train_data_len  # 20% for validation

# Create custom dataset and split into train and validation sets
dataset = CustomTextDataset(data, tokenizer)
train_dataset, val_dataset = random_split(dataset, [train_data_len, val_data_len])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
