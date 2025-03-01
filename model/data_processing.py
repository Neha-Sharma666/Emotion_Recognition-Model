import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset

# Load Tokenizer Globally to Avoid Re-downloading
MODEL_NAME = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, force_download=True, use_fast=False)

# class EmotionDataset(Dataset):
    # def __init__(self, texts, labels, tokenizer, model, device, label_encoder):
    #     self.texts = texts
    #     self.labels = label_encoder.transform(labels)  # Use the passed label_encoder
    #     self.tokenizer = tokenizer
    #     self.model = model
    #     self.device = device

    # def __len__(self):
    #     return len(self.labels)

    # def __getitem__(self, idx):
    #     text = self.texts[idx]
    #     label = self.labels[idx]

    #     # Tokenize Text
    #     encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    #     input_ids = encoding['input_ids'].squeeze()
    #     attention_mask = encoding['attention_mask'].squeeze()

    #     input_ids = input_ids.to(self.device)
    #     attention_mask = attention_mask.to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
    #         hidden_states = outputs.hidden_states
    #         second_to_last_layer = hidden_states[-2]
    #         sentence_embedding = second_to_last_layer.squeeze().detach().to(self.device)

    #     # Convert Label to Integer
    #     return {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'labels': torch.tensor(int(label), dtype=torch.long, device=self.device),  # Convert label to integer
    #         'embedding': sentence_embedding
    #     }
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model, device, label_encoder):
        self.texts = texts
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        # Ensure labels are transformed into integers
        self.labels = label_encoder.transform(labels) if isinstance(labels[0], str) else labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])  # Ensure label is an integer

        # Tokenize Text
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            hidden_states = outputs.hidden_states
            second_to_last_layer = hidden_states[-2]
            sentence_embedding = second_to_last_layer.squeeze().detach().to(self.device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long, device=self.device),  # Convert label to integer
            'embedding': sentence_embedding
        }

def load_data(file_path, model_name, device):
    df = pd.read_csv(file_path)

    # Encode Labels as Integers before Splitting
    label_encoder = LabelEncoder()
    df['emotion'] = label_encoder.fit_transform(df['emotion'])  # Encode the entire column

    # Now split data after encoding
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['text'].values, df['emotion'].values, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer

# def load_data(file_path, model_name, device):
    # df = pd.read_csv(file_path)

    # # 🔹 Encode Labels as Integers
    # label_encoder = LabelEncoder()
    # df['emotion'] = label_encoder.fit_transform(df['emotion'])  # Encode entire column

    # # Split Data
    # train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['text'].values, df['emotion'].values, test_size=0.2, random_state=42)
    # val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    # return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer
    
if __name__ == "__main__":
    file_path = "data/outputfinal.csv"
    model_name = "ai4bharat/indic-bert"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer = load_data(file_path, model_name, device)
    print("Data loaded successfully!")
